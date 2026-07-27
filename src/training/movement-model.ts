// Local-movement learning subsystem — in-process model backend.
//
// This module closes the gap between the capture/dataset pipeline
// (`src/capture`, `src/training/exporter.ts`) and the standing objective's
// parts (c) "post-train a local model to repeat recorded movements" and
// (d) "generalize to perform new but related movements".
//
// The `LocalAppleSiliconTrainingRunner` shells out to MLX/axolotl, which cannot
// run — or be tested — in the cloud. This module instead defines a *pluggable*
// backend interface plus a fully deterministic, dependency-free reference
// implementation (a variable-order Markov model with stupid-backoff) that trains
// and predicts entirely in-process. It lets us validate the recording → dataset
// → train → infer round-trip with synthetic movement streams, and gives a clear
// seam where a real on-device small model can be dropped in later.

import type { TrajectorySpan } from "../capture/trajectory.js";
import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";

/** A single movement token — an opaque, canonical string for one recorded action. */
export type MovementToken = string;

/** Sentinel appended by the default tokenizer to mark a completed movement run. */
export const MOVEMENT_END: MovementToken = "<end>";

/** An ordered sequence of movement tokens derived from one trajectory/replay. */
export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

/** One candidate next-movement with its learned likelihood. */
export type MovementCandidate = {
  token: MovementToken;
  probability: number;
  count: number;
};

/** The model's answer for "what movement comes next after this context?". */
export type MovementPrediction = {
  /** Argmax next token, or undefined if the model has learned nothing. */
  token: MovementToken | undefined;
  /** Probability of `token` within the backoff context that was used. */
  probability: number;
  /**
   * Context length actually consulted after backoff (`order` of the match).
   * Equals the requested context length for an exact recorded match; a smaller
   * value means the model generalized by backing off to a shorter suffix.
   * `-1` means the model is empty.
   */
  order: number;
  /** All candidates for the winning context, sorted best-first (deterministic). */
  candidates: MovementCandidate[];
};

export type TrainMovementModelOptions = {
  /** Highest Markov order (context length) to learn. Default 3. */
  maxOrder?: number;
};

/** A trained model: deterministic next-movement inference + serialization. */
export interface TrainedMovementModel {
  readonly backendId: string;
  readonly maxOrder: number;
  /** Predict the single most likely next movement after `context`. */
  predictNext(context: MovementToken[]): MovementPrediction;
  /**
   * Roll the policy forward from `prefix`, appending predicted movements until
   * `MOVEMENT_END` is produced, prediction is empty, or `maxSteps` is reached.
   * The returned array excludes the prefix and the terminal `MOVEMENT_END`.
   */
  generate(prefix: MovementToken[], maxSteps?: number): MovementToken[];
  /** Plain-object form so the trained policy can be persisted as an artifact. */
  serialize(): SerializedMovementModel;
}

/** Backend contract — swap the Markov reference for a real on-device model. */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementSequence[], options?: TrainMovementModelOptions): Promise<TrainedMovementModel>;
}

export type SerializedMovementModel = {
  version: 1;
  backendId: string;
  maxOrder: number;
  /** order -> contextKey -> token -> count. Order 0 uses the empty-string key. */
  counts: Record<string, Record<string, Record<string, number>>>;
};

const CONTEXT_SEPARATOR = "";

/**
 * Variable-order Markov backend with stupid-backoff.
 *
 * Reproduction (objective c): a context seen verbatim during training resolves
 * at full order, so recorded movements replay exactly.
 * Generalization (objective d): an unseen long context still matches on its
 * longest recorded *suffix* — the model falls back to shorter, shared movement
 * patterns instead of failing, which is how it performs new-but-related runs.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov-backoff";

  async train(
    dataset: MovementSequence[],
    options: TrainMovementModelOptions = {},
  ): Promise<TrainedMovementModel> {
    const maxOrder = Math.max(1, Math.floor(options.maxOrder ?? 3));
    // counts[order][contextKey][token] = frequency
    const counts: CountTable = new Map();
    for (let order = 0; order <= maxOrder; order += 1) {
      counts.set(order, new Map());
    }

    for (const sequence of dataset) {
      const tokens = sequence.tokens;
      for (let i = 0; i < tokens.length; i += 1) {
        const next = tokens[i]!;
        for (let order = 0; order <= maxOrder; order += 1) {
          if (order > i) {
            break; // not enough history for this order at this position
          }
          const context = tokens.slice(i - order, i);
          incrementCount(counts, order, contextKey(context), next);
        }
      }
    }

    return new MarkovMovementModel(this.id, maxOrder, counts);
  }

  /** Rehydrate a persisted model (e.g. a saved training artifact). */
  static deserialize(serialized: SerializedMovementModel): TrainedMovementModel {
    const counts: CountTable = new Map();
    for (const [orderKey, contexts] of Object.entries(serialized.counts)) {
      const order = Number(orderKey);
      const contextMap: ContextTable = new Map();
      for (const [ctx, tokenCounts] of Object.entries(contexts)) {
        contextMap.set(ctx, new Map(Object.entries(tokenCounts)));
      }
      counts.set(order, contextMap);
    }
    for (let order = 0; order <= serialized.maxOrder; order += 1) {
      if (!counts.has(order)) {
        counts.set(order, new Map());
      }
    }
    return new MarkovMovementModel(serialized.backendId, serialized.maxOrder, counts);
  }
}

type TokenCounts = Map<MovementToken, number>;
type ContextTable = Map<string, TokenCounts>;
type CountTable = Map<number, ContextTable>;

class MarkovMovementModel implements TrainedMovementModel {
  constructor(
    readonly backendId: string,
    readonly maxOrder: number,
    private readonly counts: CountTable,
  ) {}

  predictNext(context: MovementToken[]): MovementPrediction {
    const highest = Math.min(this.maxOrder, context.length);
    for (let order = highest; order >= 0; order -= 1) {
      const table = this.counts.get(order);
      if (!table) {
        continue;
      }
      const key = contextKey(context.slice(context.length - order));
      const tokenCounts = table.get(key);
      if (!tokenCounts || tokenCounts.size === 0) {
        continue;
      }
      const candidates = rankCandidates(tokenCounts);
      const best = candidates[0]!;
      return { token: best.token, probability: best.probability, order, candidates };
    }
    return { token: undefined, probability: 0, order: -1, candidates: [] };
  }

  generate(prefix: MovementToken[], maxSteps = 64): MovementToken[] {
    const generated: MovementToken[] = [];
    const context = [...prefix];
    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predictNext(context);
      if (prediction.token === undefined || prediction.token === MOVEMENT_END) {
        break;
      }
      generated.push(prediction.token);
      context.push(prediction.token);
    }
    return generated;
  }

  serialize(): SerializedMovementModel {
    const counts: SerializedMovementModel["counts"] = {};
    for (const [order, table] of this.counts) {
      const contexts: Record<string, Record<string, number>> = {};
      for (const [ctx, tokenCounts] of table) {
        contexts[ctx] = Object.fromEntries(tokenCounts);
      }
      counts[String(order)] = contexts;
    }
    return { version: 1, backendId: this.backendId, maxOrder: this.maxOrder, counts };
  }
}

function contextKey(context: MovementToken[]): string {
  return context.join(CONTEXT_SEPARATOR);
}

function incrementCount(counts: CountTable, order: number, key: string, token: MovementToken): void {
  const table = counts.get(order)!;
  let tokenCounts = table.get(key);
  if (!tokenCounts) {
    tokenCounts = new Map();
    table.set(key, tokenCounts);
  }
  tokenCounts.set(token, (tokenCounts.get(token) ?? 0) + 1);
}

/** Deterministic ranking: higher count first, ties broken lexicographically. */
function rankCandidates(tokenCounts: TokenCounts): MovementCandidate[] {
  let total = 0;
  for (const count of tokenCounts.values()) {
    total += count;
  }
  return [...tokenCounts.entries()]
    .map(([token, count]) => ({ token, count, probability: total === 0 ? 0 : count / total }))
    .sort((a, b) => (b.count !== a.count ? b.count - a.count : a.token < b.token ? -1 : a.token > b.token ? 1 : 0));
}

// ---------------------------------------------------------------------------
// Tokenization — trajectory / replay actions -> movement tokens
// ---------------------------------------------------------------------------

export type MovementTokenizerOptions = {
  /** Append `MOVEMENT_END` so `generate` learns where a run terminates. Default true. */
  appendEnd?: boolean;
  /**
   * When true, keep concrete targets in the token (higher fidelity, less
   * generalization). When false (default) targets are dropped so related runs
   * over different targets share movement structure and back off cleanly.
   */
  includeTargets?: boolean;
};

/** Canonicalize one trajectory action into a movement token. */
export function tokenizeAction(
  action: { tool: string; summary: string; metadata?: Record<string, unknown> },
  options: MovementTokenizerOptions = {},
): MovementToken {
  const metadata = action.metadata ?? {};
  const gesture = typeof metadata.gesture === "string" ? metadata.gesture : undefined;
  const direction = typeof metadata.direction === "string" ? metadata.direction : undefined;
  const target = typeof metadata.target === "string" ? metadata.target : undefined;
  const kind = gesture ?? normalizeVerb(action.summary);
  const parts = [action.tool, kind];
  if (direction) {
    parts.push(direction);
  }
  if (options.includeTargets && target) {
    parts.push(target);
  }
  return parts.filter(Boolean).join("/");
}

/** Turn a captured trajectory span into an ordered movement sequence. */
export function tokenizeTrajectorySpan(
  span: TrajectorySpan,
  options: MovementTokenizerOptions = {},
): MovementSequence {
  const actions = [...span.actions].sort((a, b) => a.ts - b.ts);
  const tokens = actions.map((action) => tokenizeAction(action, options));
  if (options.appendEnd !== false) {
    tokens.push(MOVEMENT_END);
  }
  return { id: span.id, tokens };
}

/** Turn a replay manifest's action timeline into a movement sequence. */
export function tokenizeReplayManifest(
  manifest: ReplayManifest,
  options: MovementTokenizerOptions = {},
): MovementSequence {
  const actions = manifest.events.filter(
    (event): event is Extract<ReplayTimelineEvent, { kind: "action" }> => event.kind === "action",
  );
  const tokens = actions.map((action) => tokenizeAction({ tool: action.tool, summary: action.summary }, options));
  if (options.appendEnd !== false) {
    tokens.push(MOVEMENT_END);
  }
  return { id: manifest.sessionId, tokens };
}

function normalizeVerb(summary: string): string {
  const first = summary.trim().toLowerCase().split(/\s+/)[0] ?? "act";
  return first.replace(/[^a-z0-9-]/g, "") || "act";
}

// ---------------------------------------------------------------------------
// Generalization eval harness
// ---------------------------------------------------------------------------

export type MovementEvalReport = {
  sequenceCount: number;
  /** Positions scored (excludes the first token of each sequence). */
  predictedPositions: number;
  /** Top-1 next-movement accuracy under teacher forcing across all positions. */
  nextTokenAccuracy: number;
  /**
   * Fraction of eval positions the model answered at full `maxOrder` context
   * (exact recorded match) rather than backing off. A proxy for reproduction.
   */
  exactContextRate: number;
  /** Fraction of sequences whose free-run `generate` reproduced them exactly. */
  exactSequenceRate: number;
};

/**
 * Measure replay fidelity on held-out (but related) sequences. Runs teacher-
 * forced next-token prediction for accuracy plus a free-run `generate` per
 * sequence for whole-sequence reproduction — the two facets of objectives c/d.
 */
export function evaluateMovementModel(
  model: TrainedMovementModel,
  heldOut: MovementSequence[],
): MovementEvalReport {
  let predictedPositions = 0;
  let correct = 0;
  let exactContext = 0;
  let exactSequences = 0;

  for (const sequence of heldOut) {
    const tokens = sequence.tokens;
    for (let i = 1; i < tokens.length; i += 1) {
      const context = tokens.slice(0, i);
      const prediction = model.predictNext(context);
      predictedPositions += 1;
      if (prediction.token === tokens[i]) {
        correct += 1;
      }
      if (prediction.order === Math.min(model.maxOrder, context.length)) {
        exactContext += 1;
      }
    }

    if (tokens.length > 0) {
      const prefix = tokens.slice(0, 1);
      const expected = tokens.slice(1, tokens[tokens.length - 1] === MOVEMENT_END ? tokens.length - 1 : tokens.length);
      const generated = model.generate(prefix, tokens.length + 4);
      if (sequencesEqual(generated, expected)) {
        exactSequences += 1;
      }
    }
  }

  return {
    sequenceCount: heldOut.length,
    predictedPositions,
    nextTokenAccuracy: predictedPositions === 0 ? 0 : correct / predictedPositions,
    exactContextRate: predictedPositions === 0 ? 0 : exactContext / predictedPositions,
    exactSequenceRate: heldOut.length === 0 ? 0 : exactSequences / heldOut.length,
  };
}

function sequencesEqual(a: MovementToken[], b: MovementToken[]): boolean {
  if (a.length !== b.length) {
    return false;
  }
  return a.every((token, index) => token === b[index]);
}
