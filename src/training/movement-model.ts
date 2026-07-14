/**
 * Pluggable local-movement model backend.
 *
 * This module closes objective #2 parts (c) "post-train a local model on the
 * recorded dataset to repeat the movements" and (d) "generalize to perform new
 * but related movements" with an in-process, deterministic implementation that
 * runs anywhere (cloud/CI included) on synthetic data.
 *
 * The real on-device training pipeline (MLX / axolotl launch scripts) lives in
 * `runner.ts` and executes only on the user's machine. This module provides the
 * *backend seam*: a `MovementModelBackend` interface plus a reference n-gram
 * backend. A production backend (a small local transformer, an ONNX policy,
 * etc.) implements the same interface and drops in unchanged — capture →
 * dataset → train → infer/generalize → eval all flow through these types.
 */
import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/** A single canonicalized movement/action step (e.g. a tool invocation). */
export type MovementToken = string;

/** One ordered trajectory of movement tokens. */
export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

/** A replayable, model-trainable dataset of movement sequences. */
export type MovementDataset = {
  sequences: MovementSequence[];
};

export type MovementTokenizerOptions = {
  /**
   * How to fold an action into a token. `"tool"` (default) groups by tool so
   * the model learns tool-level movement grammar and generalizes across
   * summaries; `"tool+summary"` keeps the summary for finer-grained replay.
   */
  granularity?: "tool" | "tool+summary";
};

export type MovementTrainOptions = {
  /** Max n-gram order (context window + 1). Defaults to 3 (trigram). */
  order?: number;
};

export type MovementGenerateOptions = {
  /** Maximum tokens to emit before forcing a stop. Defaults to 64. */
  maxTokens?: number;
  /** Stop generating once the model predicts sequence end. Defaults to true. */
  stopAtEnd?: boolean;
};

export type MovementCandidate = {
  token: MovementToken;
  probability: number;
};

export type MovementPrediction = {
  /** Argmax next token, or undefined when the model has an empty vocabulary. */
  token: MovementToken | undefined;
  probability: number;
  /** The n-gram order actually used after backoff (1 = unigram). */
  order: number;
  /** All candidates at the used order, most probable first. */
  candidates: MovementCandidate[];
};

export type SerializedMovementModel = {
  version: 1;
  backendId: string;
  order: number;
  vocabulary: MovementToken[];
  /** counts[k] maps a context key (k tokens) to token->count. */
  counts: Array<Record<string, Record<MovementToken, number>>>;
};

/** A trained, queryable movement policy. */
export interface TrainedMovementModel {
  readonly backendId: string;
  readonly order: number;
  readonly vocabulary: MovementToken[];
  /** Predict the next movement given a (possibly empty) context. */
  predictNext(context: MovementToken[]): MovementPrediction;
  /** Roll out a movement sequence from an optional prefix (deterministic). */
  generate(prefix?: MovementToken[], options?: MovementGenerateOptions): MovementToken[];
  serialize(): SerializedMovementModel;
}

/** Backend seam — implement this to plug in a real on-device model. */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainOptions): Promise<TrainedMovementModel>;
  restore(model: SerializedMovementModel): TrainedMovementModel;
}

// Sentinels use a control char (U+0001) prefix so they cannot collide with real
// tokens; they are stripped from the exposed vocabulary and generated output.
const START_TOKEN = "\u0001<start>";
const END_TOKEN = "\u0001<end>";
const CONTEXT_SEPARATOR = "\u0001";

function tokenizeAction(action: { tool: string; summary: string }, options: MovementTokenizerOptions): MovementToken {
  const tool = action.tool.trim() || "unknown";
  if (options.granularity === "tool+summary") {
    return `${tool}\u0002${action.summary.trim()}`;
  }
  return tool;
}

/** Build a dataset from raw trajectory spans (actions ordered by timestamp). */
export function datasetFromTrajectories(
  trajectories: TrajectorySpan[],
  options: MovementTokenizerOptions = {},
): MovementDataset {
  const sequences = trajectories
    .map((trajectory) => ({
      id: trajectory.id,
      tokens: [...trajectory.actions]
        .sort((a, b) => a.ts - b.ts)
        .map((action) => tokenizeAction(action, options)),
    }))
    .filter((sequence) => sequence.tokens.length > 0);
  return { sequences };
}

/** Build a dataset from reviewed replay manifests (their action events). */
export function datasetFromReplayManifests(
  manifests: ReplayManifest[],
  options: MovementTokenizerOptions = {},
): MovementDataset {
  const sequences = manifests
    .map((manifest) => ({
      id: manifest.sessionId,
      tokens: manifest.events
        .filter((event): event is Extract<ReplayTimelineEvent, { kind: "action" }> => event.kind === "action")
        .map((event) => tokenizeAction(event, options)),
    }))
    .filter((sequence) => sequence.tokens.length > 0);
  return { sequences };
}

function contextKey(tokens: MovementToken[]): string {
  return tokens.join(CONTEXT_SEPARATOR);
}

function padContext(context: MovementToken[], order: number): MovementToken[] {
  const window = context.slice(-(order - 1));
  if (window.length >= order - 1) {
    return window;
  }
  return [...Array<MovementToken>(order - 1 - window.length).fill(START_TOKEN), ...window];
}

function compareTokens(a: MovementToken, b: MovementToken): number {
  return a < b ? -1 : a > b ? 1 : 0;
}

/**
 * Deterministic n-gram movement backend with stupid-backoff.
 *
 * - Repeats recorded movements: high-order n-grams reproduce trained sequences.
 * - Generalizes to related movements: an unseen context backs off to shorter
 *   contexts (down to the unigram prior), yielding a plausible continuation for
 *   novel-but-related prefixes rather than failing.
 * - Fully deterministic (argmax with lexicographic tie-break) so replay and
 *   tests are reproducible.
 */
export class NGramMovementBackend implements MovementModelBackend {
  readonly id = "ngram";

  async train(dataset: MovementDataset, options: MovementTrainOptions = {}): Promise<TrainedMovementModel> {
    const order = Math.max(1, Math.floor(options.order ?? 3));
    // counts[k] holds contexts of length k (k = 0..order-1).
    const counts: Array<Map<string, Map<MovementToken, number>>> = Array.from(
      { length: order },
      () => new Map<string, Map<MovementToken, number>>(),
    );
    const vocabulary = new Set<MovementToken>();

    for (const sequence of dataset.sequences) {
      const padded = [
        ...Array<MovementToken>(order - 1).fill(START_TOKEN),
        ...sequence.tokens,
        END_TOKEN,
      ];
      for (const token of sequence.tokens) {
        vocabulary.add(token);
      }
      for (let i = order - 1; i < padded.length; i += 1) {
        const target = padded[i]!;
        for (let k = 0; k < order; k += 1) {
          const context = padded.slice(i - k, i);
          const key = contextKey(context);
          const table = counts[k]!;
          const row = table.get(key) ?? new Map<MovementToken, number>();
          row.set(target, (row.get(target) ?? 0) + 1);
          table.set(key, row);
        }
      }
    }

    return new NGramMovementModel(this.id, order, counts, [...vocabulary].sort());
  }

  restore(model: SerializedMovementModel): TrainedMovementModel {
    const counts: Array<Map<string, Map<MovementToken, number>>> = model.counts.map((table) => {
      const map = new Map<string, Map<MovementToken, number>>();
      for (const [context, row] of Object.entries(table)) {
        map.set(context, new Map(Object.entries(row)));
      }
      return map;
    });
    return new NGramMovementModel(model.backendId, model.order, counts, [...model.vocabulary]);
  }
}

class NGramMovementModel implements TrainedMovementModel {
  constructor(
    readonly backendId: string,
    readonly order: number,
    private readonly counts: Array<Map<string, Map<MovementToken, number>>>,
    readonly vocabulary: MovementToken[],
  ) {}

  predictNext(context: MovementToken[]): MovementPrediction {
    const padded = padContext(context, this.order);

    for (let k = Math.min(this.order - 1, padded.length); k >= 0; k -= 1) {
      const key = contextKey(padded.slice(padded.length - k));
      const row = this.counts[k]?.get(key);
      if (!row || row.size === 0) {
        continue;
      }
      const total = [...row.values()].reduce((sum, value) => sum + value, 0);
      const candidates = [...row.entries()]
        .filter(([token]) => token !== START_TOKEN)
        .map(([token, count]) => ({ token, probability: count / total }))
        .sort((a, b) => (b.probability - a.probability) || compareTokens(a.token, b.token));
      if (candidates.length === 0) {
        continue;
      }
      const best = candidates[0]!;
      return {
        token: best.token === END_TOKEN ? undefined : best.token,
        probability: best.probability,
        order: k + 1,
        candidates,
      };
    }

    return { token: undefined, probability: 0, order: 0, candidates: [] };
  }

  generate(prefix: MovementToken[] = [], options: MovementGenerateOptions = {}): MovementToken[] {
    const maxTokens = options.maxTokens ?? 64;
    const stopAtEnd = options.stopAtEnd ?? true;
    const emitted = [...prefix];
    const generated: MovementToken[] = [];

    while (generated.length < maxTokens) {
      const next = this.rawPredict(emitted.slice(-(this.order - 1)));
      if (next === undefined) {
        break;
      }
      if (next === END_TOKEN) {
        if (stopAtEnd) {
          break;
        }
        // stopAtEnd disabled: nothing else to emit deterministically.
        break;
      }
      generated.push(next);
      emitted.push(next);
    }
    return generated;
  }

  // Argmax including the END sentinel so generation can terminate naturally.
  private rawPredict(context: MovementToken[]): MovementToken | undefined {
    const padded = padContext(context, this.order);
    for (let k = Math.min(this.order - 1, padded.length); k >= 0; k -= 1) {
      const row = this.counts[k]?.get(contextKey(padded.slice(padded.length - k)));
      if (!row || row.size === 0) {
        continue;
      }
      let best: MovementToken | undefined;
      let bestCount = -1;
      for (const [token, count] of row.entries()) {
        if (token === START_TOKEN) {
          continue;
        }
        if (count > bestCount || (count === bestCount && best !== undefined && compareTokens(token, best) < 0)) {
          best = token;
          bestCount = count;
        }
      }
      if (best !== undefined) {
        return best;
      }
    }
    return undefined;
  }

  serialize(): SerializedMovementModel {
    return {
      version: 1,
      backendId: this.backendId,
      order: this.order,
      vocabulary: [...this.vocabulary],
      counts: this.counts.map((table) => {
        const record: Record<string, Record<MovementToken, number>> = {};
        for (const [context, row] of table.entries()) {
          record[context] = Object.fromEntries(row);
        }
        return record;
      }),
    };
  }
}

export type ReplayFidelityReport = {
  /** Fraction of next-token predictions that matched the held-out sequence. */
  tokenAccuracy: number;
  /** Fraction of sequences reproduced exactly by greedy generation. */
  sequenceExactMatch: number;
  /** Total next-token predictions scored. */
  predictions: number;
  /** Number of sequences evaluated (non-empty). */
  sequences: number;
};

/**
 * Generalization eval harness: score a trained model on held-out (but related)
 * movement sequences. `tokenAccuracy` measures how well the policy predicts the
 * next movement from a growing prefix; `sequenceExactMatch` measures whether a
 * cold-start greedy rollout reproduces the whole sequence.
 */
export function evaluateReplayFidelity(
  model: TrainedMovementModel,
  heldOut: MovementSequence[],
): ReplayFidelityReport {
  let correct = 0;
  let predictions = 0;
  let exactMatches = 0;
  let evaluated = 0;

  for (const sequence of heldOut) {
    if (sequence.tokens.length === 0) {
      continue;
    }
    evaluated += 1;
    for (let i = 0; i < sequence.tokens.length; i += 1) {
      const prediction = model.predictNext(sequence.tokens.slice(0, i));
      predictions += 1;
      if (prediction.token === sequence.tokens[i]) {
        correct += 1;
      }
    }
    const rollout = model.generate([], { maxTokens: sequence.tokens.length });
    if (rollout.length === sequence.tokens.length && rollout.every((token, index) => token === sequence.tokens[index])) {
      exactMatches += 1;
    }
  }

  return {
    tokenAccuracy: predictions === 0 ? 0 : correct / predictions,
    sequenceExactMatch: evaluated === 0 ? 0 : exactMatches / evaluated,
    predictions,
    sequences: evaluated,
  };
}
