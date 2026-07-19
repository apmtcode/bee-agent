// Pluggable local-movement model backend.
//
// This module is the in-process, cloud-safe counterpart to the on-device
// training runner (`runner.ts`, which shells out to mlx/axolotl on Apple
// silicon). It lets bee-agent actually *learn* from recorded movement
// trajectories and *repeat / generalize* them — objective #2(c) and #2(d) of
// the self-evolution mandate — without any real OS or GPU.
//
// The `MovementModelBackend` interface is the documented seam: the default
// backend is a deterministic Markov n-gram policy (no randomness, no clock, no
// I/O) so cloud/CI tests are reproducible. A real on-device small-model backend
// (e.g. an mlx-lm adapter) can implement the same interface and be swapped in
// via `MovementModelBackendRegistry` with zero call-site changes.

import type { ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/** A single learned/predicted movement, encoded as an opaque token string. */
export type MovementToken = string;

/** One recorded run of movements (e.g. the actions of a single trajectory). */
export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

/** A collection of movement sequences forming a training/eval dataset. */
export type MovementDataset = {
  sequences: MovementSequence[];
};

/** Turns a recorded action into a movement token. Pluggable per dataset. */
export type MovementActionTokenizer = (action: { tool: string; summary: string }) => MovementToken;

export type MovementTrainingConfig = {
  /** Maximum context length (n-gram order) the model conditions on. */
  order: number;
  /** Transitions observed fewer than this many times are ignored at inference. */
  minCount: number;
};

export const DEFAULT_MOVEMENT_TRAINING_CONFIG: MovementTrainingConfig = {
  order: 2,
  minCount: 1,
};

export type MovementPrediction = {
  token: MovementToken;
  probability: number;
  /**
   * How many prior tokens were actually matched to make this prediction.
   * Equal to the model order → the exact recorded context was memorized;
   * lower (backed-off) → the model generalized from a shorter, related context.
   */
  contextOrder: number;
  alternatives: Array<{ token: MovementToken; probability: number }>;
};

export type MovementGenerateOptions = {
  maxSteps?: number;
  stopTokens?: MovementToken[];
};

export type SerializedMovementModel = {
  version: 1;
  backendId: string;
  order: number;
  /**
   * Transition counts keyed by context. The context key is the prior tokens
   * joined by a unit-separator; the empty string is the order-0 (unigram) table.
   */
  transitions: Record<string, Record<MovementToken, number>>;
};

export interface MovementModel {
  readonly backendId: string;
  readonly order: number;
  /** Predict the next movement given the trailing context, or undefined if the model is empty. */
  predictNext(context: MovementToken[]): MovementPrediction | undefined;
  /** Roll the policy forward from a prefix, returning the generated continuation. */
  generate(prefix: MovementToken[], options?: MovementGenerateOptions): MovementToken[];
  toJSON(): SerializedMovementModel;
}

export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, config?: Partial<MovementTrainingConfig>): MovementModel;
}

const CONTEXT_SEPARATOR = "";

function contextKey(context: MovementToken[]): string {
  return context.join(CONTEXT_SEPARATOR);
}

/** Deterministic argmax over a count table: highest count wins, ties break by token asc. */
function rankCounts(counts: Record<MovementToken, number>, minCount: number): Array<{ token: MovementToken; count: number }> {
  return Object.entries(counts)
    .filter(([, count]) => count >= minCount)
    .map(([token, count]) => ({ token, count }))
    .sort((a, b) => (a.count !== b.count ? b.count - a.count : a.token < b.token ? -1 : a.token > b.token ? 1 : 0));
}

class MarkovMovementModel implements MovementModel {
  constructor(
    readonly backendId: string,
    readonly order: number,
    private readonly transitions: Map<string, Map<MovementToken, number>>,
    private readonly minCount: number,
  ) {}

  predictNext(context: MovementToken[]): MovementPrediction | undefined {
    const maxK = Math.min(this.order, context.length);
    for (let k = maxK; k >= 0; k -= 1) {
      const slice = context.slice(context.length - k);
      const table = this.transitions.get(contextKey(slice));
      if (!table) {
        continue;
      }
      const ranked = rankCounts(Object.fromEntries(table), this.minCount);
      if (ranked.length === 0) {
        continue;
      }
      const total = ranked.reduce((sum, entry) => sum + entry.count, 0);
      const best = ranked[0]!;
      return {
        token: best.token,
        probability: best.count / total,
        contextOrder: k,
        alternatives: ranked.slice(1, 4).map((entry) => ({ token: entry.token, probability: entry.count / total })),
      };
    }
    return undefined;
  }

  generate(prefix: MovementToken[], options: MovementGenerateOptions = {}): MovementToken[] {
    const maxSteps = options.maxSteps ?? 64;
    const stop = new Set(options.stopTokens ?? []);
    const running = [...prefix];
    const generated: MovementToken[] = [];
    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predictNext(running);
      if (!prediction || stop.has(prediction.token)) {
        break;
      }
      generated.push(prediction.token);
      running.push(prediction.token);
    }
    return generated;
  }

  toJSON(): SerializedMovementModel {
    const transitions: Record<string, Record<MovementToken, number>> = {};
    for (const [key, table] of this.transitions) {
      transitions[key] = Object.fromEntries(table);
    }
    return { version: 1, backendId: this.backendId, order: this.order, transitions };
  }
}

/**
 * Deterministic Markov n-gram backend. Counts token transitions for every
 * context length 0..order, enabling exact recall of memorized movement chains
 * and Katz-style backoff to shorter, related contexts for generalization.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov-ngram";

  train(dataset: MovementDataset, config: Partial<MovementTrainingConfig> = {}): MovementModel {
    const order = Math.max(0, config.order ?? DEFAULT_MOVEMENT_TRAINING_CONFIG.order);
    const minCount = Math.max(1, config.minCount ?? DEFAULT_MOVEMENT_TRAINING_CONFIG.minCount);
    const transitions = new Map<string, Map<MovementToken, number>>();

    for (const sequence of dataset.sequences) {
      const tokens = sequence.tokens;
      for (let i = 0; i < tokens.length; i += 1) {
        const next = tokens[i]!;
        for (let k = 0; k <= order && k <= i; k += 1) {
          const key = contextKey(tokens.slice(i - k, i));
          let table = transitions.get(key);
          if (!table) {
            table = new Map();
            transitions.set(key, table);
          }
          table.set(next, (table.get(next) ?? 0) + 1);
        }
      }
    }

    return new MarkovMovementModel(this.id, order, transitions, minCount);
  }
}

/** Reconstruct a model from its serialized form (round-trips `model.toJSON()`). */
export function loadMovementModel(serialized: SerializedMovementModel, minCount = 1): MovementModel {
  const transitions = new Map<string, Map<MovementToken, number>>();
  for (const [key, table] of Object.entries(serialized.transitions)) {
    transitions.set(key, new Map(Object.entries(table)));
  }
  return new MarkovMovementModel(serialized.backendId, serialized.order, transitions, Math.max(1, minCount));
}

/** Registry making the model backend pluggable — register a real on-device backend here. */
export class MovementModelBackendRegistry {
  private readonly backends = new Map<string, MovementModelBackend>();

  register(backend: MovementModelBackend): this {
    this.backends.set(backend.id, backend);
    return this;
  }

  get(id: string): MovementModelBackend | undefined {
    return this.backends.get(id);
  }

  list(): string[] {
    return [...this.backends.keys()].sort();
  }
}

export function createDefaultMovementBackendRegistry(): MovementModelBackendRegistry {
  return new MovementModelBackendRegistry().register(new MarkovMovementBackend());
}

/** Default tokenizer: the tool name is the movement class. */
export function defaultMovementTokenizer(action: { tool: string; summary: string }): MovementToken {
  return action.tool;
}

/** Richer tokenizer: `tool:<first-summary-word>` — keeps generalization by tool while distinguishing variants. */
export function detailedMovementTokenizer(action: { tool: string; summary: string }): MovementToken {
  const keyword = action.summary.trim().toLowerCase().split(/\s+/, 1)[0] ?? "";
  return keyword ? `${action.tool}:${keyword}` : action.tool;
}

export function buildMovementSequenceFromActions(
  id: string,
  actions: Array<{ tool: string; summary: string; ts: number }>,
  tokenize: MovementActionTokenizer = defaultMovementTokenizer,
): MovementSequence {
  const tokens = [...actions]
    .sort((a, b) => a.ts - b.ts)
    .map((action) => tokenize({ tool: action.tool, summary: action.summary }));
  return { id, tokens };
}

export function buildMovementDatasetFromTrajectories(
  trajectories: TrajectorySpan[],
  tokenize: MovementActionTokenizer = defaultMovementTokenizer,
): MovementDataset {
  const sequences = trajectories
    .map((trajectory) => buildMovementSequenceFromActions(trajectory.id, trajectory.actions, tokenize))
    .filter((sequence) => sequence.tokens.length > 0);
  return { sequences };
}

export function buildMovementDatasetFromReplays(
  replays: Array<{ events: ReplayTimelineEvent[] }>,
  tokenize: MovementActionTokenizer = defaultMovementTokenizer,
): MovementDataset {
  const byTrajectory = new Map<string, Array<{ tool: string; summary: string; ts: number }>>();
  const order: string[] = [];
  for (const replay of replays) {
    for (const event of replay.events) {
      if (event.kind !== "action") {
        continue;
      }
      let bucket = byTrajectory.get(event.trajectoryId);
      if (!bucket) {
        bucket = [];
        byTrajectory.set(event.trajectoryId, bucket);
        order.push(event.trajectoryId);
      }
      bucket.push({ tool: event.tool, summary: event.summary, ts: event.ts });
    }
  }
  const sequences = order
    .map((trajectoryId) => buildMovementSequenceFromActions(trajectoryId, byTrajectory.get(trajectoryId)!, tokenize))
    .filter((sequence) => sequence.tokens.length > 0);
  return { sequences };
}

export type MovementEvalResult = {
  sequenceCount: number;
  positionCount: number;
  correct: number;
  /** Fraction of positions where the model's top prediction matched the recorded next movement. */
  nextTokenAccuracy: number;
  /** Mean matched context length across predictions — higher = more memorized, lower = more generalized. */
  averageContextOrder: number;
  perSequence: Array<{ id: string; positions: number; correct: number; accuracy: number }>;
};

/**
 * Generalization eval harness: measure next-movement prediction accuracy on
 * held-out (but related) sequences. Use held-out data the model never trained
 * on to distinguish genuine generalization from memorization.
 */
export function evaluateMovementModel(model: MovementModel, heldOut: MovementDataset): MovementEvalResult {
  let positionCount = 0;
  let correct = 0;
  let contextOrderSum = 0;
  const perSequence: MovementEvalResult["perSequence"] = [];

  for (const sequence of heldOut.sequences) {
    let seqPositions = 0;
    let seqCorrect = 0;
    for (let i = 0; i < sequence.tokens.length; i += 1) {
      const prediction = model.predictNext(sequence.tokens.slice(0, i));
      if (!prediction) {
        continue;
      }
      seqPositions += 1;
      positionCount += 1;
      contextOrderSum += prediction.contextOrder;
      if (prediction.token === sequence.tokens[i]) {
        seqCorrect += 1;
        correct += 1;
      }
    }
    perSequence.push({
      id: sequence.id,
      positions: seqPositions,
      correct: seqCorrect,
      accuracy: seqPositions === 0 ? 0 : seqCorrect / seqPositions,
    });
  }

  return {
    sequenceCount: heldOut.sequences.length,
    positionCount,
    correct,
    nextTokenAccuracy: positionCount === 0 ? 0 : correct / positionCount,
    averageContextOrder: positionCount === 0 ? 0 : contextOrderSum / positionCount,
    perSequence,
  };
}
