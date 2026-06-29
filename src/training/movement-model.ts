/**
 * Local-movement learning subsystem — pluggable model backend.
 *
 * Standing objective #2 (local-movement learning): bee-agent must be able to
 * post-train a local model on recorded movements and generalize to related
 * movements. The existing {@link LocalAppleSiliconTrainingRunner} only emits
 * launch artifacts for real on-device training; it cannot run anything in the
 * cloud. This module adds a *pluggable, backend-agnostic* training+inference
 * seam plus a deterministic in-process backend so the whole
 * capture → dataset → train → infer loop is exercisable in CI without an OS,
 * a GPU, or a real model.
 *
 * The shipped {@link NGramMovementBackend} is an order-N Markov model with
 * stupid-backoff. It "repeats recorded movements" exactly (an exact prefix
 * reproduces the recorded continuation) and "generalizes to related
 * movements" because shared sub-sequences across different trajectories bridge
 * contexts (A→B→C and X→B→D let an A→B prefix still continue to C, and the
 * backoff lets an unseen high-order context fall back to what was learned at
 * lower order). A real on-device small model can be dropped in behind the same
 * {@link LocalMovementModelBackend} interface.
 */

import type { ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/** The atomic unit a movement model learns to predict. */
export type MovementToken = {
  /** The actuator that produced the movement (tool / device / UI surface). */
  tool: string;
  /** A compact, replayable description of the movement. */
  summary: string;
};

/** An ordered run of movements captured from a single trajectory. */
export type MovementSequence = {
  trajectoryId: string;
  tokens: MovementToken[];
};

/** A training corpus of movement sequences. */
export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

/** A single next-movement prediction with the evidence behind it. */
export type MovementPrediction = {
  token: MovementToken;
  /** Conditional probability within the backed-off context. */
  probability: number;
  /** Number of observations supporting this prediction. */
  support: number;
  /** Markov order actually used after backoff (0 = unigram prior). */
  order: number;
};

/** A trained model that can predict and roll out movements. */
export interface TrainedMovementModel {
  readonly backendId: string;
  readonly order: number;
  /** Predict the next movement given a context of preceding movements. */
  predict(context: MovementToken[]): MovementPrediction | undefined;
  /**
   * Greedily roll out a continuation after `prefix`, stopping at `maxSteps`
   * or when no movement can be predicted. Deterministic.
   */
  generate(prefix: MovementToken[], maxSteps: number): MovementToken[];
  /** A serialisable snapshot for persistence / inspection. */
  snapshot(): MovementModelSnapshot;
}

/** Options accepted by a backend's `train`. */
export type MovementTrainOptions = {
  /** Markov order for n-gram-style backends (ignored by others). */
  order?: number;
};

/** A backend turns a dataset into a {@link TrainedMovementModel}. */
export interface LocalMovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainOptions): TrainedMovementModel;
}

/** Serialised form of a trained n-gram model. */
export type MovementModelSnapshot = {
  version: 1;
  backendId: string;
  order: number;
  /** context-key → (token-key → count). The empty string is the unigram bucket. */
  transitions: Record<string, Record<string, number>>;
  /** token-key → token, so snapshots are self-describing. */
  vocabulary: Record<string, MovementToken>;
};

const KEY_SEPARATOR = "\u0000";
const CONTEXT_SEPARATOR = "\u0001";
/**
 * Internal end-of-sequence sentinel — the "movement" that follows the last
 * real movement of every training sequence, so the model learns where
 * recordings stop. Never surfaced as a {@link MovementToken}: when the
 * backed-off argmax is the sentinel, {@link TrainedMovementModel.predict}
 * returns `undefined` (no next movement) and `generate` halts. The
 * control-byte key cannot collide with a real token key (`tool<sep>summary`).
 */
const END_OF_SEQUENCE = "\u0002";

/** Stable, collision-resistant key for a movement token. */
export function movementTokenKey(token: MovementToken): string {
  return `${token.tool}${KEY_SEPARATOR}${token.summary}`;
}

/** Whether two movement tokens describe the same movement. */
export function movementTokensEqual(a: MovementToken, b: MovementToken): boolean {
  return a.tool === b.tool && a.summary === b.summary;
}

function contextKey(tokens: MovementToken[]): string {
  return tokens.map(movementTokenKey).join(CONTEXT_SEPARATOR);
}

/**
 * Order-N Markov backend with stupid-backoff. Deterministic: ties are broken
 * by descending count then ascending token key, so the same dataset always
 * yields the same model and the same predictions.
 */
export class NGramMovementBackend implements LocalMovementModelBackend {
  readonly id = "ngram";

  constructor(private readonly defaultOrder = 2) {}

  train(dataset: MovementDataset, options?: MovementTrainOptions): TrainedMovementModel {
    const order = Math.max(1, Math.floor(options?.order ?? this.defaultOrder));
    const transitions = new Map<string, Map<string, number>>();
    const vocabulary = new Map<string, MovementToken>();

    const bump = (ctxKey: string, tokenKey: string) => {
      let bucket = transitions.get(ctxKey);
      if (!bucket) {
        bucket = new Map<string, number>();
        transitions.set(ctxKey, bucket);
      }
      bucket.set(tokenKey, (bucket.get(tokenKey) ?? 0) + 1);
    };

    for (const sequence of dataset.sequences) {
      const tokens = sequence.tokens;
      for (let i = 0; i < tokens.length; i += 1) {
        const next = tokens[i];
        const nextKey = movementTokenKey(next);
        vocabulary.set(nextKey, next);
        // Record this transition at every order from 0..order so the model can
        // back off from the longest available context down to the unigram prior.
        for (let k = 0; k <= order; k += 1) {
          if (i - k < 0) {
            break;
          }
          const ctx = tokens.slice(i - k, i);
          bump(contextKey(ctx), nextKey);
        }
      }
      // Teach the model where this recording ends: the sentinel follows the last
      // real movement, at every backed-off context length.
      for (let k = 0; k <= order; k += 1) {
        const start = tokens.length - k;
        if (start < 0) {
          break;
        }
        bump(contextKey(tokens.slice(start, tokens.length)), END_OF_SEQUENCE);
      }
    }

    return new NGramMovementModel(this.id, order, transitions, vocabulary);
  }
}

class NGramMovementModel implements TrainedMovementModel {
  constructor(
    readonly backendId: string,
    readonly order: number,
    private readonly transitions: Map<string, Map<string, number>>,
    private readonly vocabulary: Map<string, MovementToken>,
  ) {}

  predict(context: MovementToken[]): MovementPrediction | undefined {
    const maxK = Math.min(this.order, context.length);
    for (let k = maxK; k >= 0; k -= 1) {
      const ctx = context.slice(context.length - k, context.length);
      const bucket = this.transitions.get(contextKey(ctx));
      if (!bucket || bucket.size === 0) {
        continue;
      }
      const best = pickBest(bucket);
      // The evidence at this context says the recording ends here.
      if (best.key === END_OF_SEQUENCE) {
        return undefined;
      }
      const total = [...bucket.values()].reduce((sum, count) => sum + count, 0);
      const token = this.vocabulary.get(best.key);
      if (!token) {
        continue;
      }
      return {
        token,
        probability: total > 0 ? best.count / total : 0,
        support: best.count,
        order: k,
      };
    }
    return undefined;
  }

  generate(prefix: MovementToken[], maxSteps: number): MovementToken[] {
    const out: MovementToken[] = [];
    const context = [...prefix];
    const steps = Math.max(0, Math.floor(maxSteps));
    for (let i = 0; i < steps; i += 1) {
      const prediction = this.predict(context);
      if (!prediction) {
        break;
      }
      out.push(prediction.token);
      context.push(prediction.token);
    }
    return out;
  }

  snapshot(): MovementModelSnapshot {
    const transitions: Record<string, Record<string, number>> = {};
    for (const [ctxKey, bucket] of this.transitions) {
      transitions[ctxKey] = Object.fromEntries(bucket);
    }
    return {
      version: 1,
      backendId: this.backendId,
      order: this.order,
      transitions,
      vocabulary: Object.fromEntries(this.vocabulary),
    };
  }
}

function pickBest(bucket: Map<string, number>): { key: string; count: number } {
  let bestKey = "";
  let bestCount = -1;
  for (const [key, count] of bucket) {
    if (count > bestCount || (count === bestCount && key < bestKey)) {
      bestKey = key;
      bestCount = count;
    }
  }
  return { key: bestKey, count: bestCount };
}

/** Registry of named backends so the training pipeline can stay pluggable. */
export class MovementModelBackendRegistry {
  private readonly backends = new Map<string, LocalMovementModelBackend>();

  register(backend: LocalMovementModelBackend): this {
    this.backends.set(backend.id, backend);
    return this;
  }

  get(id: string): LocalMovementModelBackend | undefined {
    return this.backends.get(id);
  }

  list(): string[] {
    return [...this.backends.keys()].sort();
  }

  train(id: string, dataset: MovementDataset, options?: MovementTrainOptions): TrainedMovementModel {
    const backend = this.backends.get(id);
    if (!backend) {
      throw new Error(`unknown movement model backend: ${id}`);
    }
    return backend.train(dataset, options);
  }
}

/** A registry pre-seeded with the in-process deterministic backend. */
export function createDefaultMovementBackendRegistry(): MovementModelBackendRegistry {
  return new MovementModelBackendRegistry().register(new NGramMovementBackend());
}

/** Build a training corpus from captured trajectory spans (actions only). */
export function buildMovementDataset(trajectories: TrajectorySpan[]): MovementDataset {
  return {
    version: 1,
    sequences: trajectories
      .map((trajectory) => ({
        trajectoryId: trajectory.id,
        tokens: [...trajectory.actions]
          .sort((a, b) => a.ts - b.ts)
          .map((action) => ({ tool: action.tool, summary: action.summary })),
      }))
      .filter((sequence) => sequence.tokens.length > 0),
  };
}

/** Extract a movement sequence from a replay manifest's timeline events. */
export function movementSequenceFromReplayEvents(
  trajectoryId: string,
  events: ReplayTimelineEvent[],
): MovementSequence {
  return {
    trajectoryId,
    tokens: events
      .filter((event): event is Extract<ReplayTimelineEvent, { kind: "action" }> => event.kind === "action")
      .map((event) => ({ tool: event.tool, summary: event.summary })),
  };
}
