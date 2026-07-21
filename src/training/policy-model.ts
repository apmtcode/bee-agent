import type { ReplayTimelineEvent } from "../capture/replay.js";

/**
 * Movement policy model
 * ---------------------
 * The capture pipeline records local movements/actions into trajectories, the
 * replay layer flattens them into an ordered {@link ReplayTimelineEvent}
 * timeline, and the exporter packages reviewed timelines for training. This
 * module is the missing piece (objective #2c/#2d): a small, *trainable* local
 * model that learns from that movement dataset so it can (c) repeat recorded
 * movements and (d) generalize to new-but-related movements.
 *
 * The model backend is pluggable via {@link MovementPolicyBackend}. The bundled
 * {@link NgramMovementPolicyBackend} is a deterministic, dependency-free
 * back-off n-gram — it trains and predicts entirely in-process so cloud/CI
 * tests validate the full capture → dataset → train → rollout loop without any
 * OS access or native ML runtime. A real on-device small model (e.g. an MLX or
 * llama.cpp backend) implements the same interface and slots in unchanged.
 */

/** A discrete movement token derived from a captured action event. */
export type MovementToken = string;

/** Sentinel prepended to every training sequence to model its first move. */
export const SEQUENCE_START: MovementToken = "start";
/** Sentinel appended to every training sequence to model termination. */
export const SEQUENCE_END: MovementToken = "end";

const CONTEXT_SEPARATOR = "";

/** One trajectory's ordered movement tokens. */
export type MovementSequence = {
  trajectoryId: string;
  tokens: MovementToken[];
};

/** A training dataset: a set of tokenized movement sequences + vocabulary. */
export type MovementPolicyDataset = {
  version: 1;
  sequences: MovementSequence[];
  vocabulary: MovementToken[];
};

/** Anything with an ordered event timeline (a {@link ReplayManifest} qualifies). */
export type MovementTimelineSource = {
  events: ReplayTimelineEvent[];
};

export type MovementPolicyTrainConfig = {
  /** Maximum context length (n-gram order). Default 2. */
  order: number;
  /** Additive (Laplace) smoothing applied over the vocabulary. Default 0. */
  smoothing: number;
};

const DEFAULT_TRAIN_CONFIG: MovementPolicyTrainConfig = { order: 2, smoothing: 0 };

/** A single next-move prediction with its full distribution + back-off depth. */
export type MovementPrediction = {
  token: MovementToken;
  probability: number;
  /** How many context tokens were actually used (back-off depth, 0..order). */
  contextOrder: number;
  distribution: Array<{ token: MovementToken; probability: number }>;
};

/** A trained, queryable movement policy. */
export interface TrainedMovementPolicy {
  readonly backendId: string;
  readonly order: number;
  readonly vocabulary: MovementToken[];
  /** Predict the next movement token given the preceding tokens. */
  predict(context: MovementToken[]): MovementPrediction;
  /** Serialize to a plain JSON structure for persistence. */
  serialize(): SerializedMovementPolicy;
}

/** Pluggable seam: swap the n-gram mock for a real on-device model. */
export interface MovementPolicyBackend {
  readonly id: string;
  train(
    dataset: MovementPolicyDataset,
    config?: Partial<MovementPolicyTrainConfig>,
  ): Promise<TrainedMovementPolicy>;
}

export type SerializedMovementPolicy = {
  version: 1;
  backendId: string;
  order: number;
  smoothing: number;
  vocabulary: MovementToken[];
  /** counts[k] = list of [contextKey, [[token, count], ...]] for k-token contexts. */
  counts: Array<Array<[string, Array<[MovementToken, number]>]>>;
};

/** Normalize a free-form summary into a stable movement token fragment. */
function normalizeSummary(summary: string): string {
  return summary.trim().toLowerCase().replace(/\s+/g, " ").replace(/[.!?]+$/g, "");
}

/**
 * Tokenize a single action timeline event into a movement token. Returns
 * `undefined` for non-action events (observations/transcript are context, not
 * movements the policy reproduces).
 */
export function tokenizeMovementEvent(event: ReplayTimelineEvent): MovementToken | undefined {
  if (event.kind !== "action") {
    return undefined;
  }
  return `${event.tool}::${normalizeSummary(event.summary)}`;
}

/**
 * Build a movement dataset from replay timelines. Action events are grouped by
 * their originating trajectory and ordered by timestamp so each trajectory
 * becomes one movement sequence.
 */
export function buildMovementDataset(sources: MovementTimelineSource[]): MovementPolicyDataset {
  const byTrajectory = new Map<string, Array<{ ts: number; token: MovementToken }>>();
  const order: string[] = [];

  for (const source of sources) {
    for (const event of source.events) {
      if (event.kind !== "action") {
        continue;
      }
      const token = tokenizeMovementEvent(event);
      if (token === undefined) {
        continue;
      }
      let bucket = byTrajectory.get(event.trajectoryId);
      if (!bucket) {
        bucket = [];
        byTrajectory.set(event.trajectoryId, bucket);
        order.push(event.trajectoryId);
      }
      bucket.push({ ts: event.ts, token });
    }
  }

  const sequences: MovementSequence[] = order.map((trajectoryId) => {
    const bucket = byTrajectory.get(trajectoryId) ?? [];
    const sorted = [...bucket].sort((a, b) => a.ts - b.ts);
    return { trajectoryId, tokens: sorted.map((entry) => entry.token) };
  });

  const vocabulary = collectVocabulary(sequences);
  return { version: 1, sequences, vocabulary };
}

function collectVocabulary(sequences: MovementSequence[]): MovementToken[] {
  const vocab = new Set<MovementToken>();
  for (const sequence of sequences) {
    for (const token of sequence.tokens) {
      vocab.add(token);
    }
  }
  return [...vocab].sort();
}

type ContextCounts = Map<string, Map<MovementToken, number>>;

class NgramMovementPolicy implements TrainedMovementPolicy {
  constructor(
    public readonly backendId: string,
    public readonly order: number,
    public readonly vocabulary: MovementToken[],
    private readonly smoothing: number,
    private readonly counts: ContextCounts[],
  ) {}

  predict(context: MovementToken[]): MovementPrediction {
    if (this.vocabulary.length === 0) {
      throw new Error("movement policy has no training data");
    }
    const effective = padContext(context, this.order);
    for (let k = this.order; k >= 0; k -= 1) {
      const key = effective.slice(this.order - k).join(CONTEXT_SEPARATOR);
      const nextCounts = this.counts[k]?.get(key);
      if (nextCounts && nextCounts.size > 0) {
        return this.distribution(nextCounts, k);
      }
    }
    // Only reachable with an empty model, which we already guarded above.
    return this.distribution(new Map([[SEQUENCE_END, 1]]), 0);
  }

  private distribution(nextCounts: Map<MovementToken, number>, contextOrder: number): MovementPrediction {
    let total = 0;
    for (const value of nextCounts.values()) {
      total += value;
    }
    const denominator = total + this.smoothing * this.vocabulary.length;
    const distribution = this.vocabulary
      .map((token) => {
        const count = nextCounts.get(token) ?? 0;
        const probability = denominator === 0 ? 0 : (count + this.smoothing) / denominator;
        return { token, probability };
      })
      .sort((a, b) => (b.probability !== a.probability ? b.probability - a.probability : a.token < b.token ? -1 : 1));
    const best = distribution[0] ?? { token: SEQUENCE_END, probability: 1 };
    return { token: best.token, probability: best.probability, contextOrder, distribution };
  }

  serialize(): SerializedMovementPolicy {
    return {
      version: 1,
      backendId: this.backendId,
      order: this.order,
      smoothing: this.smoothing,
      vocabulary: [...this.vocabulary],
      counts: this.counts.map((level) =>
        [...level.entries()].map(([contextKey, nextCounts]) => [contextKey, [...nextCounts.entries()]] as [
          string,
          Array<[MovementToken, number]>,
        ]),
      ),
    };
  }
}

/** Left-pad a context with SEQUENCE_START and trim to exactly `order` tokens. */
function padContext(context: MovementToken[], order: number): MovementToken[] {
  const trimmed = context.slice(-order);
  const padCount = Math.max(0, order - trimmed.length);
  return [...new Array<MovementToken>(padCount).fill(SEQUENCE_START), ...trimmed];
}

/** Deterministic back-off n-gram backend. Trains in-process; no OS/native deps. */
export class NgramMovementPolicyBackend implements MovementPolicyBackend {
  public readonly id = "ngram-backoff";

  async train(
    dataset: MovementPolicyDataset,
    config: Partial<MovementPolicyTrainConfig> = {},
  ): Promise<TrainedMovementPolicy> {
    const order = Math.max(0, Math.floor(config.order ?? DEFAULT_TRAIN_CONFIG.order));
    const smoothing = Math.max(0, config.smoothing ?? DEFAULT_TRAIN_CONFIG.smoothing);
    const counts: ContextCounts[] = Array.from({ length: order + 1 }, () => new Map());
    const vocabulary = new Set<MovementToken>();

    for (const sequence of dataset.sequences) {
      const padded = [
        ...new Array<MovementToken>(order).fill(SEQUENCE_START),
        ...sequence.tokens,
        SEQUENCE_END,
      ];
      for (let i = order; i < padded.length; i += 1) {
        const next = padded[i];
        vocabulary.add(next);
        for (let k = 0; k <= order; k += 1) {
          const key = padded.slice(i - k, i).join(CONTEXT_SEPARATOR);
          const level = counts[k];
          let bucket = level.get(key);
          if (!bucket) {
            bucket = new Map();
            level.set(key, bucket);
          }
          bucket.set(next, (bucket.get(next) ?? 0) + 1);
        }
      }
    }

    return new NgramMovementPolicy(this.id, order, [...vocabulary].sort(), smoothing, counts);
  }
}

/** Restore a policy previously produced by {@link TrainedMovementPolicy.serialize}. */
export function deserializeMovementPolicy(data: SerializedMovementPolicy): TrainedMovementPolicy {
  const counts: ContextCounts[] = data.counts.map((level) => {
    const map: ContextCounts = new Map();
    for (const [contextKey, nextCounts] of level) {
      map.set(contextKey, new Map(nextCounts));
    }
    return map;
  });
  return new NgramMovementPolicy(data.backendId, data.order, [...data.vocabulary], data.smoothing, counts);
}

export type RolloutOptions = {
  /** Movement tokens to prime the rollout with (default: start of a sequence). */
  prime?: MovementToken[];
  /** Hard cap on generated tokens (excludes the terminal SEQUENCE_END). */
  maxSteps: number;
  /** Include the terminal SEQUENCE_END token in the output. Default false. */
  includeEnd?: boolean;
};

/**
 * Greedily roll the policy forward to reproduce or generalize a movement
 * sequence. With an empty prime it replays the most-likely learned trajectory;
 * priming with a partial/novel context lets it generalize via back-off.
 */
export function rolloutMovementPolicy(
  policy: TrainedMovementPolicy,
  options: RolloutOptions,
): MovementToken[] {
  const generated: MovementToken[] = [];
  const context: MovementToken[] = [...(options.prime ?? [])];
  for (let step = 0; step < options.maxSteps; step += 1) {
    const prediction = policy.predict(context);
    if (prediction.token === SEQUENCE_END) {
      if (options.includeEnd) {
        generated.push(SEQUENCE_END);
      }
      break;
    }
    generated.push(prediction.token);
    context.push(prediction.token);
  }
  return generated;
}

export type MovementPolicyEvaluation = {
  totalPredictions: number;
  correct: number;
  accuracy: number;
  /** contextOrder (back-off depth) → number of predictions made at that depth. */
  backoffHistogram: Record<number, number>;
  perSequence: Array<{
    trajectoryId: string;
    predictions: number;
    correct: number;
    accuracy: number;
  }>;
};

/**
 * Generalization eval harness: measure next-token prediction accuracy on
 * held-out (but related) movement sequences. Each sequence is walked
 * left-to-right, predicting every token — including its terminal
 * SEQUENCE_END — from the true preceding context (teacher forcing).
 */
export function evaluateMovementPolicy(
  policy: TrainedMovementPolicy,
  heldOut: MovementSequence[],
): MovementPolicyEvaluation {
  let totalPredictions = 0;
  let totalCorrect = 0;
  const backoffHistogram: Record<number, number> = {};
  const perSequence: MovementPolicyEvaluation["perSequence"] = [];

  for (const sequence of heldOut) {
    const targets = [...sequence.tokens, SEQUENCE_END];
    const context: MovementToken[] = [];
    let predictions = 0;
    let correct = 0;
    for (const target of targets) {
      const prediction = policy.predict(context);
      predictions += 1;
      totalPredictions += 1;
      backoffHistogram[prediction.contextOrder] = (backoffHistogram[prediction.contextOrder] ?? 0) + 1;
      if (prediction.token === target) {
        correct += 1;
        totalCorrect += 1;
      }
      if (target === SEQUENCE_END) {
        break;
      }
      context.push(target);
    }
    perSequence.push({
      trajectoryId: sequence.trajectoryId,
      predictions,
      correct,
      accuracy: predictions === 0 ? 0 : correct / predictions,
    });
  }

  return {
    totalPredictions,
    correct: totalCorrect,
    accuracy: totalPredictions === 0 ? 0 : totalCorrect / totalPredictions,
    backoffHistogram,
    perSequence,
  };
}

/** Deterministic mulberry32 PRNG so synthetic data is stable across runs. */
function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export type SyntheticMovementOptions = {
  count: number;
  seed?: number;
  minLength?: number;
  maxLength?: number;
  targets?: string[];
};

const DEFAULT_SYNTHETIC_TARGETS = ["compose", "search", "inbox", "sidebar", "settings", "reply"];

/**
 * Generate a deterministic, repeatable set of synthetic movement sequences that
 * share sub-structure (so back-off generalization is exercisable) without any
 * real OS input. Stands in for a live capture stream in tests.
 */
export function buildSyntheticMovementSequences(options: SyntheticMovementOptions): MovementSequence[] {
  const rng = mulberry32(options.seed ?? 1);
  const targets = options.targets ?? DEFAULT_SYNTHETIC_TARGETS;
  const minLength = Math.max(1, options.minLength ?? 3);
  const maxLength = Math.max(minLength, options.maxLength ?? 6);
  const gestures = ["tapped", "swiped up", "swiped down", "scrolled down", "typed into"];

  const sequences: MovementSequence[] = [];
  for (let index = 0; index < options.count; index += 1) {
    const length = minLength + Math.floor(rng() * (maxLength - minLength + 1));
    const tokens: MovementToken[] = [];
    for (let step = 0; step < length; step += 1) {
      const gesture = gestures[Math.floor(rng() * gestures.length)];
      const target = targets[Math.floor(rng() * targets.length)];
      tokens.push(`device::${gesture} ${target}`);
    }
    sequences.push({ trajectoryId: `synthetic-${index}`, tokens });
  }
  return sequences;
}
