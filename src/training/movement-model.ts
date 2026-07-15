import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

/**
 * In-process movement-learning backend for the local-movement subsystem.
 *
 * The capture pipeline records device gestures / OS events as
 * {@link TrajectorySpan} actions. This module turns those recorded actions
 * into a compact, tokenized dataset, trains a small sequence model on it, and
 * lets the model (a) *repeat* recorded movement patterns and (b) *generalize*
 * to new-but-related movements it never saw verbatim.
 *
 * The heavy on-device training runners (`LocalAppleSiliconTrainingRunner`)
 * shell out to MLX / Axolotl and cannot run in the cloud. This backend is the
 * pluggable, dependency-free counterpart: {@link MovementModelBackend} is the
 * seam a real small on-device model would implement, and
 * {@link MarkovMovementBackend} is a deterministic reference/mock backend that
 * runs anywhere (including CI) so the pipeline can be validated end-to-end with
 * synthetic event streams.
 */

/** A single normalized movement token, e.g. `swipe:down`, `tap`, `type`. */
export type MovementToken = string;

/** Sentinel tokens that bound a movement sequence for the sequence model. */
export const MOVEMENT_START_TOKEN: MovementToken = "<start>";
export const MOVEMENT_END_TOKEN: MovementToken = "<end>";

/** A tokenized movement sequence derived from one trajectory. */
export type MovementSequence = {
  trajectoryId: string;
  sessionId: string;
  tokens: MovementToken[];
};

/** A tokenized, model-ready dataset built from recorded trajectories. */
export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
  vocabulary: MovementToken[];
};

export type MovementPrediction = {
  token: MovementToken;
  probability: number;
  /** Markov order actually used after back-off (k..0). */
  order: number;
};

export type MovementTrainingOptions = {
  /** Markov order (context length). Defaults to 2. Clamped to >= 1. */
  order?: number;
  /** Additive (Laplace) smoothing weight used when scoring. Defaults to 0.1. */
  smoothing?: number;
};

export type MovementGenerateOptions = {
  /** Seed tokens to continue from (excluding the `<start>` padding). */
  seed?: MovementToken[];
  /** Hard cap on generated tokens (excluding boundary tokens). Defaults to 32. */
  maxLength?: number;
  /**
   * Optional deterministic RNG in [0, 1). When provided, the model *samples*
   * from the learned distribution; otherwise it greedily takes the most likely
   * next token (fully deterministic).
   */
  rng?: () => number;
};

export type SerializedMovementModel = {
  backendId: string;
  order: number;
  smoothing: number;
  vocabulary: MovementToken[];
  /** contextKey -> { token -> count } for every order 0..k. */
  counts: Record<string, Record<MovementToken, number>>;
};

/** A trained, queryable movement model. */
export interface TrainedMovementModel {
  readonly backendId: string;
  readonly order: number;
  /** Most likely next token given a raw (un-padded) context, with back-off. */
  predictNext(context: MovementToken[]): MovementPrediction | undefined;
  /** Smoothed probability of `target` following `context` (0..1), with back-off. */
  conditionalProbability(context: MovementToken[], target: MovementToken): number;
  /** Produce a movement sequence (repeat or generalize depending on rng/seed). */
  generate(options?: MovementGenerateOptions): MovementToken[];
  /** Average per-token probability the model assigns to a sequence (0..1). */
  score(tokens: MovementToken[]): number;
  toJSON(): SerializedMovementModel;
}

/**
 * Pluggable training backend. A real on-device small model implements this same
 * interface; {@link MarkovMovementBackend} is the deterministic reference.
 */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainingOptions): Promise<TrainedMovementModel>;
}

/**
 * Turn one recorded action into a compact, generalizable movement token.
 *
 * Device gestures collapse to `<gesture>` or `<gesture>:<direction>` so that
 * the model learns the *shape* of movement (swipe-down, tap, type) rather than
 * memorizing exact UI target strings. Non-device tool actions become
 * `action:<tool>`.
 */
export function tokenizeAction(action: TrajectoryAction): MovementToken {
  const metadata = action.metadata ?? {};
  const gesture = typeof metadata.gesture === "string" ? metadata.gesture : undefined;
  if (gesture) {
    const direction = typeof metadata.direction === "string" ? metadata.direction : undefined;
    return direction ? `${gesture}:${direction}` : gesture;
  }
  const tool = action.tool.trim().toLowerCase();
  return tool ? `action:${tool}` : "action:unknown";
}

/** Tokenize a trajectory's actions in timestamp order into a movement sequence. */
export function tokenizeTrajectory(trajectory: TrajectorySpan): MovementSequence {
  const tokens = [...trajectory.actions]
    .sort((a, b) => a.ts - b.ts)
    .map((action) => tokenizeAction(action));
  return { trajectoryId: trajectory.id, sessionId: trajectory.sessionId, tokens };
}

/** Build a model-ready dataset from recorded trajectories (skips empty ones). */
export function buildMovementDataset(trajectories: TrajectorySpan[]): MovementDataset {
  const sequences = trajectories
    .map((trajectory) => tokenizeTrajectory(trajectory))
    .filter((sequence) => sequence.tokens.length > 0);
  const vocabulary = new Set<MovementToken>([MOVEMENT_END_TOKEN]);
  for (const sequence of sequences) {
    for (const token of sequence.tokens) {
      vocabulary.add(token);
    }
  }
  return { version: 1, sequences, vocabulary: [...vocabulary].sort() };
}

function contextKey(context: MovementToken[]): string {
  return `${context.length}␟${context.join("␟")}`;
}

function padContext(context: MovementToken[], order: number): MovementToken[] {
  const trimmed = context.slice(-order);
  const padding: MovementToken[] = [];
  for (let i = trimmed.length; i < order; i += 1) {
    padding.push(MOVEMENT_START_TOKEN);
  }
  return [...padding, ...trimmed];
}

function argmax(distribution: Map<MovementToken, number>): { token: MovementToken; count: number } | undefined {
  let best: { token: MovementToken; count: number } | undefined;
  // Deterministic tie-break: lexicographically smallest token wins.
  for (const token of [...distribution.keys()].sort()) {
    const count = distribution.get(token) ?? 0;
    if (!best || count > best.count) {
      best = { token, count };
    }
  }
  return best;
}

class MarkovMovementModel implements TrainedMovementModel {
  constructor(
    readonly backendId: string,
    readonly order: number,
    private readonly smoothing: number,
    private readonly vocabulary: MovementToken[],
    private readonly counts: Map<string, Map<MovementToken, number>>,
  ) {}

  /** Highest-order observed distribution for a raw context (back-off). */
  private resolveDistribution(
    context: MovementToken[],
  ): { order: number; distribution: Map<MovementToken, number> } | undefined {
    for (let order = this.order; order >= 0; order -= 1) {
      const key = contextKey(padContext(context, order));
      const distribution = this.counts.get(key);
      if (distribution && distribution.size > 0) {
        return { order, distribution };
      }
    }
    return undefined;
  }

  predictNext(context: MovementToken[]): MovementPrediction | undefined {
    const resolved = this.resolveDistribution(context);
    if (!resolved) {
      return undefined;
    }
    const top = argmax(resolved.distribution);
    if (!top) {
      return undefined;
    }
    const total = totalCount(resolved.distribution);
    return {
      token: top.token,
      probability: total > 0 ? top.count / total : 0,
      order: resolved.order,
    };
  }

  conditionalProbability(context: MovementToken[], target: MovementToken): number {
    const vocabSize = Math.max(this.vocabulary.length, 1);
    const resolved = this.resolveDistribution(context);
    const distribution = resolved?.distribution;
    const count = distribution?.get(target) ?? 0;
    const total = distribution ? totalCount(distribution) : 0;
    // Additive smoothing keeps unseen tokens non-zero so generalization on
    // held-out sequences yields a finite, comparable likelihood.
    return (count + this.smoothing) / (total + this.smoothing * vocabSize);
  }

  score(tokens: MovementToken[]): number {
    const sequence = [...tokens, MOVEMENT_END_TOKEN];
    let sum = 0;
    for (let i = 0; i < sequence.length; i += 1) {
      sum += this.conditionalProbability(tokens.slice(0, i), sequence[i] as MovementToken);
    }
    return sum / sequence.length;
  }

  generate(options: MovementGenerateOptions = {}): MovementToken[] {
    const maxLength = options.maxLength ?? 32;
    const generated: MovementToken[] = [...(options.seed ?? [])];
    for (let step = 0; step < maxLength; step += 1) {
      const resolved = this.resolveDistribution(generated);
      if (!resolved) {
        break;
      }
      const next = options.rng
        ? sampleToken(resolved.distribution, options.rng)
        : argmax(resolved.distribution)?.token;
      if (!next || next === MOVEMENT_END_TOKEN) {
        break;
      }
      generated.push(next);
    }
    return options.seed ? generated.slice(options.seed.length) : generated;
  }

  toJSON(): SerializedMovementModel {
    const counts: Record<string, Record<MovementToken, number>> = {};
    for (const [key, distribution] of this.counts) {
      const entry: Record<MovementToken, number> = {};
      for (const [token, count] of distribution) {
        entry[token] = count;
      }
      counts[key] = entry;
    }
    return {
      backendId: this.backendId,
      order: this.order,
      smoothing: this.smoothing,
      vocabulary: [...this.vocabulary],
      counts,
    };
  }
}

function totalCount(distribution: Map<MovementToken, number>): number {
  let total = 0;
  for (const count of distribution.values()) {
    total += count;
  }
  return total;
}

function sampleToken(distribution: Map<MovementToken, number>, rng: () => number): MovementToken | undefined {
  const total = totalCount(distribution);
  if (total <= 0) {
    return undefined;
  }
  // Deterministic ordering so the same rng stream yields the same draw.
  const entries = [...distribution.entries()].sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
  let threshold = Math.min(Math.max(rng(), 0), 0.999999) * total;
  for (const [token, count] of entries) {
    threshold -= count;
    if (threshold < 0) {
      return token;
    }
  }
  return entries[entries.length - 1]?.[0];
}

/**
 * Deterministic, dependency-free reference backend: an order-k Markov chain
 * with back-off. Learns transition frequencies over movement tokens, repeats
 * recorded patterns greedily, and generalizes to unseen contexts by backing off
 * to shorter histories. Serves as the mock backend for cloud/CI while a real
 * on-device model implements the same {@link MovementModelBackend} interface.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov-movement";

  async train(dataset: MovementDataset, options: MovementTrainingOptions = {}): Promise<TrainedMovementModel> {
    const order = Math.max(1, Math.floor(options.order ?? 2));
    const smoothing = options.smoothing ?? 0.1;
    const counts = new Map<string, Map<MovementToken, number>>();

    for (const sequence of dataset.sequences) {
      const stream = [MOVEMENT_START_TOKEN, ...sequence.tokens, MOVEMENT_END_TOKEN];
      for (let i = 1; i < stream.length; i += 1) {
        const target = stream[i] as MovementToken;
        const history = stream.slice(0, i);
        // Record this transition at every order 0..k so back-off has data.
        for (let o = 0; o <= order; o += 1) {
          const key = contextKey(padContext(history, o));
          let distribution = counts.get(key);
          if (!distribution) {
            distribution = new Map<MovementToken, number>();
            counts.set(key, distribution);
          }
          distribution.set(target, (distribution.get(target) ?? 0) + 1);
        }
      }
    }

    return new MarkovMovementModel(this.id, order, smoothing, [...dataset.vocabulary], counts);
  }
}

/** Convenience: build a dataset from trajectories and train a model on it. */
export async function trainMovementModelFromTrajectories(
  trajectories: TrajectorySpan[],
  options: MovementTrainingOptions & { backend?: MovementModelBackend } = {},
): Promise<{ model: TrainedMovementModel; dataset: MovementDataset }> {
  const dataset = buildMovementDataset(trajectories);
  const backend = options.backend ?? new MarkovMovementBackend();
  const model = await backend.train(dataset, options);
  return { model, dataset };
}

export type MovementGeneralizationReport = {
  sequenceCount: number;
  tokenCount: number;
  /** Greedy top-1 next-token accuracy on held-out sequences (0..1). */
  nextTokenAccuracy: number;
  /** Mean per-token probability assigned to held-out sequences (0..1). */
  averageLikelihood: number;
  /** Perplexity (lower is better); Infinity if there is nothing to evaluate. */
  perplexity: number;
  /** Fraction of held-out contexts seen at full Markov order (memorization). */
  fullOrderCoverage: number;
};

/**
 * Generalization eval harness: measures how well a trained model predicts the
 * next movement on *held-out but related* sequences. `fullOrderCoverage`
 * separates memorization (context seen verbatim) from true generalization
 * (correct next-token via back-off on an unseen context).
 */
export function evaluateMovementGeneralization(
  model: TrainedMovementModel,
  heldOut: MovementSequence[],
): MovementGeneralizationReport {
  let tokenCount = 0;
  let correct = 0;
  let fullOrderHits = 0;
  let logLikelihood = 0;

  for (const sequence of heldOut) {
    const stream = [...sequence.tokens, MOVEMENT_END_TOKEN];
    for (let i = 0; i < stream.length; i += 1) {
      const context = sequence.tokens.slice(0, i);
      const target = stream[i] as MovementToken;
      const prediction = model.predictNext(context);
      tokenCount += 1;
      if (prediction?.token === target) {
        correct += 1;
      }
      if (prediction && prediction.order === model.order && i >= model.order) {
        fullOrderHits += 1;
      }
      const likelihood = model.conditionalProbability(context, target);
      logLikelihood += Math.log(Math.max(likelihood, Number.EPSILON));
    }
  }

  const averageLikelihood = tokenCount > 0 ? Math.exp(logLikelihood / tokenCount) : 0;
  return {
    sequenceCount: heldOut.length,
    tokenCount,
    nextTokenAccuracy: tokenCount > 0 ? correct / tokenCount : 0,
    averageLikelihood,
    perplexity: tokenCount > 0 ? Math.exp(-logLikelihood / tokenCount) : Infinity,
    fullOrderCoverage: tokenCount > 0 ? fullOrderHits / tokenCount : 0,
  };
}
