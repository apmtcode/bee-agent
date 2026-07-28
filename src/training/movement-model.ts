/**
 * Pluggable local movement-model backend.
 *
 * Objective 2(c)/(d) of the self-evolution charter: post-train a *local* model
 * on a recorded-movement dataset so it can (1) repeat the recorded movements and
 * (2) generalise to new-but-related movements. The real on-device training path
 * lives in {@link ./runner.ts} (it shells out to `mlx`/`axolotl` on Apple
 * Silicon and therefore cannot run in the cloud). This module provides a small,
 * fully in-process, deterministic backend behind a stable interface so the whole
 * train -> infer -> generalise loop is validated in CI without any real OS input
 * or GPU. Swap {@link MarkovMovementBackend} for a neural backend later without
 * touching call sites — the {@link MovementModelBackend} seam is the contract.
 */
import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/** The atomic unit a movement model learns — a normalised `tool::summary` pair. */
export type MovementToken = string;

/** An ordered movement sequence extracted from one trajectory / replay. */
export type MovementSequence = {
  trajectoryId?: string;
  tokens: MovementToken[];
};

/** Sentinel tokens framing every training sequence so the model learns starts/ends. */
export const MOVEMENT_START = "start";
export const MOVEMENT_END = "end";

const CONTEXT_SEP = "";

/** A ranked next-movement candidate. */
export type MovementPrediction = {
  token: MovementToken;
  probability: number;
  /** Length of the context suffix the prediction was drawn from (backoff depth). */
  contextOrder: number;
};

export type PredictOptions = {
  /** Cap the number of ranked candidates returned. Default: all. */
  topK?: number;
};

export type GenerateOptions = {
  /** Priming context; sequences are auto-prefixed with {@link MOVEMENT_START}. */
  context?: MovementToken[];
  /** Hard cap on generated tokens (excludes the terminal END). Default: 64. */
  maxSteps?: number;
  /**
   * Deterministic seed. When provided, generation samples from the predicted
   * distribution using a seeded LCG (reproducible variety). When omitted,
   * generation is greedy (highest probability, ties broken by token order).
   */
  seed?: number;
};

/** JSON-able snapshot for persistence or hand-off to a device runtime. */
export type MovementModelSnapshot = {
  version: 1;
  backend: string;
  order: number;
  smoothing: number;
  vocab: MovementToken[];
  /** contextKey -> token -> count. Context keys are order-0..order suffixes. */
  counts: Record<string, Record<MovementToken, number>>;
};

/** A trained, queryable movement model. */
export type TrainedMovementModel = {
  readonly backend: string;
  readonly order: number;
  readonly vocab: readonly MovementToken[];
  /** Rank the most likely next tokens given the prior context. */
  predictNext(context: MovementToken[], options?: PredictOptions): MovementPrediction[];
  /** Generate a movement sequence from a (possibly empty) priming context. */
  generate(options?: GenerateOptions): MovementToken[];
  /** Serialise to a JSON-able snapshot. */
  serialize(): MovementModelSnapshot;
};

export type TrainMovementOptions = {
  /** Markov order (context window length). Default: 2. */
  order?: number;
  /** Additive (Laplace) smoothing mass — enables generalisation. Default: 0.01. */
  smoothing?: number;
};

/** The pluggable backend seam. Implement this to add a real neural backend. */
export type MovementModelBackend = {
  readonly name: string;
  train(sequences: MovementSequence[], options?: TrainMovementOptions): TrainedMovementModel;
  restore(snapshot: MovementModelSnapshot): TrainedMovementModel;
};

// ---------------------------------------------------------------------------
// Tokenisation — turn recorded movements into learnable token sequences.
// ---------------------------------------------------------------------------

/** Normalise an action into a stable movement token. */
export function movementTokenFromAction(tool: string, summary: string): MovementToken {
  const normalizedTool = tool.trim().toLowerCase() || "unknown";
  const normalizedSummary = summary.replace(/\s+/g, " ").trim().toLowerCase();
  return normalizedSummary ? `${normalizedTool}::${normalizedSummary}` : normalizedTool;
}

/** Extract the ordered movement token sequence from replay timeline events. */
export function tokenizeReplayEvents(events: readonly ReplayTimelineEvent[]): MovementToken[] {
  return events
    .filter((event): event is Extract<ReplayTimelineEvent, { kind: "action" }> => event.kind === "action")
    .map((event) => movementTokenFromAction(event.tool, event.summary));
}

/** Build one {@link MovementSequence} per replay manifest. */
export function sequencesFromReplays(replays: readonly ReplayManifest[]): MovementSequence[] {
  return replays
    .map<MovementSequence>((replay) => ({
      trajectoryId: replay.trajectoryIds[0],
      tokens: tokenizeReplayEvents(replay.events),
    }))
    .filter((sequence) => sequence.tokens.length > 0);
}

/** Build a {@link MovementSequence} directly from a trajectory's actions. */
export function sequenceFromTrajectory(trajectory: TrajectorySpan): MovementSequence {
  return {
    trajectoryId: trajectory.id,
    tokens: trajectory.actions.map((action) => movementTokenFromAction(action.tool, action.summary)),
  };
}

// ---------------------------------------------------------------------------
// MarkovMovementBackend — a small, deterministic, on-device backend.
// ---------------------------------------------------------------------------

/**
 * Order-k Markov backend with stupid-backoff and additive smoothing.
 *
 * - **Repeat**: for a context seen in training, the highest-order transition
 *   counts reproduce the recorded continuation exactly.
 * - **Generalise**: for an unseen context that shares a shorter suffix with the
 *   training data, prediction *backs off* to that suffix; additive smoothing
 *   keeps the full vocabulary reachable so novel-but-related contexts still rank
 *   plausible next movements instead of returning nothing.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly name = "markov";

  constructor(private readonly defaults: TrainMovementOptions = {}) {}

  train(sequences: MovementSequence[], options: TrainMovementOptions = {}): TrainedMovementModel {
    const order = Math.max(1, options.order ?? this.defaults.order ?? 2);
    const smoothing = Math.max(0, options.smoothing ?? this.defaults.smoothing ?? 0.01);
    const counts: Record<string, Record<MovementToken, number>> = {};
    const vocab = new Set<MovementToken>();

    for (const sequence of sequences) {
      const framed = [MOVEMENT_START, ...sequence.tokens, MOVEMENT_END];
      for (const token of sequence.tokens) {
        vocab.add(token);
      }
      vocab.add(MOVEMENT_END);
      for (let i = 1; i < framed.length; i += 1) {
        const target = framed[i]!;
        // Record every backoff order 0..order for this position.
        for (let ctxLen = 0; ctxLen <= order; ctxLen += 1) {
          if (ctxLen > i) {
            break;
          }
          const contextTokens = framed.slice(i - ctxLen, i);
          const key = contextKey(contextTokens);
          const bucket = (counts[key] ??= {});
          bucket[target] = (bucket[target] ?? 0) + 1;
        }
      }
    }

    return new MarkovMovementModel(order, smoothing, [...vocab].sort(), counts);
  }

  restore(snapshot: MovementModelSnapshot): TrainedMovementModel {
    if (snapshot.backend !== this.name) {
      throw new Error(`snapshot backend "${snapshot.backend}" does not match "${this.name}"`);
    }
    return new MarkovMovementModel(snapshot.order, snapshot.smoothing, [...snapshot.vocab], cloneCounts(snapshot.counts));
  }
}

class MarkovMovementModel implements TrainedMovementModel {
  readonly backend = "markov";

  constructor(
    readonly order: number,
    private readonly smoothing: number,
    readonly vocab: readonly MovementToken[],
    private readonly counts: Record<string, Record<MovementToken, number>>,
  ) {}

  predictNext(context: MovementToken[], options: PredictOptions = {}): MovementPrediction[] {
    // Stupid-backoff: use the longest context suffix (up to `order`) that has
    // observed transitions. Falls back through shorter suffixes to the unigram.
    let bucket: Record<MovementToken, number> | undefined;
    let contextOrder = 0;
    for (let ctxLen = Math.min(this.order, context.length); ctxLen >= 0; ctxLen -= 1) {
      const suffix = context.slice(context.length - ctxLen);
      const candidate = this.counts[contextKey(suffix)];
      if (candidate && Object.keys(candidate).length > 0) {
        bucket = candidate;
        contextOrder = ctxLen;
        break;
      }
    }

    const targets = this.vocab.length > 0 ? this.vocab : Object.keys(bucket ?? {});
    const denominator =
      Object.values(bucket ?? {}).reduce((sum, count) => sum + count, 0) + this.smoothing * targets.length;

    const predictions: MovementPrediction[] = targets.map((token) => {
      const observed = bucket?.[token] ?? 0;
      const probability = denominator > 0 ? (observed + this.smoothing) / denominator : 0;
      return { token, probability, contextOrder };
    });

    predictions.sort((a, b) => {
      if (b.probability !== a.probability) {
        return b.probability - a.probability;
      }
      return a.token < b.token ? -1 : a.token > b.token ? 1 : 0;
    });

    return options.topK !== undefined ? predictions.slice(0, Math.max(0, options.topK)) : predictions;
  }

  generate(options: GenerateOptions = {}): MovementToken[] {
    const maxSteps = options.maxSteps ?? 64;
    const context: MovementToken[] = [MOVEMENT_START, ...(options.context ?? [])];
    const output: MovementToken[] = [...(options.context ?? [])];
    let rng = options.seed !== undefined ? makeLcg(options.seed) : undefined;

    while (output.length < maxSteps) {
      const ranked = this.predictNext(context).filter((prediction) => prediction.token !== MOVEMENT_START);
      if (ranked.length === 0) {
        break;
      }
      const next = rng ? sample(ranked, rng()) : ranked[0]!.token;
      if (next === MOVEMENT_END) {
        break;
      }
      output.push(next);
      context.push(next);
    }

    return output;
  }

  serialize(): MovementModelSnapshot {
    return {
      version: 1,
      backend: this.backend,
      order: this.order,
      smoothing: this.smoothing,
      vocab: [...this.vocab],
      counts: cloneCounts(this.counts),
    };
  }
}

// ---------------------------------------------------------------------------
// Convenience + evaluation (generalisation harness).
// ---------------------------------------------------------------------------

/** Train the default (Markov) backend on a movement dataset. */
export function trainMovementModel(
  sequences: MovementSequence[],
  options: TrainMovementOptions = {},
): TrainedMovementModel {
  return new MarkovMovementBackend().train(sequences, options);
}

export type MovementEval = {
  sequenceCount: number;
  tokenCount: number;
  /** Fraction of held-out next-tokens the model ranked #1 (repeat fidelity). */
  top1Accuracy: number;
  /** Fraction where the true next-token fell within the top-K ranking. */
  topKAccuracy: number;
  /** Mean per-token log-loss (lower is better; measures generalisation). */
  meanLogLoss: number;
};

/**
 * Score a trained model against held-out sequences. Feeds each prefix and checks
 * how well the model predicts the true next movement — the generalisation metric
 * for measuring replay fidelity on unseen-but-related trajectories.
 */
export function evaluateMovementModel(
  model: TrainedMovementModel,
  heldOut: readonly MovementSequence[],
  options: { topK?: number } = {},
): MovementEval {
  const topK = options.topK ?? 3;
  let tokenCount = 0;
  let top1 = 0;
  let topKHits = 0;
  let logLossSum = 0;

  for (const sequence of heldOut) {
    const framed = [...sequence.tokens, MOVEMENT_END];
    // Prime with START exactly as generation does, so the first-token prediction
    // is drawn from the learned sequence-start distribution rather than a tie.
    const context: MovementToken[] = [MOVEMENT_START];
    for (const trueNext of framed) {
      const ranked = model.predictNext(context);
      if (ranked.length > 0) {
        if (ranked[0]!.token === trueNext) {
          top1 += 1;
        }
        if (ranked.slice(0, topK).some((prediction) => prediction.token === trueNext)) {
          topKHits += 1;
        }
        const match = ranked.find((prediction) => prediction.token === trueNext);
        const probability = match?.probability ?? 1e-9;
        logLossSum += -Math.log(Math.max(probability, 1e-9));
      }
      tokenCount += 1;
      context.push(trueNext);
    }
  }

  return {
    sequenceCount: heldOut.length,
    tokenCount,
    top1Accuracy: tokenCount > 0 ? top1 / tokenCount : 0,
    topKAccuracy: tokenCount > 0 ? topKHits / tokenCount : 0,
    meanLogLoss: tokenCount > 0 ? logLossSum / tokenCount : 0,
  };
}

// ---------------------------------------------------------------------------
// internals
// ---------------------------------------------------------------------------

function contextKey(tokens: readonly MovementToken[]): string {
  return `${tokens.length}${CONTEXT_SEP}${tokens.join(CONTEXT_SEP)}`;
}

function cloneCounts(
  counts: Record<string, Record<MovementToken, number>>,
): Record<string, Record<MovementToken, number>> {
  const clone: Record<string, Record<MovementToken, number>> = {};
  for (const [key, bucket] of Object.entries(counts)) {
    clone[key] = { ...bucket };
  }
  return clone;
}

/** Deterministic linear congruential generator returning values in [0, 1). */
function makeLcg(seed: number): () => number {
  let state = (Math.floor(seed) % 2147483647 + 2147483647) % 2147483647 || 1;
  return () => {
    state = (state * 16807) % 2147483647;
    return (state - 1) / 2147483646;
  };
}

function sample(ranked: MovementPrediction[], roll: number): MovementToken {
  const total = ranked.reduce((sum, prediction) => sum + prediction.probability, 0);
  let threshold = roll * total;
  for (const prediction of ranked) {
    threshold -= prediction.probability;
    if (threshold <= 0) {
      return prediction.token;
    }
  }
  return ranked[ranked.length - 1]!.token;
}
