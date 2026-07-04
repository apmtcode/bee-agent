import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

/**
 * Local-movement learning model.
 *
 * Objective #2 of the self-evolution mandate asks bee-agent to (a) record
 * local movements, (b) store them as a replayable dataset, (c) post-train a
 * local model to *repeat* the recorded movements, and (d) *generalize* to new
 * but related movements.
 *
 * The capture/replay/export pipeline already exists (`src/capture`,
 * `src/training/exporter.ts`). The training *runner* (`runner.ts`) builds
 * launch scripts for real on-device training (mlx/axolotl) — but nothing in
 * the codebase can actually *learn* from a movement sequence and *predict* the
 * next movement in-process. That is what this module adds: a pluggable model
 * backend plus a deterministic reference implementation that runs anywhere
 * (cloud/CI included), so the whole train→repeat→generalize loop is testable
 * without a real GPU or real OS input.
 *
 * The reference backend is a variable-order Markov model with Katz-style
 * back-off. Back-off is the generalization mechanism: an unseen context still
 * yields a sensible prediction by falling back to the longest *seen* suffix,
 * so a "new but related" movement (one that shares a recent sub-sequence with
 * training data) is handled without ever having been observed verbatim.
 *
 * The backend is an interface so a real on-device small model can be dropped
 * in later (load a `.gguf`, call into mlx, etc.) behind the same
 * `train()`/`predictNext()`/`generate()` surface.
 */

/** A single discrete movement, tokenized to a stable string. */
export type MovementToken = string;

/** Ordered movements captured from one trajectory. */
export type MovementSequence = {
  trajectoryId: string;
  tokens: MovementToken[];
};

/** A dataset of tokenized movement sequences, ready for training. */
export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

/** Sentinel tokens marking the start/end of a sequence during training. */
export const MOVEMENT_START_TOKEN = "start";
export const MOVEMENT_END_TOKEN = "end";

const CONTEXT_SEPARATOR = "";

/**
 * Turn a captured action into a stable movement token. Discriminative fields
 * (tool + gesture kind + direction + target) are joined so that structurally
 * identical movements collapse to the same token — which is what lets a model
 * learn and generalize over them.
 */
export function tokenizeAction(action: TrajectoryAction): MovementToken {
  const metadata = action.metadata ?? {};
  const parts: string[] = [action.tool];
  const gesture = readStringField(metadata, "gesture");
  if (gesture) {
    parts.push(gesture);
  }
  const direction = readStringField(metadata, "direction");
  if (direction) {
    parts.push(direction);
  }
  const target = readStringField(metadata, "target");
  if (target) {
    parts.push(`@${normalizeTargetLabel(target)}`);
  }
  return parts.join(":");
}

function readStringField(metadata: Record<string, unknown>, key: string): string | undefined {
  const value = metadata[key];
  return typeof value === "string" && value.trim().length > 0 ? value.trim() : undefined;
}

function normalizeTargetLabel(target: string): string {
  return target.trim().toLowerCase().replace(/\s+/g, "-");
}

/** Build a training dataset from captured trajectories (actions only, time-ordered). */
export function buildMovementDataset(trajectories: TrajectorySpan[]): MovementDataset {
  const sequences: MovementSequence[] = [];
  for (const trajectory of trajectories) {
    const tokens = [...trajectory.actions]
      .sort((a, b) => a.ts - b.ts)
      .map((action) => tokenizeAction(action));
    if (tokens.length > 0) {
      sequences.push({ trajectoryId: trajectory.id, tokens });
    }
  }
  return { version: 1, sequences };
}

/** Result of asking the model what movement comes next. */
export type MovementPrediction = {
  /** Most likely next token, or `undefined` if the model is empty. */
  token: MovementToken | undefined;
  /** Estimated probability of `token` under the matched context. */
  probability: number;
  /** Context order (back-off level) used: N for an N-token match, 0 for the
   *  unconditional prior, -1 if the model has no data at all. */
  order: number;
  /** All candidate tokens for the matched context, most likely first. */
  candidates: Array<{ token: MovementToken; probability: number }>;
};

/** A trained, queryable movement model. */
export type TrainedMovementModel = {
  readonly backendId: string;
  readonly order: number;
  readonly vocabulary: MovementToken[];
  /** Predict the next movement given a context (most-recent-last). */
  predictNext(context: MovementToken[]): MovementPrediction;
  /** Autoregressively generate a movement sequence from a seed context. */
  generate(seed: MovementToken[], maxLength: number): MovementToken[];
  /** Serialize the learned parameters for persistence / later inference. */
  serialize(): MovementModelSnapshot;
};

/** Options controlling how a backend trains. */
export type MovementTrainingOptions = {
  /** Maximum Markov order (context window). Default 3. */
  order?: number;
};

/** A pluggable local-movement model backend. */
export type MovementModelBackend = {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainingOptions): Promise<TrainedMovementModel>;
};

/** Portable, serialized model parameters. */
export type MovementModelSnapshot = {
  version: 1;
  backendId: string;
  order: number;
  vocabulary: MovementToken[];
  ngrams: Array<{ context: MovementToken[]; token: MovementToken; count: number }>;
};

const DEFAULT_ORDER = 3;

/**
 * Deterministic reference backend: a variable-order Markov model with
 * back-off. Fully in-process, no external deps — the mock/simulation backend
 * the mandate calls for, and a genuinely useful baseline movement predictor.
 */
export class MarkovMovementModelBackend implements MovementModelBackend {
  readonly id = "markov-backoff";

  async train(dataset: MovementDataset, options: MovementTrainingOptions = {}): Promise<TrainedMovementModel> {
    const order = Math.max(1, Math.floor(options.order ?? DEFAULT_ORDER));
    const counts = new Map<string, Map<MovementToken, number>>();
    const vocabulary = new Set<MovementToken>();

    for (const sequence of dataset.sequences) {
      const padded = [MOVEMENT_START_TOKEN, ...sequence.tokens, MOVEMENT_END_TOKEN];
      for (const token of sequence.tokens) {
        vocabulary.add(token);
      }
      for (let index = 1; index < padded.length; index += 1) {
        const token = padded[index] as MovementToken;
        for (let contextLength = 0; contextLength <= order; contextLength += 1) {
          if (contextLength > index) {
            break;
          }
          const context = padded.slice(index - contextLength, index);
          addCount(counts, contextKey(context), token);
        }
      }
    }

    return new MarkovMovementModel(this.id, order, counts, [...vocabulary].sort());
  }

  /** Rebuild a model from a serialized snapshot for later inference. */
  load(snapshot: MovementModelSnapshot): TrainedMovementModel {
    const counts = new Map<string, Map<MovementToken, number>>();
    for (const entry of snapshot.ngrams) {
      const bucket = counts.get(contextKey(entry.context)) ?? new Map<MovementToken, number>();
      bucket.set(entry.token, entry.count);
      counts.set(contextKey(entry.context), bucket);
    }
    return new MarkovMovementModel(snapshot.backendId, snapshot.order, counts, [...snapshot.vocabulary]);
  }
}

class MarkovMovementModel implements TrainedMovementModel {
  constructor(
    readonly backendId: string,
    readonly order: number,
    private readonly counts: Map<string, Map<MovementToken, number>>,
    readonly vocabulary: MovementToken[],
  ) {}

  predictNext(context: MovementToken[]): MovementPrediction {
    for (let contextLength = Math.min(this.order, context.length); contextLength >= 0; contextLength -= 1) {
      const suffix = context.slice(context.length - contextLength);
      const bucket = this.counts.get(contextKey(suffix));
      if (bucket && bucket.size > 0) {
        return summarizeBucket(bucket, contextLength);
      }
    }
    return { token: undefined, probability: 0, order: -1, candidates: [] };
  }

  generate(seed: MovementToken[], maxLength: number): MovementToken[] {
    const generated: MovementToken[] = [];
    const context = [MOVEMENT_START_TOKEN, ...seed];
    while (generated.length < maxLength) {
      const prediction = this.predictNext(context);
      if (prediction.token === undefined || prediction.token === MOVEMENT_END_TOKEN) {
        break;
      }
      generated.push(prediction.token);
      context.push(prediction.token);
    }
    return generated;
  }

  serialize(): MovementModelSnapshot {
    const ngrams: MovementModelSnapshot["ngrams"] = [];
    for (const [key, bucket] of this.counts) {
      const context = key.length === 0 ? [] : key.split(CONTEXT_SEPARATOR);
      for (const [token, count] of bucket) {
        ngrams.push({ context, token, count });
      }
    }
    ngrams.sort((a, b) => contextKey(a.context).localeCompare(contextKey(b.context)) || a.token.localeCompare(b.token));
    return {
      version: 1,
      backendId: this.backendId,
      order: this.order,
      vocabulary: [...this.vocabulary],
      ngrams,
    };
  }
}

function addCount(counts: Map<string, Map<MovementToken, number>>, key: string, token: MovementToken): void {
  const bucket = counts.get(key) ?? new Map<MovementToken, number>();
  bucket.set(token, (bucket.get(token) ?? 0) + 1);
  counts.set(key, bucket);
}

function contextKey(context: MovementToken[]): string {
  return context.join(CONTEXT_SEPARATOR);
}

function summarizeBucket(bucket: Map<MovementToken, number>, order: number): MovementPrediction {
  const total = [...bucket.values()].reduce((sum, count) => sum + count, 0);
  const candidates = [...bucket.entries()]
    .map(([token, count]) => ({ token, probability: count / total }))
    // Deterministic ordering: higher probability first, then lexicographic.
    .sort((a, b) => b.probability - a.probability || a.token.localeCompare(b.token));
  const best = candidates[0];
  return {
    token: best?.token,
    probability: best?.probability ?? 0,
    order,
    candidates,
  };
}

// ---------------------------------------------------------------------------
// Generalization eval harness
// ---------------------------------------------------------------------------

/** Deterministically split a dataset into train / holdout partitions. */
export function splitMovementDataset(
  dataset: MovementDataset,
  holdoutRatio: number,
): { train: MovementDataset; holdout: MovementDataset } {
  const ordered = [...dataset.sequences].sort((a, b) => a.trajectoryId.localeCompare(b.trajectoryId));
  const ratio = Math.min(0.9, Math.max(0, holdoutRatio));
  const holdoutCount = Math.min(ordered.length, Math.floor(ordered.length * ratio));
  const holdoutSequences = holdoutCount > 0 ? ordered.slice(ordered.length - holdoutCount) : [];
  const trainSequences = ordered.slice(0, ordered.length - holdoutSequences.length);
  return {
    train: { version: 1, sequences: trainSequences },
    holdout: { version: 1, sequences: holdoutSequences },
  };
}

/** Outcome of a generalization evaluation. */
export type MovementGeneralizationEval = {
  backendId: string;
  order: number;
  trainSequences: number;
  holdoutSequences: number;
  /** Teacher-forced next-token accuracy on the training set (replay fidelity). */
  replayFidelity: number;
  /** Teacher-forced next-token accuracy on held-out (unseen) sequences. */
  generalizationAccuracy: number;
  predictedTokens: number;
  correctTokens: number;
};

/**
 * Train the backend on the train split, then measure how well it (a) repeats
 * the training movements (replay fidelity) and (b) predicts held-out but
 * related movements (generalization). Both use teacher-forced next-token
 * accuracy over the padded sequence, including the terminal stop token.
 */
export async function evaluateMovementGeneralization(
  backend: MovementModelBackend,
  dataset: MovementDataset,
  options: { holdoutRatio?: number; order?: number } = {},
): Promise<MovementGeneralizationEval> {
  const { train, holdout } = splitMovementDataset(dataset, options.holdoutRatio ?? 0.3);
  const model = await backend.train(train, { order: options.order });

  const replay = scoreSequences(model, train.sequences);
  const generalization = scoreSequences(model, holdout.sequences);

  return {
    backendId: model.backendId,
    order: model.order,
    trainSequences: train.sequences.length,
    holdoutSequences: holdout.sequences.length,
    replayFidelity: replay.total > 0 ? replay.correct / replay.total : 0,
    generalizationAccuracy: generalization.total > 0 ? generalization.correct / generalization.total : 0,
    predictedTokens: generalization.total,
    correctTokens: generalization.correct,
  };
}

function scoreSequences(
  model: TrainedMovementModel,
  sequences: MovementSequence[],
): { correct: number; total: number } {
  let correct = 0;
  let total = 0;
  for (const sequence of sequences) {
    const targets = [...sequence.tokens, MOVEMENT_END_TOKEN];
    const context: MovementToken[] = [MOVEMENT_START_TOKEN];
    for (const target of targets) {
      const prediction = model.predictNext(context);
      total += 1;
      if (prediction.token === target) {
        correct += 1;
      }
      context.push(target);
    }
  }
  return { correct, total };
}
