/**
 * Pluggable local movement-model backend.
 *
 * This is the inference half of the local-movement learning subsystem
 * (objective #2c/#2d): given a dataset of recorded movement sequences it
 * post-trains a small model that can (a) *repeat* the recorded movements and
 * (b) *generalize* to new-but-related movements.
 *
 * The engine runs in Anthropic's cloud with no access to the user's real
 * machine and no heavy ML runtime, so the concrete backend shipped here is a
 * deterministic, dependency-free n-gram Markov model with back-off. It is both
 * the reference/mock backend that keeps cloud + CI tests green AND a genuinely
 * useful on-device baseline. A heavier real backend (a small local transformer,
 * an mlx/axolotl-trained policy, ...) implements the same {@link
 * MovementModelBackend} interface and drops in without touching callers.
 */

/** A movement token is a compact discrete symbol describing one movement/event. */
export type MovementToken = string;

/** Marks the start of a sequence so generation can seed from an empty context. */
export const MOVEMENT_SEQUENCE_START = "<seq>";
/** Marks the end of a sequence so generation knows when to stop. */
export const MOVEMENT_SEQUENCE_END = "<end>";

/** Default n-gram order (max context length) used when a request omits one. */
export const DEFAULT_MOVEMENT_MODEL_ORDER = 3;

export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

export type MovementTrainingRequest = {
  sequences: MovementSequence[];
  /** Max context length; higher = more faithful replay, lower = more general. */
  order?: number;
};

export type MovementCandidate = {
  token: MovementToken;
  probability: number;
};

export type MovementPrediction = {
  token: MovementToken;
  probability: number;
  /** How many context tokens matched (0 = unigram back-off). */
  matchedOrder: number;
  /** Ranked alternatives for the matched context (includes the winner). */
  candidates: MovementCandidate[];
};

export type MovementModelStats = {
  order: number;
  vocabularySize: number;
  sequenceCount: number;
  tokenCount: number;
  contextCount: number;
};

export type SerializedMovementModel = {
  backend: string;
  version: 1;
  order: number;
  vocabulary: MovementToken[];
  /** contextKey (JSON array) -> next token -> observed count. */
  transitions: Record<string, Record<MovementToken, number>>;
  stats: MovementModelStats;
};

export interface TrainedMovementModel {
  readonly backendId: string;
  readonly stats: MovementModelStats;
  /** Predict the single most likely next token for a context, or undefined if untrained. */
  predictNext(context: MovementToken[]): MovementPrediction | undefined;
  /** Autoregressively generate up to `maxSteps` tokens from a seed context. */
  generate(seed: MovementToken[], maxSteps: number): MovementToken[];
  serialize(): SerializedMovementModel;
}

export interface MovementModelBackend {
  readonly id: string;
  train(request: MovementTrainingRequest): Promise<TrainedMovementModel>;
  load(serialized: SerializedMovementModel): TrainedMovementModel;
}

const NGRAM_BACKEND_ID = "ngram-backoff";

/** Wrap a raw token list with sequence boundary markers for training/eval. */
export function wrapMovementSequence(tokens: MovementToken[]): MovementToken[] {
  return [MOVEMENT_SEQUENCE_START, ...tokens, MOVEMENT_SEQUENCE_END];
}

function contextKey(context: MovementToken[]): string {
  return JSON.stringify(context);
}

/**
 * Deterministic n-gram Markov backend with stupid-backoff.
 *
 * Training counts, for every context length 0..order, how often each token
 * follows each context. Prediction tries the longest available context and
 * backs off to shorter ones (ultimately the unigram distribution) — this is
 * what lets the model *generalize*: an unseen k-gram context degrades
 * gracefully to a (k-1)-gram it has seen, rather than failing.
 *
 * All tie-breaks are deterministic (count desc, then token asc) so training and
 * inference are fully reproducible in cloud/CI.
 */
export class NGramMovementModelBackend implements MovementModelBackend {
  readonly id = NGRAM_BACKEND_ID;

  async train(request: MovementTrainingRequest): Promise<TrainedMovementModel> {
    const order = normalizeOrder(request.order);
    const transitions: Record<string, Record<MovementToken, number>> = {};
    const vocabulary = new Set<MovementToken>();
    let tokenCount = 0;

    for (const sequence of request.sequences) {
      const wrapped = wrapMovementSequence(sequence.tokens);
      for (let index = 1; index < wrapped.length; index += 1) {
        const next = wrapped[index]!;
        vocabulary.add(next);
        tokenCount += 1;
        const maxContext = Math.min(order, index);
        for (let k = 0; k <= maxContext; k += 1) {
          const context = wrapped.slice(index - k, index);
          const key = contextKey(context);
          const bucket = (transitions[key] ??= {});
          bucket[next] = (bucket[next] ?? 0) + 1;
        }
      }
    }

    const stats: MovementModelStats = {
      order,
      vocabularySize: vocabulary.size,
      sequenceCount: request.sequences.length,
      tokenCount,
      contextCount: Object.keys(transitions).length,
    };

    return new NGramMovementModel({
      backend: this.id,
      version: 1,
      order,
      vocabulary: [...vocabulary].sort(),
      transitions,
      stats,
    });
  }

  load(serialized: SerializedMovementModel): TrainedMovementModel {
    return new NGramMovementModel(serialized);
  }
}

class NGramMovementModel implements TrainedMovementModel {
  readonly backendId: string;
  readonly stats: MovementModelStats;
  private readonly order: number;
  private readonly transitions: Record<string, Record<MovementToken, number>>;

  constructor(serialized: SerializedMovementModel) {
    this.backendId = serialized.backend;
    this.order = serialized.order;
    this.transitions = serialized.transitions;
    this.stats = serialized.stats;
  }

  predictNext(context: MovementToken[]): MovementPrediction | undefined {
    const maxContext = Math.min(this.order, context.length);
    for (let k = maxContext; k >= 0; k -= 1) {
      const slice = context.slice(context.length - k);
      const bucket = this.transitions[contextKey(slice)];
      if (!bucket) {
        continue;
      }
      const candidates = rankCandidates(bucket);
      const best = candidates[0];
      if (!best) {
        continue;
      }
      return {
        token: best.token,
        probability: best.probability,
        matchedOrder: k,
        candidates,
      };
    }
    return undefined;
  }

  generate(seed: MovementToken[], maxSteps: number): MovementToken[] {
    const output: MovementToken[] = [];
    const context: MovementToken[] = seed.length > 0 ? [...seed] : [MOVEMENT_SEQUENCE_START];
    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predictNext(context);
      if (!prediction || prediction.token === MOVEMENT_SEQUENCE_END) {
        break;
      }
      output.push(prediction.token);
      context.push(prediction.token);
    }
    return output;
  }

  serialize(): SerializedMovementModel {
    return {
      backend: this.backendId,
      version: 1,
      order: this.order,
      vocabulary: [...new Set(Object.values(this.transitions).flatMap((bucket) => Object.keys(bucket)))].sort(),
      transitions: this.transitions,
      stats: this.stats,
    };
  }
}

function rankCandidates(bucket: Record<MovementToken, number>): MovementCandidate[] {
  const total = Object.values(bucket).reduce((sum, count) => sum + count, 0);
  return Object.entries(bucket)
    .map(([token, count]) => ({ token, probability: total > 0 ? count / total : 0 }))
    .sort((a, b) => {
      if (b.probability !== a.probability) {
        return b.probability - a.probability;
      }
      return a.token < b.token ? -1 : a.token > b.token ? 1 : 0;
    });
}

function normalizeOrder(order: number | undefined): number {
  if (order === undefined || !Number.isFinite(order)) {
    return DEFAULT_MOVEMENT_MODEL_ORDER;
  }
  return Math.max(1, Math.floor(order));
}

export type MovementEvalResult = {
  /** Number of next-token predictions scored. */
  predictions: number;
  /** How many predictions matched the held-out ground truth. */
  correct: number;
  /** correct / predictions (0 when no predictions were made). */
  accuracy: number;
};

/**
 * Generalization eval harness: measure next-token replay fidelity on held-out
 * (but related) sequences. Every position after the first is scored, including
 * the terminal end-of-sequence token, so both "repeat" and "know when to stop"
 * are exercised.
 */
export function evaluateNextTokenAccuracy(
  model: TrainedMovementModel,
  sequences: MovementSequence[],
): MovementEvalResult {
  let predictions = 0;
  let correct = 0;
  for (const sequence of sequences) {
    const wrapped = wrapMovementSequence(sequence.tokens);
    for (let index = 1; index < wrapped.length; index += 1) {
      const context = wrapped.slice(0, index);
      const prediction = model.predictNext(context);
      predictions += 1;
      if (prediction?.token === wrapped[index]) {
        correct += 1;
      }
    }
  }
  return {
    predictions,
    correct,
    accuracy: predictions > 0 ? correct / predictions : 0,
  };
}
