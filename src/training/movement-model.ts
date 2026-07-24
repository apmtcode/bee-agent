import type { ReplayTimelineEvent } from "../capture/replay.js";

/**
 * Local-movement learning subsystem — model layer (objective #2, parts c & d).
 *
 * bee-agent records movements/actions into replayable trajectories
 * (`src/capture`) and can plan real on-device training runs (`runner.ts`,
 * MLX/axolotl). What was missing was a *pluggable model backend* that can
 * actually train on a movement dataset and infer the next movement — in a way
 * that runs deterministically in the cloud/CI, with a documented seam for a
 * real on-device small model.
 *
 * This module provides:
 *   - a discrete tokenisation of replay-timeline events into "movement tokens",
 *   - a dataset shape built from replay manifests,
 *   - a `MovementModelBackend` interface (the pluggable seam),
 *   - a deterministic n-gram backend (the mock/default) that *learns* from the
 *     dataset, *reproduces* recorded movement sequences, and *generalises* to
 *     new-but-related contexts via backoff, and
 *   - a generalisation eval harness for held-out sequences.
 *
 * The n-gram backend is intentionally free of randomness so cloud tests are
 * reproducible. A real on-device backend (e.g. an MLX-trained small model) can
 * implement the same `MovementModelBackend` interface and be swapped in without
 * touching call sites.
 */

export type MovementToken = string;

/** Terminal token appended to sequences so a model can learn to stop. */
export const MOVEMENT_END_TOKEN = "<end>";

export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

export type MovementDataset = {
  sequences: MovementSequence[];
};

/**
 * Convert a replay-timeline event into a discrete, order-preserving token.
 * Tokens are intentionally coarse (kind + primary identifier) so the model
 * generalises across sessions rather than memorising free-text summaries.
 */
export function tokenizeMovementEvent(event: ReplayTimelineEvent): MovementToken {
  switch (event.kind) {
    case "action":
      return `action:${event.tool}`;
    case "observation":
      return `observation:${event.source}`;
    case "transcript":
      return `transcript:${event.role}`;
  }
}

export type ReplayLike = {
  sessionId?: string;
  trajectoryIds?: string[];
  events: ReplayTimelineEvent[];
};

export type BuildMovementDatasetOptions = {
  /** Restrict which event kinds become tokens. Defaults to all kinds. */
  includeKinds?: ReplayTimelineEvent["kind"][];
  /** Append {@link MOVEMENT_END_TOKEN} to each sequence. Defaults to true. */
  appendEndToken?: boolean;
};

/**
 * Build a movement dataset from replay manifests. Each replay becomes one token
 * sequence, preserving the manifest's chronological event order.
 */
export function buildMovementDataset(
  replays: ReplayLike[],
  options: BuildMovementDatasetOptions = {},
): MovementDataset {
  const includeKinds = options.includeKinds;
  const appendEndToken = options.appendEndToken ?? true;
  const sequences = replays.map((replay, index) => {
    const filtered = includeKinds
      ? replay.events.filter((event) => includeKinds.includes(event.kind))
      : replay.events;
    const tokens = filtered.map(tokenizeMovementEvent);
    if (appendEndToken) {
      tokens.push(MOVEMENT_END_TOKEN);
    }
    return {
      id: replay.sessionId ?? `replay-${index}`,
      tokens,
    };
  });
  return { sequences };
}

export type MovementPredictionCandidate = {
  token: MovementToken;
  probability: number;
  count: number;
};

export type MovementPrediction = {
  /** The most-likely next token, or undefined if the model is empty. */
  token: MovementToken | undefined;
  probability: number;
  /** Full candidate list for the chosen backoff order, most-likely first. */
  candidates: MovementPredictionCandidate[];
  /** Context length actually used to make the prediction (0 = unigram). */
  backoffOrder: number;
};

export type MovementModelState = {
  version: 1;
  backendId: string;
  order: number;
  vocabulary: MovementToken[];
  /** context-key -> next-token -> count. Context-key encodes its own length. */
  transitions: Record<string, Record<MovementToken, number>>;
  sequenceCount: number;
  tokenCount: number;
};

export type MovementGenerateOptions = {
  maxSteps?: number;
  stopAtEndToken?: boolean;
};

export interface MovementModel {
  readonly backendId: string;
  readonly order: number;
  readonly vocabulary: readonly MovementToken[];
  /** Predict the next movement token given the preceding context. */
  predictNext(context: MovementToken[]): MovementPrediction;
  /** Autoregressively generate a movement sequence from a seed context. */
  generate(seed: MovementToken[], options?: MovementGenerateOptions): MovementToken[];
  /** Serialise to a plain, replayable state object. */
  serialize(): MovementModelState;
}

export type MovementTrainOptions = {
  /** Maximum context length (n-1). Defaults to the backend's configured order. */
  order?: number;
};

/**
 * Pluggable seam: a training backend produces a {@link MovementModel} from a
 * dataset and can restore one from serialized state. Swap the deterministic
 * n-gram backend for a real on-device model backend without changing callers.
 */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainOptions): Promise<MovementModel>;
  restore(state: MovementModelState): MovementModel;
}

const CONTEXT_SEPARATOR = "";

function contextKey(order: number, context: MovementToken[]): string {
  // Length prefix keeps unigram ("0|") distinct from any real token.
  return `${order}|${context.join(CONTEXT_SEPARATOR)}`;
}

/**
 * Deterministic n-gram next-movement model with stupid-backoff generalisation.
 * Tie-breaks are resolved by (count desc, token asc) so identical datasets
 * always yield identical predictions — essential for reproducible cloud tests.
 */
class NgramMovementModel implements MovementModel {
  readonly vocabulary: readonly MovementToken[];

  constructor(
    readonly backendId: string,
    readonly order: number,
    private readonly transitions: Record<string, Record<MovementToken, number>>,
    vocabulary: MovementToken[],
    private readonly sequenceCount: number,
    private readonly tokenCount: number,
  ) {
    this.vocabulary = vocabulary;
  }

  predictNext(context: MovementToken[]): MovementPrediction {
    const maxOrder = Math.min(this.order, context.length);
    for (let k = maxOrder; k >= 0; k -= 1) {
      const suffix = k === 0 ? [] : context.slice(context.length - k);
      const counts = this.transitions[contextKey(k, suffix)];
      if (!counts) {
        continue;
      }
      const candidates = rankCandidates(counts);
      if (candidates.length === 0) {
        continue;
      }
      const best = candidates[0];
      return {
        token: best.token,
        probability: best.probability,
        candidates,
        backoffOrder: k,
      };
    }
    return { token: undefined, probability: 0, candidates: [], backoffOrder: 0 };
  }

  generate(seed: MovementToken[], options: MovementGenerateOptions = {}): MovementToken[] {
    const maxSteps = options.maxSteps ?? 64;
    const stopAtEndToken = options.stopAtEndToken ?? true;
    const context = [...seed];
    const produced: MovementToken[] = [];
    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predictNext(context);
      const token = prediction.token;
      if (token === undefined) {
        break;
      }
      if (stopAtEndToken && token === MOVEMENT_END_TOKEN) {
        break;
      }
      produced.push(token);
      context.push(token);
    }
    return produced;
  }

  serialize(): MovementModelState {
    const transitions: Record<string, Record<MovementToken, number>> = {};
    for (const [key, counts] of Object.entries(this.transitions)) {
      transitions[key] = { ...counts };
    }
    return {
      version: 1,
      backendId: this.backendId,
      order: this.order,
      vocabulary: [...this.vocabulary],
      transitions,
      sequenceCount: this.sequenceCount,
      tokenCount: this.tokenCount,
    };
  }
}

function rankCandidates(counts: Record<MovementToken, number>): MovementPredictionCandidate[] {
  const entries = Object.entries(counts);
  const total = entries.reduce((sum, [, count]) => sum + count, 0);
  if (total === 0) {
    return [];
  }
  return entries
    .map(([token, count]) => ({ token, count, probability: count / total }))
    .sort((a, b) => {
      if (b.count !== a.count) {
        return b.count - a.count;
      }
      return a.token < b.token ? -1 : a.token > b.token ? 1 : 0;
    });
}

export type NgramMovementBackendOptions = {
  id?: string;
  order?: number;
};

export class NgramMovementBackend implements MovementModelBackend {
  readonly id: string;
  private readonly order: number;

  constructor(options: NgramMovementBackendOptions = {}) {
    this.id = options.id ?? "ngram-mock";
    this.order = Math.max(1, options.order ?? 2);
  }

  async train(dataset: MovementDataset, options: MovementTrainOptions = {}): Promise<MovementModel> {
    const order = Math.max(1, options.order ?? this.order);
    const transitions: Record<string, Record<MovementToken, number>> = {};
    const vocabulary = new Set<MovementToken>();
    let tokenCount = 0;

    for (const sequence of dataset.sequences) {
      const tokens = sequence.tokens;
      for (let i = 0; i < tokens.length; i += 1) {
        const next = tokens[i];
        vocabulary.add(next);
        tokenCount += 1;
        const maxK = Math.min(order, i);
        for (let k = 0; k <= maxK; k += 1) {
          const suffix = k === 0 ? [] : tokens.slice(i - k, i);
          const key = contextKey(k, suffix);
          const counts = transitions[key] ?? (transitions[key] = {});
          counts[next] = (counts[next] ?? 0) + 1;
        }
      }
    }

    return new NgramMovementModel(
      this.id,
      order,
      transitions,
      [...vocabulary].sort(),
      dataset.sequences.length,
      tokenCount,
    );
  }

  restore(state: MovementModelState): MovementModel {
    const transitions: Record<string, Record<MovementToken, number>> = {};
    for (const [key, counts] of Object.entries(state.transitions)) {
      transitions[key] = { ...counts };
    }
    return new NgramMovementModel(
      state.backendId,
      state.order,
      transitions,
      [...state.vocabulary],
      state.sequenceCount,
      state.tokenCount,
    );
  }
}

export type MovementEvalPerSequence = {
  id: string;
  predictions: number;
  correct: number;
  accuracy: number;
};

export type MovementEvalResult = {
  sequenceCount: number;
  predictionCount: number;
  correct: number;
  /** Fraction of positions where the argmax prediction matched the actual token. */
  accuracy: number;
  topKCorrect: number;
  /** Fraction of positions where the actual token was among the top-K candidates. */
  topKAccuracy: number;
  topK: number;
  perSequence: MovementEvalPerSequence[];
};

export type EvaluateMovementModelOptions = {
  topK?: number;
};

/**
 * Generalisation eval: for each held-out sequence, predict every token from its
 * preceding context and measure next-token accuracy (and top-K recall). Run on
 * sequences the model was *not* trained on to measure generalisation rather than
 * memorisation.
 */
export function evaluateMovementModel(
  model: MovementModel,
  heldOut: MovementDataset,
  options: EvaluateMovementModelOptions = {},
): MovementEvalResult {
  const topK = Math.max(1, options.topK ?? 3);
  let predictionCount = 0;
  let correct = 0;
  let topKCorrect = 0;
  const perSequence: MovementEvalPerSequence[] = [];

  for (const sequence of heldOut.sequences) {
    const tokens = sequence.tokens;
    let seqPredictions = 0;
    let seqCorrect = 0;
    for (let i = 1; i < tokens.length; i += 1) {
      const context = tokens.slice(0, i);
      const actual = tokens[i];
      const prediction = model.predictNext(context);
      seqPredictions += 1;
      predictionCount += 1;
      if (prediction.token === actual) {
        seqCorrect += 1;
        correct += 1;
      }
      if (prediction.candidates.slice(0, topK).some((candidate) => candidate.token === actual)) {
        topKCorrect += 1;
      }
    }
    perSequence.push({
      id: sequence.id,
      predictions: seqPredictions,
      correct: seqCorrect,
      accuracy: seqPredictions === 0 ? 0 : seqCorrect / seqPredictions,
    });
  }

  return {
    sequenceCount: heldOut.sequences.length,
    predictionCount,
    correct,
    accuracy: predictionCount === 0 ? 0 : correct / predictionCount,
    topKCorrect,
    topKAccuracy: predictionCount === 0 ? 0 : topKCorrect / predictionCount,
    topK,
    perSequence,
  };
}
