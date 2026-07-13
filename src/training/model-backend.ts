import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";

/**
 * In-process, pluggable "local model" seam for the movement-learning subsystem.
 *
 * Standing objective #2 (c)+(d): post-train a local model on the recorded
 * movement dataset so it can (c) repeat the recorded movements and (d)
 * generalize to new-but-related movements.
 *
 * The heavy on-device training path (mlx / axolotl) is produced by
 * `LocalAppleSiliconTrainingRunner`, which only runs on the user's real
 * machine. This module provides the *backend interface* plus a fully
 * deterministic reference backend (an order-k Markov / n-gram model) so the
 * train -> infer -> evaluate loop can be exercised end-to-end in the cloud/CI
 * with no native deps and no randomness. A real small on-device model can
 * implement {@link MovementModelBackend} and slot in behind the same seam.
 */

/** A single movement step, encoded as a compact string token. */
export type MovementToken = string;

/** One recorded (or generated) ordered sequence of movement tokens. */
export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

/** The training dataset: a set of tokenized movement sequences. */
export type MovementDataset = {
  sequences: MovementSequence[];
};

/** Context handed to the model when asking for the next movement. */
export type MovementContext = {
  history: MovementToken[];
};

/** A ranked candidate for the next movement token. */
export type MovementCandidate = {
  token: MovementToken;
  probability: number;
};

/** The model's prediction for the next movement token. */
export type MovementPrediction = {
  /** Best next token, or `undefined` when the model has learned nothing. */
  token: MovementToken | undefined;
  /** Probability mass of `token` within the matched context (0..1). */
  confidence: number;
  /** Context order (suffix length) that produced the prediction; 0 = unigram. */
  order: number;
  /** Full ranked candidate list for the matched context. */
  candidates: MovementCandidate[];
};

export type TrainMovementModelOptions = {
  /** Maximum context order (n-gram length - 1). Defaults to 3. */
  order?: number;
};

export type SerializedMovementModel = {
  version: 1;
  backend: string;
  order: number;
  /** contextKey -> (token -> count). Empty-string key holds unigram counts. */
  counts: Record<string, Record<MovementToken, number>>;
};

/** A trained, queryable movement model. */
export interface TrainedMovementModel {
  readonly backend: string;
  readonly order: number;
  /** Predict the next movement token given prior history. */
  predictNext(context: MovementContext): MovementPrediction;
  /** Roll out up to `steps` predicted tokens, appending each to the history. */
  generate(context: MovementContext, steps: number): MovementToken[];
  /** Persist the model to a plain JSON structure. */
  serialize(): SerializedMovementModel;
}

/** A pluggable movement-model backend (deterministic mock or real on-device). */
export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, options?: TrainMovementModelOptions): Promise<TrainedMovementModel>;
  restore(serialized: SerializedMovementModel): TrainedMovementModel;
}

const CONTEXT_SEPARATOR = "␟"; // symbol for UNIT SEPARATOR — never appears in tokens

/** Encode a single replay timeline event as a compact movement token. */
export function tokenizeEvent(event: ReplayTimelineEvent): MovementToken {
  switch (event.kind) {
    case "action":
      return `act:${event.tool}`;
    case "observation":
      return `obs:${event.source}`;
    case "transcript":
      return `msg:${event.role}`;
  }
}

/** Build a tokenized {@link MovementSequence} from ordered replay events. */
export function sequenceFromEvents(id: string, events: ReplayTimelineEvent[]): MovementSequence {
  return { id, tokens: events.map(tokenizeEvent) };
}

/** Build a {@link MovementDataset} from a set of replay manifests. */
export function datasetFromReplays(replays: ReplayManifest[]): MovementDataset {
  return {
    sequences: replays.map((replay) => sequenceFromEvents(replay.sessionId, replay.events)),
  };
}

function contextKey(tokens: MovementToken[]): string {
  return tokens.join(CONTEXT_SEPARATOR);
}

/**
 * Deterministic reference backend: an order-k n-gram model with stupid-backoff.
 *
 * Repeats recorded movements exactly (the longest matching suffix of a recorded
 * trajectory reproduces its recorded continuation) and generalizes to related
 * movements by backing off to shorter suffixes shared across trajectories.
 * Fully deterministic — ties break by descending count then ascending token —
 * so it is safe to assert on in CI.
 */
export class NgramMovementBackend implements MovementModelBackend {
  readonly name = "ngram";

  async train(dataset: MovementDataset, options?: TrainMovementModelOptions): Promise<TrainedMovementModel> {
    const order = Math.max(1, Math.trunc(options?.order ?? 3));
    const counts = new Map<string, Map<MovementToken, number>>();

    const bump = (key: string, token: MovementToken): void => {
      let inner = counts.get(key);
      if (!inner) {
        inner = new Map<MovementToken, number>();
        counts.set(key, inner);
      }
      inner.set(token, (inner.get(token) ?? 0) + 1);
    };

    for (const sequence of dataset.sequences) {
      const tokens = sequence.tokens;
      for (let i = 0; i < tokens.length; i += 1) {
        const next = tokens[i]!;
        bump("", next); // unigram
        for (let n = 1; n <= order; n += 1) {
          if (i - n < 0) {
            break;
          }
          bump(contextKey(tokens.slice(i - n, i)), next);
        }
      }
    }

    return new NgramMovementModel(order, counts);
  }

  restore(serialized: SerializedMovementModel): TrainedMovementModel {
    if (serialized.backend !== this.name) {
      throw new Error(`Cannot restore ${serialized.backend} model with the ${this.name} backend`);
    }
    const counts = new Map<string, Map<MovementToken, number>>();
    for (const [key, tokenCounts] of Object.entries(serialized.counts)) {
      counts.set(key, new Map(Object.entries(tokenCounts)));
    }
    return new NgramMovementModel(serialized.order, counts);
  }
}

class NgramMovementModel implements TrainedMovementModel {
  readonly backend = "ngram";

  constructor(
    readonly order: number,
    private readonly counts: Map<string, Map<MovementToken, number>>,
  ) {}

  predictNext(context: MovementContext): MovementPrediction {
    const history = context.history;
    for (let n = Math.min(this.order, history.length); n >= 0; n -= 1) {
      const key = n === 0 ? "" : contextKey(history.slice(history.length - n));
      const inner = this.counts.get(key);
      if (!inner || inner.size === 0) {
        continue;
      }
      const candidates = rankCandidates(inner);
      return {
        token: candidates[0]?.token,
        confidence: candidates[0]?.probability ?? 0,
        order: n,
        candidates,
      };
    }
    return { token: undefined, confidence: 0, order: 0, candidates: [] };
  }

  generate(context: MovementContext, steps: number): MovementToken[] {
    const history = [...context.history];
    const generated: MovementToken[] = [];
    for (let i = 0; i < steps; i += 1) {
      const prediction = this.predictNext({ history });
      if (prediction.token === undefined) {
        break;
      }
      generated.push(prediction.token);
      history.push(prediction.token);
    }
    return generated;
  }

  serialize(): SerializedMovementModel {
    const serializedCounts: Record<string, Record<MovementToken, number>> = {};
    for (const [key, inner] of this.counts) {
      serializedCounts[key] = Object.fromEntries(inner);
    }
    return {
      version: 1,
      backend: this.backend,
      order: this.order,
      counts: serializedCounts,
    };
  }
}

function rankCandidates(counts: Map<MovementToken, number>): MovementCandidate[] {
  const total = [...counts.values()].reduce((sum, value) => sum + value, 0);
  return [...counts.entries()]
    .map(([token, count]) => ({ token, probability: total === 0 ? 0 : count / total }))
    .sort((a, b) => {
      if (b.probability !== a.probability) {
        return b.probability - a.probability;
      }
      return a.token < b.token ? -1 : a.token > b.token ? 1 : 0;
    });
}

export type MovementEvalResult = {
  /** Number of (context -> next) prediction points scored. */
  total: number;
  /** How many predictions matched the held-out next token exactly. */
  correct: number;
  /** correct / total (0 when total is 0). */
  accuracy: number;
};

/**
 * Generalization eval harness: score a trained model's next-token accuracy on
 * held-out sequences it was not trained on. Each position in each sequence is a
 * prediction point (context = prefix, label = the actual next token).
 */
export function evaluateNextTokenAccuracy(
  model: TrainedMovementModel,
  heldOut: MovementDataset,
): MovementEvalResult {
  let total = 0;
  let correct = 0;
  for (const sequence of heldOut.sequences) {
    for (let i = 0; i < sequence.tokens.length; i += 1) {
      const prediction = model.predictNext({ history: sequence.tokens.slice(0, i) });
      total += 1;
      if (prediction.token !== undefined && prediction.token === sequence.tokens[i]) {
        correct += 1;
      }
    }
  }
  return { total, correct, accuracy: total === 0 ? 0 : correct / total };
}
