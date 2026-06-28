import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";

/**
 * Local-movement learning: a pluggable, on-device model backend that learns
 * recorded movement/action sequences and can both *repeat* them and
 * *generalize* to new-but-related movements.
 *
 * This module is the cloud-testable core of standing objective #2(c)/#2(d): the
 * real on-device training (MLX/axolotl shell plans live in `runner.ts`) executes
 * when the user runs bee-agent locally, but the dataset shaping, the model
 * interface, a deterministic in-process backend, and the evaluation harness all
 * run and are validated here with synthetic/replayed event streams.
 *
 * The default backend is a deterministic order-k Markov (n-gram) model with
 * stupid-backoff. It needs no native deps, so it trains and infers in CI, yet it
 * exhibits the two behaviours the objective demands:
 *   - *repeat*: greedy generation from a recorded seed reproduces the recorded
 *     sequence (argmax over learned transitions).
 *   - *generalize*: action summaries are templatized (concrete coordinates /
 *     indices collapse to `#`), so a movement performed at new positions maps to
 *     the same learned token sequence; and backoff lets an unseen higher-order
 *     prefix fall back to the most specific context the model *has* seen.
 */

export type MovementToken = string;

export type MovementSequence = {
  trajectoryId: string;
  sessionId: string;
  tokens: MovementToken[];
};

export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
  /** Sorted, de-duplicated set of every token observed across all sequences. */
  vocabulary: MovementToken[];
};

export type MovementCandidate = {
  token: MovementToken;
  probability: number;
};

export type MovementPrediction = {
  /** Most likely next token, or `null` when the model has learned nothing. */
  token: MovementToken | null;
  probability: number;
  /** Context order (number of preceding tokens) the prediction backed off to. */
  order: number;
  /** All candidates for the backed-off context, sorted high→low then by token. */
  candidates: MovementCandidate[];
};

export type MovementModelSnapshot = {
  version: 1;
  backend: string;
  order: number;
  /** Per-order transition counts: `orders[n]` maps a context key → token → count. */
  orders: Array<Record<string, Record<MovementToken, number>>>;
};

export interface MovementModel {
  readonly backend: string;
  readonly order: number;
  /** Predict the next movement token given the trailing context. */
  predictNext(context: MovementToken[]): MovementPrediction;
  /** Greedily extend `seed` by up to `steps` tokens (stops if nothing learned). */
  generate(seed: MovementToken[], steps: number): MovementToken[];
  /** Serialize to a plain JSON snapshot for persistence alongside artifacts. */
  toJSON(): MovementModelSnapshot;
}

export type MovementTrainingOptions = {
  /** Maximum Markov order; defaults to the backend's configured order. */
  order?: number;
};

export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainingOptions): MovementModel;
  restore(snapshot: MovementModelSnapshot): MovementModel;
}

const CONTEXT_SEPARATOR = "";

/**
 * Collapse a free-text action summary into a movement *template* by replacing
 * numeric runs (coordinates, indices, pixel deltas) with `#`. This is what lets
 * the model treat "click at (12, 40)" and "click at (880, 5)" as the same
 * learned movement class — the basis for generalization.
 */
export function templatizeSummary(summary: string): string {
  return summary
    .trim()
    .toLowerCase()
    .replace(/\s+/g, " ")
    .replace(/-?\d+(?:\.\d+)?/g, "#");
}

export function actionToMovementToken(action: { tool: string; summary: string }): MovementToken {
  return `${action.tool}|${templatizeSummary(action.summary)}`;
}

/**
 * Shape one or more replay manifests into a movement dataset: one sequence per
 * trajectory, ordered by timestamp, containing only `action` events (the
 * movements a model should learn to reproduce).
 */
export function buildMovementDataset(replays: ReplayManifest[]): MovementDataset {
  const sequences: MovementSequence[] = [];
  const vocabulary = new Set<MovementToken>();

  for (const replay of replays) {
    const byTrajectory = new Map<string, Array<{ ts: number; token: MovementToken }>>();
    for (const event of replay.events) {
      if (event.kind !== "action") {
        continue;
      }
      const token = actionToMovementToken(event);
      vocabulary.add(token);
      const list = byTrajectory.get(event.trajectoryId) ?? [];
      list.push({ ts: event.ts, token });
      byTrajectory.set(event.trajectoryId, list);
    }
    for (const [trajectoryId, items] of byTrajectory) {
      const tokens = items
        .slice()
        .sort((a, b) => a.ts - b.ts)
        .map((item) => item.token);
      if (tokens.length > 0) {
        sequences.push({ trajectoryId, sessionId: replay.sessionId, tokens });
      }
    }
  }

  return {
    version: 1,
    sequences: sequences.sort((a, b) => a.trajectoryId.localeCompare(b.trajectoryId)),
    vocabulary: [...vocabulary].sort(),
  };
}

function contextKey(context: MovementToken[]): string {
  return context.join(CONTEXT_SEPARATOR);
}

function argmax(distribution: Record<MovementToken, number>): MovementCandidate[] {
  const total = Object.values(distribution).reduce((sum, count) => sum + count, 0);
  if (total === 0) {
    return [];
  }
  return Object.entries(distribution)
    .map(([token, count]) => ({ token, probability: count / total }))
    .sort((a, b) => {
      if (b.probability !== a.probability) {
        return b.probability - a.probability;
      }
      return a.token.localeCompare(b.token);
    });
}

class NgramMovementModel implements MovementModel {
  constructor(
    readonly backend: string,
    readonly order: number,
    private readonly orders: Array<Map<string, Record<MovementToken, number>>>,
  ) {}

  predictNext(context: MovementToken[]): MovementPrediction {
    const maxOrder = Math.min(this.order, context.length);
    for (let n = maxOrder; n >= 0; n -= 1) {
      const table = this.orders[n];
      if (!table) {
        continue;
      }
      const key = contextKey(context.slice(context.length - n));
      const distribution = table.get(key);
      if (!distribution) {
        continue;
      }
      const candidates = argmax(distribution);
      if (candidates.length > 0) {
        return {
          token: candidates[0].token,
          probability: candidates[0].probability,
          order: n,
          candidates,
        };
      }
    }
    return { token: null, probability: 0, order: 0, candidates: [] };
  }

  generate(seed: MovementToken[], steps: number): MovementToken[] {
    const output = [...seed];
    for (let step = 0; step < steps; step += 1) {
      const prediction = this.predictNext(output);
      if (prediction.token === null) {
        break;
      }
      output.push(prediction.token);
    }
    return output;
  }

  toJSON(): MovementModelSnapshot {
    return {
      version: 1,
      backend: this.backend,
      order: this.order,
      orders: this.orders.map((table) => {
        const record: Record<string, Record<MovementToken, number>> = {};
        for (const [key, distribution] of table) {
          record[key] = { ...distribution };
        }
        return record;
      }),
    };
  }
}

export type NgramMovementBackendOptions = {
  /** Default Markov order used when `train` is called without an override. */
  order?: number;
};

/**
 * Deterministic, dependency-free Markov movement backend. Counts token
 * transitions at every order 0..k so prediction can back off from the most
 * specific seen context down to the unigram prior.
 */
export class NgramMovementBackend implements MovementModelBackend {
  readonly id = "ngram";
  private readonly defaultOrder: number;

  constructor(options: NgramMovementBackendOptions = {}) {
    this.defaultOrder = Math.max(0, options.order ?? 2);
  }

  train(dataset: MovementDataset, options: MovementTrainingOptions = {}): MovementModel {
    const order = Math.max(0, options.order ?? this.defaultOrder);
    const orders: Array<Map<string, Record<MovementToken, number>>> = Array.from(
      { length: order + 1 },
      () => new Map<string, Record<MovementToken, number>>(),
    );

    for (const sequence of dataset.sequences) {
      const { tokens } = sequence;
      for (let i = 0; i < tokens.length; i += 1) {
        const next = tokens[i];
        for (let n = 0; n <= order; n += 1) {
          if (i - n < 0) {
            break;
          }
          const key = contextKey(tokens.slice(i - n, i));
          const table = orders[n];
          const distribution = table.get(key) ?? {};
          distribution[next] = (distribution[next] ?? 0) + 1;
          table.set(key, distribution);
        }
      }
    }

    return new NgramMovementModel(this.id, order, orders);
  }

  restore(snapshot: MovementModelSnapshot): MovementModel {
    const orders = snapshot.orders.map((record) => {
      const table = new Map<string, Record<MovementToken, number>>();
      for (const [key, distribution] of Object.entries(record)) {
        table.set(key, { ...distribution });
      }
      return table;
    });
    return new NgramMovementModel(snapshot.backend, snapshot.order, orders);
  }
}

export type MovementBackendName = "ngram";

/**
 * Backend registry. The backend is intentionally pluggable: a real on-device
 * small model (e.g. an MLX-trained policy loaded for inference) can register
 * here under a new name without touching dataset shaping or the eval harness.
 */
export function createMovementBackend(
  name: MovementBackendName = "ngram",
  options: NgramMovementBackendOptions = {},
): MovementModelBackend {
  switch (name) {
    case "ngram":
      return new NgramMovementBackend(options);
    default: {
      const exhaustive: never = name;
      throw new Error(`unknown movement backend: ${String(exhaustive)}`);
    }
  }
}

export type MovementEvalResult = {
  /** Total (context → next) predictions scored. */
  predictions: number;
  /** Predictions whose argmax token matched the held-out next token. */
  correct: number;
  /** correct / predictions (0 when there were no scorable predictions). */
  accuracy: number;
  /** Per-sequence accuracy, keyed by trajectory id, for drill-down. */
  perSequence: Array<{ trajectoryId: string; predictions: number; correct: number }>;
};

/**
 * Generalization eval harness: score next-action prediction on held-out
 * sequences. Each token (after the first) is predicted from its true prefix and
 * compared against ground truth — measuring how well a model trained on one set
 * of movements reproduces *related* movements it was not trained on.
 */
export function evaluateNextActionAccuracy(
  model: MovementModel,
  sequences: MovementSequence[],
): MovementEvalResult {
  let predictions = 0;
  let correct = 0;
  const perSequence: MovementEvalResult["perSequence"] = [];

  for (const sequence of sequences) {
    let seqPredictions = 0;
    let seqCorrect = 0;
    for (let i = 1; i < sequence.tokens.length; i += 1) {
      const prediction = model.predictNext(sequence.tokens.slice(0, i));
      seqPredictions += 1;
      if (prediction.token === sequence.tokens[i]) {
        seqCorrect += 1;
      }
    }
    predictions += seqPredictions;
    correct += seqCorrect;
    perSequence.push({ trajectoryId: sequence.trajectoryId, predictions: seqPredictions, correct: seqCorrect });
  }

  return {
    predictions,
    correct,
    accuracy: predictions === 0 ? 0 : correct / predictions,
    perSequence,
  };
}

/** Re-exported for callers shaping events without importing the replay module. */
export type { ReplayTimelineEvent };
