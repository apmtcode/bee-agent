// Pluggable local-model backend for movement learning.
//
// Objective #2(c/d): post-train a local model on recorded movements so it can
// repeat them and generalize to new-but-related movements. The training step
// runs on-device on the user's machine with a real small model; to keep the
// backend swappable — and to keep cloud/CI deterministic — we define a narrow
// `MovementModelBackend` interface and ship a dependency-free, fully
// deterministic reference implementation.
//
// The reference backend is a variable-order Markov model with stupid-backoff.
// It is a genuine sequence learner: it estimates P(next token | last k tokens)
// from the tokenized dataset and, at inference, uses the longest context it has
// seen, backing off to shorter contexts when a full context is novel. That
// backoff is exactly what lets it generalize — a held-out trajectory that
// recombines sub-patterns seen during training is still predicted correctly.

import {
  MOVEMENT_EOS,
  tokenizeTrajectory,
  type CoordinateQuantizer,
  type MovementDataset,
} from "./events.js";

export type MovementPrediction = {
  token: string;
  probability: number;
  /** Context order (in tokens) that produced this prediction after backoff. */
  order: number;
};

/** A model that has been trained on a movement dataset and can predict tokens. */
export interface TrainedMovementModel {
  readonly backendId: string;
  readonly quantizer: CoordinateQuantizer;
  /** Predict the most likely next token given a context (recent tokens). */
  predictNextToken(context: string[]): MovementPrediction | undefined;
  /** Greedily roll out a token sequence from a seed until EOS or maxSteps. */
  generate(seed: string[], maxSteps: number): string[];
  /** Serializable snapshot for persistence / inspection. */
  snapshot(): TrainedMovementModelSnapshot;
}

export type TrainedMovementModelSnapshot = {
  backendId: string;
  order: number;
  vocabularySize: number;
  contextCount: number;
};

export type TrainMovementModelOptions = {
  /** Maximum context length (Markov order). Higher = more specific, less general. */
  maxOrder?: number;
};

/** The swappable backend seam. Real backends (on-device small models) implement this. */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: TrainMovementModelOptions): Promise<TrainedMovementModel>;
}

type ContextTable = Map<string, Map<string, number>>;

/**
 * Deterministic reference backend. No external model runtime, no randomness —
 * suitable for cloud/CI. Ties in argmax are broken by lexical token order so the
 * output is stable across runs and platforms.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov-backoff";

  async train(
    dataset: MovementDataset,
    options: TrainMovementModelOptions = {},
  ): Promise<TrainedMovementModel> {
    const maxOrder = Math.max(1, options.maxOrder ?? 3);
    // tables[k] maps a k-token context (joined) -> next-token counts.
    const tables: ContextTable[] = Array.from({ length: maxOrder + 1 }, () => new Map());
    const vocabulary = new Set<string>();

    for (const trajectory of dataset.trajectories) {
      const tokens = tokenizeTrajectory(trajectory, dataset.quantizer);
      for (const token of tokens) {
        vocabulary.add(token);
      }
      for (let i = 1; i < tokens.length; i += 1) {
        const next = tokens[i]!;
        for (let order = 0; order <= maxOrder; order += 1) {
          if (order > i) break;
          const context = tokens.slice(i - order, i);
          const key = joinContext(context);
          const table = tables[order]!;
          const counts = table.get(key) ?? new Map<string, number>();
          counts.set(next, (counts.get(next) ?? 0) + 1);
          table.set(key, counts);
        }
      }
    }

    return new MarkovTrainedModel(this.id, dataset.quantizer, maxOrder, tables, vocabulary.size);
  }
}

class MarkovTrainedModel implements TrainedMovementModel {
  constructor(
    readonly backendId: string,
    readonly quantizer: CoordinateQuantizer,
    private readonly maxOrder: number,
    private readonly tables: ContextTable[],
    private readonly vocabularySize: number,
  ) {}

  predictNextToken(context: string[]): MovementPrediction | undefined {
    // Stupid-backoff: try the longest context we have counts for, then shorten.
    for (let order = Math.min(this.maxOrder, context.length); order >= 0; order -= 1) {
      const slice = context.slice(context.length - order);
      const counts = this.tables[order]?.get(joinContext(slice));
      if (!counts || counts.size === 0) continue;
      const best = argmax(counts);
      if (!best) continue;
      const total = totalCount(counts);
      return { token: best, probability: total > 0 ? (counts.get(best) ?? 0) / total : 0, order };
    }
    return undefined;
  }

  generate(seed: string[], maxSteps: number): string[] {
    const out = [...seed];
    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predictNextToken(out);
      if (!prediction) break;
      out.push(prediction.token);
      if (prediction.token === MOVEMENT_EOS) break;
    }
    return out;
  }

  snapshot(): TrainedMovementModelSnapshot {
    let contextCount = 0;
    for (const table of this.tables) {
      contextCount += table.size;
    }
    return {
      backendId: this.backendId,
      order: this.maxOrder,
      vocabularySize: this.vocabularySize,
      contextCount,
    };
  }
}

function joinContext(context: string[]): string {
  return context.join("␟"); // unit separator, cannot appear in tokens
}

function totalCount(counts: Map<string, number>): number {
  let total = 0;
  for (const value of counts.values()) total += value;
  return total;
}

/** Deterministic argmax: highest count, ties broken by lexical token order. */
function argmax(counts: Map<string, number>): string | undefined {
  let bestToken: string | undefined;
  let bestCount = -Infinity;
  for (const [token, count] of counts) {
    if (count > bestCount || (count === bestCount && (bestToken === undefined || token < bestToken))) {
      bestToken = token;
      bestCount = count;
    }
  }
  return bestToken;
}
