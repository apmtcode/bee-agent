/**
 * Movement-policy model backend (standing objective #2c/#2d).
 *
 * This module provides the first *in-process, cloud-runnable* model backend for
 * the local-movement learning subsystem. The existing `runner.ts` only renders
 * launch scripts that shell out to `mlx`/`axolotl` on the user's Apple-silicon
 * machine -- nothing that can be trained or evaluated inside the cloud engine or
 * CI. This backend closes that gap with a deterministic, dependency-free
 * sequence model that:
 *
 *   (c) *repeats* recorded movements -- feeding a recorded prefix reproduces the
 *       recorded next action at the highest matching context order; and
 *   (d) *generalizes* to new-but-related movements -- for an unseen prefix that
 *       shares a suffix with the training data, it backs off to shorter matched
 *       contexts and predicts a plausible next movement.
 *
 * The backend is intentionally pluggable: {@link MovementPolicyBackend} is the
 * seam a real on-device small model implements later. The n-gram backend here
 * is the deterministic mock that keeps cloud/CI green while exercising the whole
 * train -> predict -> rollout -> evaluate pipeline.
 */

/** A single recorded movement, tokenized to a device-agnostic action label. */
export type MovementToken = string;

/** An ordered sequence of movements captured from one trajectory/session. */
export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

/** The training dataset: a bag of movement sequences. */
export type MovementDataset = {
  sequences: MovementSequence[];
};

export type MovementPredictionSource = "exact" | "backoff" | "unigram" | "none";

/** A next-movement prediction with provenance for explainability/eval. */
export type MovementPrediction = {
  /** Predicted next movement, or `undefined` when the model has no evidence. */
  token: MovementToken | undefined;
  /** Relative frequency of `token` within the matched context (0..1). */
  confidence: number;
  /** Length of the prior context actually matched (0 = unigram/prior). */
  contextOrder: number;
  source: MovementPredictionSource;
};

/** A trained model instance produced by a {@link MovementPolicyBackend}. */
export interface TrainedMovementModel {
  readonly backend: string;
  readonly maxOrder: number;
  /** Predict the next movement given a (possibly empty) prior context. */
  predict(context: readonly MovementToken[]): MovementPrediction;
  /**
   * Deterministically roll out a continuation of `seed`, appending predicted
   * movements until `maxSteps` is reached or the model runs out of evidence.
   * Returns only the newly generated tail (not the seed).
   */
  rollout(seed: readonly MovementToken[], maxSteps: number): MovementToken[];
}

/** Pluggable backend seam: the real on-device model implements this later. */
export interface MovementPolicyBackend {
  readonly name: string;
  train(dataset: MovementDataset): TrainedMovementModel;
}

export type NgramMovementBackendOptions = {
  /** Longest prior-context length to model (n-gram order). Default 3. */
  maxOrder?: number;
};

// Unit-separator delimiter: collision-safe against arbitrary token contents
// (dotted identifiers, replay tool names) while keeping this source pure ASCII.
const CONTEXT_SEPARATOR = "\u001f";

function contextKey(context: readonly MovementToken[]): string {
  return context.join(CONTEXT_SEPARATOR);
}

type CountRow = Map<MovementToken, number>;
type CountTable = Map<string, CountRow>;

/**
 * Pick the highest-count token in a row. Ties break lexicographically so the
 * model is fully deterministic (important for reproducible cloud/CI runs).
 */
function bestToken(row: CountRow): { token: MovementToken; count: number; total: number } {
  let bestTok: MovementToken | undefined;
  let bestCount = -1;
  let total = 0;
  for (const [token, count] of row) {
    total += count;
    if (count > bestCount || (count === bestCount && (bestTok === undefined || token < bestTok))) {
      bestCount = count;
      bestTok = token;
    }
  }
  return { token: bestTok ?? "", count: bestCount, total };
}

class NgramMovementModel implements TrainedMovementModel {
  constructor(
    readonly backend: string,
    readonly maxOrder: number,
    private readonly tables: CountTable[],
  ) {}

  predict(context: readonly MovementToken[]): MovementPrediction {
    const startOrder = Math.min(this.maxOrder, context.length);
    for (let k = startOrder; k >= 0; k -= 1) {
      const table = this.tables[k];
      if (!table) {
        continue;
      }
      const row = table.get(contextKey(context.slice(context.length - k)));
      if (row && row.size > 0) {
        const { token, count, total } = bestToken(row);
        const source: MovementPredictionSource =
          k === 0 ? "unigram" : k === startOrder ? "exact" : "backoff";
        return {
          token,
          confidence: total > 0 ? count / total : 0,
          contextOrder: k,
          source,
        };
      }
    }
    return { token: undefined, confidence: 0, contextOrder: 0, source: "none" };
  }

  rollout(seed: readonly MovementToken[], maxSteps: number): MovementToken[] {
    const context: MovementToken[] = [...seed];
    const generated: MovementToken[] = [];
    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predict(context);
      if (prediction.token === undefined) {
        break;
      }
      generated.push(prediction.token);
      context.push(prediction.token);
    }
    return generated;
  }
}

/**
 * Deterministic n-gram sequence-model backend -- the default local mock. It
 * memorizes recorded transitions (enabling exact repeat) and backs off to
 * shorter contexts (enabling generalization to related movements).
 */
export class NgramMovementPolicyBackend implements MovementPolicyBackend {
  readonly name = "ngram-mock";
  private readonly maxOrder: number;

  constructor(options: NgramMovementBackendOptions = {}) {
    this.maxOrder = Math.max(1, Math.floor(options.maxOrder ?? 3));
  }

  train(dataset: MovementDataset): TrainedMovementModel {
    const tables: CountTable[] = [];
    for (let k = 0; k <= this.maxOrder; k += 1) {
      tables.push(new Map());
    }

    for (const sequence of dataset.sequences) {
      const { tokens } = sequence;
      for (let i = 0; i < tokens.length; i += 1) {
        const next = tokens[i];
        if (next === undefined) {
          continue;
        }
        for (let k = 0; k <= this.maxOrder; k += 1) {
          if (i - k < 0) {
            break;
          }
          const key = contextKey(tokens.slice(i - k, i));
          const table = tables[k];
          if (!table) {
            continue;
          }
          let row = table.get(key);
          if (!row) {
            row = new Map();
            table.set(key, row);
          }
          row.set(next, (row.get(next) ?? 0) + 1);
        }
      }
    }

    return new NgramMovementModel(this.name, this.maxOrder, tables);
  }
}

export type MovementFidelityResult = {
  /** Total next-token predictions attempted (one per non-first token). */
  totalPredictions: number;
  /** Predictions whose token equalled the recorded next movement. */
  correct: number;
  /** correct / total (0 when there is nothing to predict). */
  accuracy: number;
  /** Fraction of predictions where the model produced any token at all. */
  coverage: number;
};

/**
 * Teacher-forced next-movement fidelity: for every prefix of every sequence,
 * ask the model to predict the true next movement and score exact matches.
 * Used both for *repeat* fidelity (evaluate on training sequences) and
 * *generalization* fidelity (evaluate on held-out related sequences).
 */
export function evaluateNextTokenFidelity(
  model: TrainedMovementModel,
  sequences: readonly MovementSequence[],
): MovementFidelityResult {
  let total = 0;
  let correct = 0;
  let predicted = 0;
  for (const sequence of sequences) {
    const { tokens } = sequence;
    for (let i = 1; i < tokens.length; i += 1) {
      const prediction = model.predict(tokens.slice(0, i));
      total += 1;
      if (prediction.token !== undefined) {
        predicted += 1;
      }
      if (prediction.token === tokens[i]) {
        correct += 1;
      }
    }
  }
  return {
    totalPredictions: total,
    correct,
    accuracy: total > 0 ? correct / total : 0,
    coverage: total > 0 ? predicted / total : 0,
  };
}
