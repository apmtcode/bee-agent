import {
  MOVEMENT_END_TOKEN,
  MOVEMENT_START_TOKEN,
  type MovementDataset,
  type MovementModelBackend,
  type MovementPrediction,
  type MovementPredictOptions,
  type MovementToken,
  type TrainedMovementModel,
} from "./types.js";

const CONTEXT_SEPARATOR = "";

/** JSON-serializable trained state for the Markov backend. */
type MarkovModelState = {
  order: number;
  /** `orders[k]` maps a k-token context key -> token -> observed count. */
  orders: Record<string, Record<MovementToken, number>>[];
};

export type MarkovMovementBackendOptions = {
  /**
   * Maximum context length (highest n-gram order). Order 2 conditions the next
   * movement on the previous two. Prediction backs off to shorter contexts when
   * the full context was never seen, which is what yields generalization to
   * new-but-related movement sequences.
   */
  order?: number;
};

/**
 * Deterministic n-gram movement model with stupid-backoff inference.
 *
 * Fully in-process and dependency-free: it "post-trains" on the recorded
 * dataset by counting movement transitions, "repeats" recorded movements by
 * argmax rollout, and "generalizes" to unseen contexts by backing off to
 * lower-order statistics. It is the default {@link MovementModelBackend}; a real
 * on-device small model can be dropped in behind the same interface.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly name = "markov-ngram";
  private readonly order: number;

  constructor(options: MarkovMovementBackendOptions = {}) {
    this.order = Math.max(1, Math.floor(options.order ?? 2));
  }

  train(dataset: MovementDataset): TrainedMovementModel {
    const orders: Record<string, Record<MovementToken, number>>[] = Array.from(
      { length: this.order + 1 },
      () => ({}),
    );
    const vocabulary = new Set<MovementToken>();
    let transitionCount = 0;

    for (const sequence of dataset.sequences) {
      const tokens = [
        ...Array.from({ length: this.order }, () => MOVEMENT_START_TOKEN),
        ...sequence.steps.map((step) => step.token),
        MOVEMENT_END_TOKEN,
      ];
      for (const step of sequence.steps) {
        vocabulary.add(step.token);
      }
      for (let i = this.order; i < tokens.length; i += 1) {
        const next = tokens[i]!;
        for (let k = 0; k <= this.order; k += 1) {
          const context = tokens.slice(i - k, i);
          const key = contextKey(context);
          const bucket = (orders[k]![key] ??= {});
          bucket[next] = (bucket[next] ?? 0) + 1;
          transitionCount += 1;
        }
      }
    }

    const state: MarkovModelState = { order: this.order, orders };
    return {
      metadata: {
        backend: this.name,
        order: this.order,
        vocabularySize: vocabulary.size,
        sequenceCount: dataset.sequences.length,
        transitionCount,
      },
      state,
    };
  }

  predict(
    model: TrainedMovementModel,
    context: readonly MovementToken[],
    options: MovementPredictOptions = {},
  ): MovementPrediction[] {
    const state = model.state as MarkovModelState;
    const exclude = new Set(options.exclude ?? []);
    const topK = options.topK ?? Infinity;

    for (let k = Math.min(state.order, context.length); k >= 0; k -= 1) {
      const key = contextKey(context.slice(context.length - k));
      const bucket = state.orders[k]?.[key];
      if (!bucket) {
        continue;
      }
      const total = Object.values(bucket).reduce((sum, count) => sum + count, 0);
      const ranked = Object.entries(bucket)
        .filter(([token]) => !exclude.has(token))
        .map<MovementPrediction>(([token, count]) => ({
          token,
          probability: count / total,
          order: k,
        }))
        .sort(comparePredictions);
      if (ranked.length > 0) {
        return Number.isFinite(topK) ? ranked.slice(0, topK) : ranked;
      }
    }
    return [];
  }

  generate(
    model: TrainedMovementModel,
    seed: readonly MovementToken[],
    maxSteps: number,
  ): MovementToken[] {
    const context: MovementToken[] = [...seed];
    const output: MovementToken[] = [];
    for (let step = 0; step < maxSteps; step += 1) {
      const [prediction] = this.predict(model, context, { topK: 1 });
      if (!prediction || prediction.token === MOVEMENT_END_TOKEN) {
        break;
      }
      output.push(prediction.token);
      context.push(prediction.token);
    }
    return output;
  }
}

function contextKey(context: readonly MovementToken[]): string {
  return context.join(CONTEXT_SEPARATOR);
}

function comparePredictions(a: MovementPrediction, b: MovementPrediction): number {
  if (b.probability !== a.probability) {
    return b.probability - a.probability;
  }
  // Deterministic tie-break so argmax rollout is reproducible.
  return a.token < b.token ? -1 : a.token > b.token ? 1 : 0;
}
