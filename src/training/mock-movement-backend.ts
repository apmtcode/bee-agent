import type {
  MovementModel,
  MovementPrediction,
  MovementTrainingBackend,
  MovementTrainingRequest,
} from "./movement-model.js";

/**
 * Deterministic, dependency-free movement backend used for cloud/CI validation
 * and as the reference implementation of {@link MovementTrainingBackend}.
 *
 * It learns a backoff n-gram (Markov) model: for every training position it
 * records the observed next token conditioned on each context length from 0
 * (unigram) up to the configured order. At inference time it uses the longest
 * context that was seen during training and backs off to shorter contexts
 * otherwise — which is what lets it both *repeat* recorded movements (long,
 * specific context matched) and *generalize* to new-but-related movements
 * (recombining transitions via shorter shared context).
 *
 * A real on-device small model can replace this by implementing the same
 * interface; no call site changes.
 */

const CONTEXT_SEPARATOR = "";

export type MarkovMovementModel = MovementModel & {
  metadata: MovementModel["metadata"] & { backend: "mock-markov" };
  /** context-key -> (nextToken -> observed count). Includes every backoff order 0..order. */
  transitions: Record<string, Record<string, number>>;
};

export class MockMarkovMovementBackend implements MovementTrainingBackend<MarkovMovementModel> {
  readonly id = "mock-markov";

  async train(request: MovementTrainingRequest): Promise<MarkovMovementModel> {
    const order = Math.max(0, Math.floor(request.config.order));
    const transitions: Record<string, Record<string, number>> = {};
    let tokenCount = 0;

    for (const sequence of request.dataset.sequences) {
      const tokens = sequence.tokens;
      tokenCount += tokens.length;
      for (let index = 0; index < tokens.length; index += 1) {
        const target = tokens[index];
        // Record the target under every context length we can afford (0..order),
        // building the full backoff table in one pass.
        for (let contextLength = 0; contextLength <= order && contextLength <= index; contextLength += 1) {
          const context = tokens.slice(index - contextLength, index);
          const key = contextKey(context);
          const bucket = (transitions[key] ??= {});
          bucket[target] = (bucket[target] ?? 0) + 1;
        }
      }
    }

    return {
      metadata: {
        backend: "mock-markov",
        order,
        sequenceCount: request.dataset.sequences.length,
        tokenCount,
        vocabulary: [...request.dataset.vocabulary],
      },
      transitions,
    };
  }

  predictNext(model: MarkovMovementModel, context: readonly string[]): MovementPrediction {
    const maxOrder = Math.min(model.metadata.order, context.length);
    for (let contextLength = maxOrder; contextLength >= 0; contextLength -= 1) {
      const key = contextKey(context.slice(context.length - contextLength));
      const bucket = model.transitions[key];
      if (bucket && Object.keys(bucket).length > 0) {
        return toPrediction(bucket, contextLength);
      }
    }
    return { token: null, probability: 0, distribution: [], contextOrderUsed: 0 };
  }
}

function contextKey(context: readonly string[]): string {
  return context.join(CONTEXT_SEPARATOR);
}

function toPrediction(bucket: Record<string, number>, contextOrderUsed: number): MovementPrediction {
  const total = Object.values(bucket).reduce((sum, count) => sum + count, 0);
  const distribution = Object.entries(bucket)
    .map(([token, count]) => ({ token, probability: count / total }))
    // Highest probability first; ties broken by token so decoding is deterministic.
    .sort((a, b) => (b.probability - a.probability) || a.token.localeCompare(b.token));

  const top = distribution[0];
  return {
    token: top.token,
    probability: top.probability,
    distribution,
    contextOrderUsed,
  };
}
