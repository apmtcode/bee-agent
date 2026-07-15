import type {
  MovementDataset,
  MovementModelArtifact,
  MovementModelBackend,
  MovementPrediction,
  MovementToken,
  MovementTrainingConfig,
} from "./movement-model.js";

/**
 * Deterministic variable-order Markov backend for local-movement learning.
 *
 * It learns n-gram transition counts up to `maxOrder` and predicts the next
 * token via Katz-style backoff: prefer the longest recorded context, falling
 * back to shorter contexts when the exact one was never seen. This gives two
 * properties the objective requires:
 *
 *  - **Repeat**: a recorded high-order context is unique within its recording,
 *    so its single observed continuation is predicted with probability 1 — the
 *    model reproduces recorded movements exactly.
 *  - **Generalize**: an unseen high-order context backs off to a shorter,
 *    previously-seen context, so the model performs a new-but-related movement
 *    instead of failing.
 *
 * The backend runs fully in-process with no OS or native deps, so it validates
 * the capture→dataset→train→infer pipeline in the cloud. A real on-device small
 * model can be dropped in behind the same {@link MovementModelBackend}
 * interface.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov-v1";

  constructor(private readonly defaults: { maxOrder?: number; smoothing?: number } = {}) {}

  train(dataset: MovementDataset, config?: MovementTrainingConfig): MovementModelArtifact {
    const maxOrder = Math.max(0, config?.maxOrder ?? this.defaults.maxOrder ?? 3);
    const smoothing = Math.max(0, config?.smoothing ?? this.defaults.smoothing ?? 0);

    const counts: MarkovCounts = {};
    const vocabulary = new Set<MovementToken>();
    let tokenCount = 0;

    for (const sequence of dataset.sequences) {
      tokenCount += sequence.tokens.length;
      for (const token of sequence.tokens) {
        vocabulary.add(token);
      }
      // Append an end-of-sequence sentinel so the model learns where a recorded
      // movement stops; `generate` halts when it predicts it. Without this,
      // rollout would never terminate (order-0 backoff always returns a token).
      const tokens = [...sequence.tokens, END_OF_SEQUENCE];
      for (let index = 0; index < tokens.length; index += 1) {
        const next = tokens[index]!;
        for (let order = 0; order <= maxOrder; order += 1) {
          if (index - order < 0) {
            continue;
          }
          const context = tokens.slice(index - order, index);
          const key = contextKey(context);
          const orderTable = (counts[order] ??= {});
          const contextTable = (orderTable[key] ??= {});
          contextTable[next] = (contextTable[next] ?? 0) + 1;
        }
      }
    }

    const parameters: MarkovParameters = {
      maxOrder,
      smoothing,
      vocabulary: [...vocabulary].sort(),
      counts,
    };

    return {
      version: 1,
      backend: this.id,
      parameters,
      metadata: {
        sequenceCount: dataset.sequences.length,
        tokenCount,
        vocabularySize: vocabulary.size,
        maxOrder,
        smoothing,
      },
    };
  }

  predict(model: MovementModelArtifact, context: MovementToken[]): MovementPrediction {
    const params = model.parameters as MarkovParameters;
    const usableOrder = Math.min(params.maxOrder, context.length);

    for (let order = usableOrder; order >= 0; order -= 1) {
      const scoped = context.slice(context.length - order);
      const table = params.counts[order]?.[contextKey(scoped)];
      if (!table) {
        continue;
      }
      const distribution = distributionFromCounts(table, params.smoothing, params.vocabulary.length);
      if (distribution.length === 0) {
        continue;
      }
      const top = distribution[0]!;
      return {
        token: top.token,
        confidence: top.probability,
        contextOrder: order,
        distribution,
      };
    }

    return { token: null, confidence: 0, contextOrder: -1, distribution: [] };
  }

  generate(model: MovementModelArtifact, seed: MovementToken[], maxSteps: number): MovementToken[] {
    const output = [...seed];
    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predict(model, output);
      if (prediction.token === null || prediction.token === END_OF_SEQUENCE) {
        break;
      }
      output.push(prediction.token);
    }
    return output;
  }
}

const CONTEXT_DELIMITER = "";

/** Sentinel appended during training to mark the end of a recorded movement. */
const END_OF_SEQUENCE = "__eos__";

type MarkovCounts = Record<number, Record<string, Record<MovementToken, number>>>;

type MarkovParameters = {
  maxOrder: number;
  smoothing: number;
  vocabulary: MovementToken[];
  counts: MarkovCounts;
};

function contextKey(context: MovementToken[]): string {
  return context.join(CONTEXT_DELIMITER);
}

function distributionFromCounts(
  table: Record<MovementToken, number>,
  smoothing: number,
  vocabularySize: number,
): Array<{ token: MovementToken; probability: number }> {
  const entries = Object.entries(table);
  const observedTotal = entries.reduce((sum, [, count]) => sum + count, 0);
  const denominator = observedTotal + smoothing * vocabularySize;
  if (denominator <= 0) {
    return [];
  }
  return entries
    .map(([token, count]) => ({ token, probability: (count + smoothing) / denominator }))
    .sort((a, b) => {
      if (b.probability !== a.probability) {
        return b.probability - a.probability;
      }
      // Deterministic lexicographic tie-break.
      return a.token < b.token ? -1 : a.token > b.token ? 1 : 0;
    });
}
