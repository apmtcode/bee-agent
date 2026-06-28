import type {
  MovementCandidate,
  MovementDataset,
  MovementModel,
  MovementModelBackend,
  MovementPrediction,
  MovementToken,
  MovementTrainingConfig,
} from "../movement-model.js";

const CONTEXT_SEPARATOR = "";

/**
 * Deterministic back-off n-gram learner. Counts how often each context of
 * length 0..order is followed by each token, then at inference time uses the
 * longest context it has evidence for and backs off toward the unigram
 * distribution otherwise. This is the cloud/CI-safe reference backend: it has
 * no native deps, trains in-process, and is fully reproducible — yet it still
 * generalizes, because an unseen full-length context falls back to shorter
 * contexts the model *has* seen.
 *
 * `data` layout: `levels[k]` maps a context key (the last k tokens joined) to a
 * `{ token: count }` table for the token that followed.
 */
type MarkovData = {
  levels: Record<string, Record<MovementToken, number>>[];
};

const DEFAULT_ORDER = 3;
const DEFAULT_SMOOTHING = 0.01;

export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov-ngram";

  async train(dataset: MovementDataset, config?: MovementTrainingConfig): Promise<MovementModel> {
    const order = clampOrder(config?.order ?? DEFAULT_ORDER);
    const smoothing = Math.max(0, config?.smoothing ?? DEFAULT_SMOOTHING);
    const levels: MarkovData["levels"] = Array.from({ length: order + 1 }, () => ({}));
    const vocabulary = new Set<MovementToken>();
    let tokenCount = 0;

    for (const sequence of dataset.sequences) {
      for (let index = 0; index < sequence.tokens.length; index += 1) {
        const next = sequence.tokens[index];
        if (next === undefined) {
          continue;
        }
        vocabulary.add(next);
        tokenCount += 1;
        for (let k = 0; k <= order; k += 1) {
          if (index - k < 0) {
            break;
          }
          const context = sequence.tokens.slice(index - k, index);
          const table = (levels[k] ??= {});
          const key = contextKey(context);
          const counts = (table[key] ??= {});
          counts[next] = (counts[next] ?? 0) + 1;
        }
      }
    }

    return {
      backendId: this.id,
      version: 1,
      order,
      vocabulary: [...vocabulary].sort(),
      data: { levels } satisfies MarkovData,
      metadata: {
        sequenceCount: dataset.sequences.length,
        tokenCount,
        smoothing,
      },
    };
  }

  predict(model: MovementModel, context: MovementToken[]): MovementPrediction {
    const data = model.data as MarkovData;
    const smoothing = model.metadata.smoothing;
    const vocabSize = model.vocabulary.length;

    for (let k = Math.min(model.order, context.length); k >= 0; k -= 1) {
      const table = data.levels[k];
      if (!table) {
        continue;
      }
      const counts = table[contextKey(context.slice(context.length - k))];
      if (!counts) {
        continue;
      }
      const ranked = rankCounts(counts, smoothing, vocabSize);
      if (ranked.length === 0) {
        continue;
      }
      const top = ranked[0];
      return {
        token: top?.token,
        probability: top?.probability ?? 0,
        ranked,
        backoffOrder: k,
      };
    }

    return { token: undefined, probability: 0, ranked: [], backoffOrder: 0 };
  }

  generate(model: MovementModel, seed: MovementToken[], steps: number): MovementToken[] {
    const generated: MovementToken[] = [];
    const context = [...seed];
    for (let step = 0; step < steps; step += 1) {
      const prediction = this.predict(model, context);
      if (prediction.token === undefined) {
        break;
      }
      generated.push(prediction.token);
      context.push(prediction.token);
    }
    return generated;
  }
}

function rankCounts(
  counts: Record<MovementToken, number>,
  smoothing: number,
  vocabSize: number,
): MovementCandidate[] {
  const entries = Object.entries(counts);
  const total = entries.reduce((sum, [, count]) => sum + count, 0);
  const denominator = total + smoothing * Math.max(vocabSize, 1);
  return entries
    .map<MovementCandidate>(([token, count]) => ({
      token,
      probability: denominator === 0 ? 0 : (count + smoothing) / denominator,
    }))
    .sort((a, b) => {
      if (b.probability !== a.probability) {
        return b.probability - a.probability;
      }
      return a.token < b.token ? -1 : a.token > b.token ? 1 : 0;
    });
}

function contextKey(context: MovementToken[]): string {
  return context.join(CONTEXT_SEPARATOR);
}

function clampOrder(order: number): number {
  if (!Number.isFinite(order)) {
    return DEFAULT_ORDER;
  }
  return Math.max(1, Math.min(8, Math.floor(order)));
}

/** A registry preloaded with the built-in deterministic backend. */
export function createDefaultMovementBackend(): MarkovMovementBackend {
  return new MarkovMovementBackend();
}
