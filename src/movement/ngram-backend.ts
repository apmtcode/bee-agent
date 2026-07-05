import {
  MOVEMENT_SEQUENCE_END,
  MOVEMENT_SEQUENCE_START,
  tokenizeSequence,
  type MovementDataset,
} from "./movement-event.js";
import {
  argmaxToken,
  sampleFromDistribution,
  type MovementGenerateParams,
  type MovementModelBackend,
  type MovementModelMetadata,
  type MovementTokenDistribution,
  type MovementTrainOptions,
  type TrainedMovementModel,
} from "./model-backend.js";

const DEFAULT_ORDER = 2;
const DEFAULT_SMOOTHING = 0.01;

type ContextCounts = Map<string, Map<string, number>>;

/**
 * Deterministic in-process n-gram (Markov) backend.
 *
 * This is the reference backend that lets the whole train → generate → evaluate
 * pipeline run in the cloud with zero native dependencies. It learns
 * next-token transition statistics at every context length from 0 (unigram) up
 * to `order`, then decodes with **stupid backoff**: at prediction time it uses
 * the longest suffix of the context that was actually observed, falling back to
 * shorter contexts (and finally the unigram prior) when the full context is
 * novel. That backoff is precisely what lets it generalize — a context it never
 * saw during training still yields a sensible distribution instead of nothing.
 */
export class NgramMovementBackend implements MovementModelBackend {
  readonly name = "ngram";

  async train(dataset: MovementDataset, options?: MovementTrainOptions): Promise<TrainedMovementModel> {
    const order = Math.max(1, Math.floor(options?.order ?? DEFAULT_ORDER));
    const smoothing = Math.max(0, options?.smoothing ?? DEFAULT_SMOOTHING);

    // countsByOrder[k] maps a k-token context to next-token counts.
    const countsByOrder: ContextCounts[] = Array.from({ length: order + 1 }, () => new Map());
    const vocabulary = new Set<string>();
    let tokenCount = 0;

    for (const sequence of dataset.sequences) {
      const tokens = [...tokenizeSequence(sequence), MOVEMENT_SEQUENCE_END];
      const padded = [...Array<string>(order).fill(MOVEMENT_SEQUENCE_START), ...tokens];
      for (let i = order; i < padded.length; i += 1) {
        const next = padded[i]!;
        vocabulary.add(next);
        tokenCount += 1;
        for (let k = 0; k <= order; k += 1) {
          const context = padded.slice(i - k, i);
          record(countsByOrder[k]!, contextKey(context), next);
        }
      }
    }

    const vocab = [...vocabulary].sort();
    const metadata: MovementModelMetadata = {
      backend: this.name,
      order,
      vocabularySize: vocab.length,
      sequenceCount: dataset.sequences.length,
      tokenCount,
    };

    return new NgramMovementModel(order, smoothing, countsByOrder, vocab, metadata);
  }
}

class NgramMovementModel implements TrainedMovementModel {
  constructor(
    private readonly order: number,
    private readonly smoothing: number,
    private readonly countsByOrder: ContextCounts[],
    readonly vocabulary: string[],
    readonly metadata: MovementModelMetadata,
  ) {}

  predictNext(context: string[]): MovementTokenDistribution {
    const counts = this.backoffCounts(context);
    return this.smooth(counts);
  }

  generate(params: MovementGenerateParams): string[] {
    const maxLength = Math.max(0, Math.floor(params.maxLength));
    // Condition on the START padding the model trained with so the first token
    // is the true start-of-sequence continuation, not the global unigram mode.
    const context = [...Array<string>(this.order).fill(MOVEMENT_SEQUENCE_START), ...(params.seed ?? [])];
    const output: string[] = [];
    for (let step = 0; step < maxLength; step += 1) {
      const distribution = this.predictNext(context);
      if (distribution.length === 0) {
        break;
      }
      const token = params.rng
        ? sampleFromDistribution(distribution, params.rng)
        : argmaxToken(distribution);
      if (token === undefined || token === MOVEMENT_SEQUENCE_END) {
        break;
      }
      output.push(token);
      context.push(token);
    }
    return output;
  }

  /** Longest-suffix match: the essence of stupid-backoff generalization. */
  private backoffCounts(context: string[]): Map<string, number> {
    for (let k = Math.min(this.order, context.length); k >= 0; k -= 1) {
      const key = contextKey(context.slice(context.length - k));
      const counts = this.countsByOrder[k]?.get(key);
      if (counts && counts.size > 0) {
        return counts;
      }
    }
    return this.countsByOrder[0]?.get(contextKey([])) ?? new Map();
  }

  private smooth(counts: Map<string, number>): MovementTokenDistribution {
    const total = [...counts.values()].reduce((sum, value) => sum + value, 0);
    const denominator = total + this.smoothing * this.vocabulary.length;
    if (denominator <= 0) {
      return [];
    }
    const distribution: MovementTokenDistribution = this.vocabulary.map((token) => ({
      token,
      probability: ((counts.get(token) ?? 0) + this.smoothing) / denominator,
    }));
    // With zero smoothing, drop unobserved tokens so argmax/sampling stays sharp.
    const filtered = this.smoothing > 0 ? distribution : distribution.filter((entry) => entry.probability > 0);
    return filtered.sort((a, b) => (b.probability - a.probability) || a.token.localeCompare(b.token));
  }
}

function record(counts: ContextCounts, key: string, token: string): void {
  let bucket = counts.get(key);
  if (!bucket) {
    bucket = new Map();
    counts.set(key, bucket);
  }
  bucket.set(token, (bucket.get(token) ?? 0) + 1);
}

function contextKey(context: string[]): string {
  return context.join("");
}
