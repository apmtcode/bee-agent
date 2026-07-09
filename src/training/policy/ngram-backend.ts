import {
  MOVEMENT_END_TOKEN,
  type MovementCandidate,
  type MovementPolicyBackend,
  type MovementPolicyMetadata,
  type MovementPrediction,
  type MovementPredictionContext,
  type MovementRolloutSeed,
  type MovementTrainingOptions,
  type TrainedMovementPolicy,
} from "./backend.js";
import type { MovementActionToken, MovementDataset } from "./movement-dataset.js";

const CONTEXT_SEPARATOR = "";
const DEFAULT_ORDER = 3;
const DEFAULT_MAX_ROLLOUT_STEPS = 64;

type CountMap = Map<MovementActionToken, number>;

/**
 * Deterministic n-gram movement policy with stupid-backoff. It learns, for every
 * context length `k` in `1..order`, the distribution of the next action token
 * given the last `k` tokens, plus a start model (entry observation → first
 * action) and a global unigram level. Prediction tries the longest matching
 * action context first and backs off to shorter contexts; from an empty prefix it
 * uses the entry observation; and it falls back to the unigram distribution — so
 * exact recorded sequences are reproduced verbatim while related-but-unseen
 * prefixes still yield a plausible next move (generalization).
 *
 * It is intentionally dependency-free and fully deterministic (ties broken by
 * count then lexical order) so it trains and infers identically in cloud/CI and
 * serves as the reference backend behind {@link MovementPolicyBackend}.
 */
export class NgramMovementBackend implements MovementPolicyBackend {
  readonly id = "ngram-backoff";

  train(dataset: MovementDataset, options: MovementTrainingOptions = {}): TrainedMovementPolicy {
    const order = Math.max(1, Math.floor(options.order ?? DEFAULT_ORDER));
    const maxRolloutSteps = Math.max(1, Math.floor(options.maxRolloutSteps ?? DEFAULT_MAX_ROLLOUT_STEPS));

    // contextModels[k] maps a k-token context key -> next-token counts.
    const contextModels: Array<Map<string, CountMap>> = Array.from({ length: order + 1 }, () => new Map());
    // startModel maps an entry-observation key -> distribution of first actions.
    const startModel = new Map<string, CountMap>();
    const unigram: CountMap = new Map();

    for (const sequence of dataset.sequences) {
      const firstStep = sequence.steps[0];
      if (firstStep?.observation) {
        bump(mapFor(startModel, normalizeContext(firstStep.observation)), firstStep.token);
      }

      const tokens = [...sequence.steps.map((step) => step.token), MOVEMENT_END_TOKEN];
      for (let index = 0; index < tokens.length; index += 1) {
        const next = tokens[index]!;
        bump(unigram, next);
        for (let k = 1; k <= order; k += 1) {
          if (index - k < 0) {
            break;
          }
          const contextTokens = tokens.slice(index - k, index);
          bump(mapFor(contextModels[k]!, contextTokens.join(CONTEXT_SEPARATOR)), next);
        }
      }
    }

    const metadata: MovementPolicyMetadata = {
      backendId: this.id,
      order,
      vocabularySize: dataset.vocabulary.length,
      sequenceCount: dataset.sequences.length,
      stepCount: dataset.stepCount,
    };

    return new NgramMovementPolicy({
      backendId: this.id,
      metadata,
      order,
      maxRolloutSteps,
      contextModels,
      startModel,
      unigram,
    });
  }
}

type NgramPolicyState = {
  backendId: string;
  metadata: MovementPolicyMetadata;
  order: number;
  maxRolloutSteps: number;
  contextModels: Array<Map<string, CountMap>>;
  startModel: Map<string, CountMap>;
  unigram: CountMap;
};

class NgramMovementPolicy implements TrainedMovementPolicy {
  readonly backendId: string;
  readonly metadata: MovementPolicyMetadata;
  private readonly state: NgramPolicyState;

  constructor(state: NgramPolicyState) {
    this.state = state;
    this.backendId = state.backendId;
    this.metadata = state.metadata;
  }

  predict(context: MovementPredictionContext): MovementPrediction {
    const { order, contextModels, startModel, unigram } = this.state;
    const prefix = context.prefix;

    const maxContext = Math.min(order, prefix.length);
    for (let k = maxContext; k >= 1; k -= 1) {
      const key = prefix.slice(prefix.length - k).join(CONTEXT_SEPARATOR);
      const counts = contextModels[k]?.get(key);
      if (counts && counts.size > 0) {
        return toPrediction(counts, k, "context");
      }
    }

    // From an empty prefix, the entry observation predicts the first action.
    if (prefix.length === 0 && context.observation) {
      const counts = startModel.get(normalizeContext(context.observation));
      if (counts && counts.size > 0) {
        return toPrediction(counts, 0, "observation");
      }
    }

    if (unigram.size > 0) {
      return toPrediction(unigram, -1, "unigram");
    }

    return { token: undefined, confidence: 0, candidates: [], backoffOrder: -1, source: "empty" };
  }

  rollout(seed: MovementRolloutSeed = {}): MovementActionToken[] {
    const maxSteps = Math.max(0, Math.floor(seed.maxSteps ?? this.state.maxRolloutSteps));
    const history: MovementActionToken[] = [...(seed.prefix ?? [])];
    const generated: MovementActionToken[] = [];

    while (generated.length < maxSteps) {
      const prediction = this.predict({ prefix: history, observation: seed.observation });
      const token = prediction.token;
      if (token === undefined || token === MOVEMENT_END_TOKEN) {
        break;
      }
      generated.push(token);
      history.push(token);
    }

    return generated;
  }
}

function toPrediction(counts: CountMap, backoffOrder: number, source: MovementPrediction["source"]): MovementPrediction {
  const total = [...counts.values()].reduce((sum, value) => sum + value, 0);
  const candidates: MovementCandidate[] = [...counts.entries()]
    .map(([token, count]) => ({ token, count, probability: count / total }))
    .sort((a, b) => (b.count !== a.count ? b.count - a.count : a.token < b.token ? -1 : a.token > b.token ? 1 : 0));
  const top = candidates[0];
  return {
    token: top?.token,
    confidence: top?.probability ?? 0,
    candidates,
    backoffOrder,
    source,
  };
}

function mapFor(store: Map<string, CountMap>, key: string): CountMap {
  let counts = store.get(key);
  if (!counts) {
    counts = new Map();
    store.set(key, counts);
  }
  return counts;
}

function bump(counts: CountMap, token: MovementActionToken): void {
  counts.set(token, (counts.get(token) ?? 0) + 1);
}

function normalizeContext(observation: string): string {
  return observation.trim().toLowerCase().replace(/\s+/g, " ");
}
