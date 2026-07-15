import {
  MOVEMENT_END,
  MOVEMENT_START,
  type MovementCandidate,
  type MovementDataset,
  type MovementGenerateOptions,
  type MovementModel,
  type MovementModelBackend,
  type MovementPrediction,
  type MovementToken,
  type MovementTrainOptions,
  type SerializedMovementModel,
} from "./movement-model.js";

const SENTINELS = new Set<MovementToken>([MOVEMENT_START, MOVEMENT_END]);
const CONTEXT_DELIMITER = "";

export type MarkovBackendOptions = {
  /** Maximum context length (n-gram order). Defaults to 2. */
  order?: number;
};

type SerializedMarkovModel = SerializedMovementModel & {
  backend: "markov";
  order: number;
  vocabulary: MovementToken[];
  transitions: Array<{ context: MovementToken[]; nexts: Array<[MovementToken, number]> }>;
};

/**
 * Deterministic, in-process order-N Markov backend for movement sequences.
 *
 * - **Repeat recorded movements** (objective 2c): trained on a single trajectory
 *   it reproduces that trajectory exactly under greedy argmax generation.
 * - **Generalize to related movements** (objective 2d): because it stores
 *   variable-length contexts and *backs off* to shorter contexts (down to the
 *   unigram distribution) when a context is unseen, it can stitch transitions
 *   observed across different trajectories into novel-but-related sequences.
 *
 * Fully deterministic — argmax with lexicographic tie-breaking, no RNG — so
 * cloud/CI tests over synthetic streams are reproducible.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly name = "markov";

  private readonly order: number;

  constructor(options: MarkovBackendOptions = {}) {
    this.order = Math.max(1, Math.floor(options.order ?? 2));
  }

  train(dataset: MovementDataset, options: MovementTrainOptions = {}): MovementModel {
    const order = Math.max(1, Math.floor((options.order as number | undefined) ?? this.order));
    const transitions = new Map<string, Map<MovementToken, number>>();
    const vocabulary = new Set<MovementToken>();

    for (const sequence of dataset.sequences) {
      const tokens = [MOVEMENT_START, ...sequence.tokens, MOVEMENT_END];
      for (const token of sequence.tokens) {
        vocabulary.add(token);
      }
      for (let index = 1; index < tokens.length; index += 1) {
        const next = tokens[index]!;
        // Record every context length from 1..order ending just before `next`,
        // enabling longest-match prediction with graceful backoff.
        const maxContext = Math.min(order, index);
        for (let length = 0; length <= maxContext; length += 1) {
          const context = tokens.slice(index - length, index);
          const key = contextKey(context);
          let counts = transitions.get(key);
          if (!counts) {
            counts = new Map<MovementToken, number>();
            transitions.set(key, counts);
          }
          counts.set(next, (counts.get(next) ?? 0) + 1);
        }
      }
    }

    return new MarkovMovementModel(order, vocabulary, transitions);
  }

  load(serialized: SerializedMovementModel): MovementModel {
    const model = serialized as SerializedMarkovModel;
    if (model.backend !== "markov") {
      throw new Error(`MarkovMovementBackend cannot load backend "${serialized.backend}"`);
    }
    const transitions = new Map<string, Map<MovementToken, number>>();
    for (const entry of model.transitions) {
      transitions.set(contextKey(entry.context), new Map(entry.nexts));
    }
    return new MarkovMovementModel(model.order, new Set(model.vocabulary), transitions);
  }
}

class MarkovMovementModel implements MovementModel {
  readonly backend = "markov";

  constructor(
    private readonly order: number,
    private readonly vocab: Set<MovementToken>,
    private readonly transitions: Map<string, Map<MovementToken, number>>,
  ) {}

  get vocabulary(): MovementToken[] {
    return [...this.vocab].sort();
  }

  predictNext(context: MovementToken[]): MovementPrediction {
    // Longest-match backoff: try the longest context first, shorten on miss.
    for (let length = Math.min(this.order, context.length); length >= 0; length -= 1) {
      const counts = this.transitions.get(contextKey(context.slice(context.length - length)));
      if (counts && counts.size > 0) {
        return toPrediction(counts, length);
      }
    }
    return { token: undefined, probability: 0, distribution: [], matchedOrder: 0 };
  }

  generate(options: MovementGenerateOptions = {}): MovementToken[] {
    const seed = options.seed ?? [MOVEMENT_START];
    const maxLength = options.maxLength ?? 64;
    const includeSentinels = options.includeSentinels ?? false;

    const history = [...seed];
    const produced: MovementToken[] = [];
    for (let step = 0; step < maxLength; step += 1) {
      const prediction = this.predictNext(history);
      const next = prediction.token;
      if (next === undefined || next === MOVEMENT_END) {
        break;
      }
      history.push(next);
      produced.push(next);
    }

    const full = [...seed, ...produced];
    if (includeSentinels) {
      return full;
    }
    return full.filter((token) => !SENTINELS.has(token));
  }

  toJSON(): SerializedMarkovModel {
    const transitions = [...this.transitions.entries()]
      .map(([key, counts]) => ({
        context: key === "" ? [] : key.split(CONTEXT_DELIMITER),
        nexts: [...counts.entries()].sort(compareCounts),
      }))
      .sort((a, b) => contextKey(a.context).localeCompare(contextKey(b.context)));
    return {
      version: 1,
      backend: "markov",
      order: this.order,
      vocabulary: this.vocabulary,
      transitions,
    };
  }
}

function contextKey(context: MovementToken[]): string {
  return context.join(CONTEXT_DELIMITER);
}

/** Sort by descending count, then ascending token for deterministic tie-breaks. */
function compareCounts(a: [MovementToken, number], b: [MovementToken, number]): number {
  if (a[1] !== b[1]) {
    return b[1] - a[1];
  }
  return a[0].localeCompare(b[0]);
}

function toPrediction(counts: Map<MovementToken, number>, matchedOrder: number): MovementPrediction {
  const total = [...counts.values()].reduce((sum, count) => sum + count, 0);
  const distribution: MovementCandidate[] = [...counts.entries()]
    .sort(compareCounts)
    .map(([token, count]) => ({ token, probability: count / total }));
  const top = distribution[0];
  return {
    token: top?.token,
    probability: top?.probability ?? 0,
    distribution,
    matchedOrder,
  };
}
