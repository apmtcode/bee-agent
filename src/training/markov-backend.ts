// Deterministic in-process movement-model backend.
//
// A back-off n-gram (Markov) model over movement tokens. It learns transition
// frequencies from the recorded dataset, so it can (a) reproduce recorded
// movements exactly by following the highest-probability path, and (b)
// generalize to novel-but-related movements by recombining learned transitions
// through back-off to shorter contexts. It is fully deterministic (argmax with
// lexical tie-breaking), so cloud/CI tests are reproducible, and it needs no OS
// access — the real on-device backend can replace it by implementing
// `MovementModelBackend`.

import {
  MOVEMENT_END_TOKEN,
  MOVEMENT_START_TOKEN,
  type MovementDataset,
  type MovementGenerateParams,
  type MovementModelBackend,
  type MovementPrediction,
  type MovementToken,
  type TrainedMovementModel,
} from "./movement-model.js";

const BACKEND_KIND = "markov";
const DEFAULT_ORDER = 2;
const DEFAULT_ALPHA = 0.1;
const MAX_ALTERNATIVES = 4;

export type MarkovBackendOptions = {
  order?: number;
  alpha?: number;
};

type GramCounts = Map<string, Map<MovementToken, number>>;

type SerializedMarkovModel = {
  backend: typeof BACKEND_KIND;
  version: 1;
  order: number;
  alpha: number;
  vocabulary: MovementToken[];
  grams: Array<[string, Array<[MovementToken, number]>]>;
};

export class MarkovMovementBackend implements MovementModelBackend {
  readonly kind = BACKEND_KIND;
  private readonly order: number;
  private readonly alpha: number;

  constructor(options: MarkovBackendOptions = {}) {
    this.order = Math.max(1, Math.floor(options.order ?? DEFAULT_ORDER));
    this.alpha = options.alpha !== undefined && options.alpha > 0 ? options.alpha : DEFAULT_ALPHA;
  }

  train(dataset: MovementDataset): TrainedMovementModel {
    const grams: GramCounts = new Map();
    const vocab = new Set<MovementToken>(dataset.vocabulary);
    vocab.add(MOVEMENT_END_TOKEN);

    for (const sequence of dataset.sequences) {
      const tokens = [MOVEMENT_START_TOKEN, ...sequence.tokens, MOVEMENT_END_TOKEN];
      for (const token of tokens) {
        if (token !== MOVEMENT_START_TOKEN) {
          vocab.add(token);
        }
      }
      // For each target position, record counts at every back-off order.
      for (let i = 1; i < tokens.length; i += 1) {
        const target = tokens[i];
        for (let k = 0; k <= this.order; k += 1) {
          const start = i - k;
          if (start < 0) {
            break;
          }
          const context = tokens.slice(start, i);
          const key = gramKey(k, context);
          let counts = grams.get(key);
          if (!counts) {
            counts = new Map();
            grams.set(key, counts);
          }
          counts.set(target, (counts.get(target) ?? 0) + 1);
        }
      }
    }

    return new MarkovMovementModel(this.order, this.alpha, [...vocab].sort(), grams);
  }

  load(serialized: unknown): TrainedMovementModel {
    const data = serialized as SerializedMarkovModel;
    if (!data || data.backend !== BACKEND_KIND || !Array.isArray(data.grams)) {
      throw new Error("invalid serialized markov movement model");
    }
    const grams: GramCounts = new Map(
      data.grams.map(([key, entries]) => [key, new Map(entries)] as const),
    );
    return new MarkovMovementModel(data.order, data.alpha, [...data.vocabulary].sort(), grams);
  }
}

export class MarkovMovementModel implements TrainedMovementModel {
  readonly backend = BACKEND_KIND;

  constructor(
    private readonly order: number,
    private readonly alpha: number,
    private readonly vocabulary: MovementToken[],
    private readonly grams: GramCounts,
  ) {}

  predictNext(context: MovementToken[]): MovementPrediction | undefined {
    const counts = this.backoffCounts(context);
    if (!counts) {
      return undefined;
    }
    const total = sumCounts(counts) + this.alpha * this.vocabulary.length;
    const ranked = [...counts.entries()]
      .map(([token, count]) => ({ token, probability: (count + this.alpha) / total }))
      .sort((a, b) => (b.probability !== a.probability ? b.probability - a.probability : compareToken(a.token, b.token)));
    const best = ranked[0];
    if (!best) {
      return undefined;
    }
    return {
      token: best.token,
      probability: best.probability,
      alternatives: ranked.slice(1, 1 + MAX_ALTERNATIVES),
    };
  }

  generate(params: MovementGenerateParams = {}): MovementToken[] {
    const seed = params.seed ?? [MOVEMENT_START_TOKEN];
    const maxSteps = params.maxSteps ?? 64;
    const stopAtEnd = params.stopAtEnd ?? true;
    const context = [...seed];
    const out: MovementToken[] = [];
    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predictNext(context);
      if (!prediction) {
        break;
      }
      const token = prediction.token;
      if (token === MOVEMENT_END_TOKEN) {
        if (stopAtEnd) {
          break;
        }
      }
      out.push(token);
      context.push(token);
    }
    return out;
  }

  scoreSequence(tokens: MovementToken[]): number {
    const full = [MOVEMENT_START_TOKEN, ...tokens, MOVEMENT_END_TOKEN];
    let logProbSum = 0;
    let count = 0;
    for (let i = 1; i < full.length; i += 1) {
      const probability = this.probabilityOf(full.slice(0, i), full[i]);
      logProbSum += Math.log(probability);
      count += 1;
    }
    return count === 0 ? 0 : logProbSum / count;
  }

  toJSON(): SerializedMarkovModel {
    return {
      backend: BACKEND_KIND,
      version: 1,
      order: this.order,
      alpha: this.alpha,
      vocabulary: [...this.vocabulary],
      grams: [...this.grams.entries()].map(
        ([key, counts]) => [key, [...counts.entries()]] as [string, Array<[MovementToken, number]>],
      ),
    };
  }

  /** Smoothed probability of `token` given `context`, using best back-off. */
  probabilityOf(context: MovementToken[], token: MovementToken): number {
    const counts = this.backoffCounts(context);
    if (!counts) {
      return this.vocabulary.length > 0 ? 1 / this.vocabulary.length : 0;
    }
    const total = sumCounts(counts) + this.alpha * this.vocabulary.length;
    return ((counts.get(token) ?? 0) + this.alpha) / total;
  }

  /** Longest-suffix context (down to unigram) that has observed counts. */
  private backoffCounts(context: MovementToken[]): Map<MovementToken, number> | undefined {
    for (let k = Math.min(this.order, context.length); k >= 0; k -= 1) {
      const slice = context.slice(context.length - k);
      const counts = this.grams.get(gramKey(k, slice));
      if (counts && counts.size > 0) {
        return counts;
      }
    }
    return undefined;
  }
}

function gramKey(order: number, context: MovementToken[]): string {
  return `${order}|${context.join("|")}`;
}

function sumCounts(counts: Map<MovementToken, number>): number {
  let total = 0;
  for (const value of counts.values()) {
    total += value;
  }
  return total;
}

function compareToken(a: MovementToken, b: MovementToken): number {
  return a < b ? -1 : a > b ? 1 : 0;
}
