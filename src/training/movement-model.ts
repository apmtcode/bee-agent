import {
  movementTokenFromKey,
  movementTokenKey,
  type MovementDataset,
  type MovementToken,
} from "./movement-dataset.js";

/**
 * Pluggable local-model backend for the movement-learning subsystem.
 *
 * bee-agent runs in the cloud with no access to the user's machine, so the
 * training + inference backend must be an interface with a deterministic,
 * dependency-free implementation that validates the whole capture → dataset →
 * train → replay → generalize pipeline in CI. Real on-device backends (a small
 * local model trained via MLX/axolotl, see {@link LocalAppleSiliconTrainingRunner})
 * implement the same seam and are swapped in at runtime.
 *
 * The shipped {@link MarkovMovementBackend} is an order-N n-gram model with
 * Katz-style backoff. It is fully deterministic (argmax with a lexicographic
 * tie-break, no randomness), so its behaviour is reproducible in tests while
 * still genuinely *learning* transition statistics from recorded movements and
 * *generalizing* to related-but-unseen prefixes via backoff.
 */
export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, options?: MovementTrainingOptions): MovementModel;
  /** Highest-probability next movement given a context prefix. */
  predictNext(model: MovementModel, context: MovementToken[]): MovementPrediction | undefined;
  /** Full ranked next-movement distribution (for top-k eval / beam use). */
  rank(model: MovementModel, context: MovementToken[]): MovementPrediction[];
  /** Roll out a full movement sequence from an optional seed prefix. */
  generate(model: MovementModel, options?: MovementGenerateOptions): MovementToken[];
}

export type MovementTrainingOptions = {
  /** n-gram context order (1..5). Higher = more literal replay, less general. */
  order?: number;
};

export type MovementGenerateOptions = {
  seed?: MovementToken[];
  maxLength?: number;
  /** Stop early if the same token repeats this many times consecutively. */
  stopOnRepeat?: number;
};

export type MovementPrediction = {
  token: MovementToken;
  key: string;
  probability: number;
  /** Context length actually used (0 = start/unigram fallback). */
  contextOrder: number;
  /** True when a shorter context than available had to be used (backed off). */
  backoff: boolean;
};

export type MovementCountMap = Record<string, number>;

export type MovementModel = {
  version: 1;
  backend: string;
  order: number;
  vocabulary: string[];
  /** key -> canonical token, so predictions reconstruct full movements. */
  tokens: Record<string, MovementToken>;
  /** contextKey -> { nextKey: count } for every context length 1..order. */
  transitions: Record<string, MovementCountMap>;
  /** first-token counts (start distribution). */
  starts: MovementCountMap;
  /** overall token counts (final backoff distribution). */
  unigram: MovementCountMap;
  trainedSequences: number;
  trainedTokens: number;
};

const CONTEXT_SEP = "";
// CONTEXT_SEP (above) is U+0001, a control char that never appears in a token
// key (keys are [a-z0-9._/|-]), so joined multi-token contexts never collide
// across token boundaries.
const DEFAULT_ORDER = 2;
const MIN_ORDER = 1;
const MAX_ORDER = 5;

function increment(map: MovementCountMap, key: string): void {
  map[key] = (map[key] ?? 0) + 1;
}

function clampOrder(order: number | undefined): number {
  if (typeof order !== "number" || Number.isNaN(order)) {
    return DEFAULT_ORDER;
  }
  return Math.max(MIN_ORDER, Math.min(MAX_ORDER, Math.trunc(order)));
}

export class MarkovMovementBackend implements MovementModelBackend {
  readonly name = "markov";

  constructor(private readonly defaultOrder: number = DEFAULT_ORDER) {}

  train(dataset: MovementDataset, options?: MovementTrainingOptions): MovementModel {
    const order = clampOrder(options?.order ?? this.defaultOrder);
    const transitions: Record<string, MovementCountMap> = {};
    const starts: MovementCountMap = {};
    const unigram: MovementCountMap = {};
    const tokens: Record<string, MovementToken> = {};
    let trainedSequences = 0;
    let trainedTokens = 0;

    for (const sequence of dataset.sequences) {
      if (sequence.tokens.length === 0) {
        continue;
      }
      trainedSequences += 1;
      const keys = sequence.tokens.map((token) => {
        const key = movementTokenKey(token);
        tokens[key] = token;
        return key;
      });
      increment(starts, keys[0]!);
      for (let i = 0; i < keys.length; i += 1) {
        trainedTokens += 1;
        increment(unigram, keys[i]!);
        const next = keys[i + 1];
        if (next === undefined) {
          continue;
        }
        for (let length = 1; length <= order; length += 1) {
          const start = i - length + 1;
          if (start < 0) {
            break;
          }
          const contextKey = keys.slice(start, i + 1).join(CONTEXT_SEP);
          const bucket = (transitions[contextKey] ??= {});
          increment(bucket, next);
        }
      }
    }

    return {
      version: 1,
      backend: this.name,
      order,
      vocabulary: Object.keys(unigram).sort(),
      tokens,
      transitions,
      starts,
      unigram,
      trainedSequences,
      trainedTokens,
    };
  }

  rank(model: MovementModel, context: MovementToken[]): MovementPrediction[] {
    const keys = context.map(movementTokenKey);
    const maxLength = Math.min(model.order, keys.length);
    for (let length = maxLength; length >= 1; length -= 1) {
      const contextKey = keys.slice(keys.length - length).join(CONTEXT_SEP);
      const bucket = model.transitions[contextKey];
      if (bucket && Object.keys(bucket).length > 0) {
        return this.toPredictions(model, bucket, length, length < maxLength);
      }
    }
    const base = keys.length === 0 ? model.starts : model.unigram;
    return this.toPredictions(model, base, 0, keys.length > 0);
  }

  predictNext(model: MovementModel, context: MovementToken[]): MovementPrediction | undefined {
    return this.rank(model, context)[0];
  }

  generate(model: MovementModel, options?: MovementGenerateOptions): MovementToken[] {
    const maxLength = Math.max(1, options?.maxLength ?? 32);
    const stopOnRepeat = Math.max(1, options?.stopOnRepeat ?? 3);
    const out: MovementToken[] = options?.seed ? [...options.seed] : [];

    if (out.length === 0) {
      const start = this.rank(model, [])[0];
      if (!start) {
        return out;
      }
      out.push(start.token);
    }

    let repeat = 1;
    while (out.length < maxLength) {
      const next = this.predictNext(model, out);
      if (!next) {
        break;
      }
      const previousKey = movementTokenKey(out[out.length - 1]!);
      if (next.key === previousKey) {
        repeat += 1;
        if (repeat >= stopOnRepeat) {
          break;
        }
      } else {
        repeat = 1;
      }
      out.push(next.token);
    }
    return out;
  }

  private toPredictions(
    model: MovementModel,
    counts: MovementCountMap,
    contextOrder: number,
    backoff: boolean,
  ): MovementPrediction[] {
    const entries = Object.entries(counts);
    const total = entries.reduce((sum, [, count]) => sum + count, 0);
    if (total === 0) {
      return [];
    }
    return entries
      .map(([key, count]) => ({
        token: model.tokens[key] ?? movementTokenFromKey(key),
        key,
        probability: count / total,
        contextOrder,
        backoff,
      }))
      .sort((a, b) => (b.probability !== a.probability ? b.probability - a.probability : a.key.localeCompare(b.key)));
  }
}

type MovementBackendFactory = () => MovementModelBackend;

const backendRegistry = new Map<string, MovementBackendFactory>();

/** Register a pluggable movement backend (real on-device backends use this seam). */
export function registerMovementBackend(name: string, factory: MovementBackendFactory): void {
  backendRegistry.set(name, factory);
}

export function listMovementBackends(): string[] {
  return [...backendRegistry.keys()].sort();
}

export function createMovementBackend(name = "markov"): MovementModelBackend {
  const factory = backendRegistry.get(name);
  if (!factory) {
    throw new Error(`unknown movement backend: ${name} (available: ${listMovementBackends().join(", ")})`);
  }
  return factory();
}

registerMovementBackend("markov", () => new MarkovMovementBackend());
