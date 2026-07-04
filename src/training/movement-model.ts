/**
 * Pluggable local-movement model backend.
 *
 * This is the in-process, cloud-testable half of the movement-learning
 * subsystem (standing objective #2d): given a dataset of recorded movement
 * sequences it can (a) train a small local model, (b) predict the next
 * movement given a context, and (c) generate a continuation that repeats the
 * recorded movements and generalizes — via back-off — to related but unseen
 * contexts.
 *
 * The {@link MovementModelBackend} interface is the seam a real on-device
 * backend (mlx / torch / llama.cpp) implements. The shipped
 * {@link MarkovMovementBackend} is a deterministic, dependency-free n-gram
 * model so the whole train → infer → generalize loop can be validated in the
 * cloud with synthetic event streams (no OS access, no Python toolchain).
 */

/** A single, tokenized movement (one gesture / action). */
export type MovementToken = string;

/** One recorded movement trajectory as an ordered token stream. */
export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

/** A replayable dataset of tokenized movement sequences. */
export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

/** Sentinel tokens marking sequence boundaries; never emitted as movements. */
export const MOVEMENT_START_TOKEN = "START" as const;
export const MOVEMENT_END_TOKEN = "END" as const;

export type MovementPrediction = {
  token: MovementToken;
  probability: number;
  /** Context length (in tokens) the prediction was drawn from after back-off. */
  order: number;
};

export type TrainMovementModelOptions = {
  /** Maximum context length the model conditions on. Defaults to 2 (trigram). */
  order?: number;
};

export type GenerateMovementOptions = {
  /** Hard cap on generated tokens. Defaults to 64. */
  maxSteps?: number;
  /** Stop before emitting this token (in addition to the end sentinel). */
  stopToken?: MovementToken;
};

export type SerializedMovementModel = {
  version: 1;
  backend: string;
  order: number;
  /** context-key -> (next-token -> count). Context key is tokens joined by . */
  transitions: Record<string, Record<string, number>>;
};

export interface MovementModel {
  readonly backend: string;
  readonly order: number;
  /** Ranked next-token distribution for a context (best first, deterministic). */
  predict(context: MovementToken[]): MovementPrediction[];
  /** Greedy next movement, or undefined if the model would end the sequence. */
  predictNext(context: MovementToken[]): MovementToken | undefined;
  /** Deterministic continuation of a seed (excludes the seed and sentinels). */
  generate(seed: MovementToken[], options?: GenerateMovementOptions): MovementToken[];
  serialize(): SerializedMovementModel;
}

export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, options?: TrainMovementModelOptions): MovementModel;
}

const CONTEXT_SEPARATOR = "";
const DEFAULT_ORDER = 2;
const DEFAULT_MAX_STEPS = 64;

function contextKey(tokens: MovementToken[]): string {
  return tokens.join(CONTEXT_SEPARATOR);
}

/**
 * Deterministic n-gram movement model with stupid-backoff generalization.
 *
 * Training counts, for every position, the continuation of each context suffix
 * from length `order` down to the empty context. At inference time it uses the
 * longest context that has observed continuations, falling back to shorter
 * contexts (and finally the unigram distribution) when the exact context was
 * never seen — this is what lets it perform new-but-related movements.
 */
export class MarkovMovementModel implements MovementModel {
  readonly backend = "markov";

  constructor(
    readonly order: number,
    private readonly transitions: Map<string, Map<MovementToken, number>>,
  ) {}

  predict(context: MovementToken[]): MovementPrediction[] {
    const recent = context.slice(-this.order);
    const padded = [
      ...Array.from({ length: Math.max(0, this.order - recent.length) }, () => MOVEMENT_START_TOKEN),
      ...recent,
    ];
    for (let length = padded.length; length >= 0; length -= 1) {
      const suffix = padded.slice(padded.length - length);
      const counts = this.transitions.get(contextKey(suffix));
      if (!counts || counts.size === 0) {
        continue;
      }
      let total = 0;
      for (const count of counts.values()) {
        total += count;
      }
      if (total === 0) {
        continue;
      }
      const predictions = [...counts.entries()]
        .filter(([token]) => token !== MOVEMENT_START_TOKEN)
        .map(([token, count]) => ({ token, probability: count / total, order: length }));
      if (predictions.length === 0) {
        continue;
      }
      return predictions.sort((a, b) =>
        a.probability !== b.probability ? b.probability - a.probability : a.token < b.token ? -1 : 1,
      );
    }
    return [];
  }

  predictNext(context: MovementToken[]): MovementToken | undefined {
    const top = this.predict(context)[0];
    if (!top || top.token === MOVEMENT_END_TOKEN) {
      return undefined;
    }
    return top.token;
  }

  generate(seed: MovementToken[], options: GenerateMovementOptions = {}): MovementToken[] {
    const maxSteps = options.maxSteps ?? DEFAULT_MAX_STEPS;
    const context = [...seed];
    const generated: MovementToken[] = [];
    for (let step = 0; step < maxSteps; step += 1) {
      const next = this.predictNext(context);
      if (next === undefined || next === options.stopToken) {
        break;
      }
      generated.push(next);
      context.push(next);
    }
    return generated;
  }

  serialize(): SerializedMovementModel {
    const transitions: Record<string, Record<string, number>> = {};
    for (const [key, counts] of this.transitions) {
      transitions[key] = Object.fromEntries(counts);
    }
    return { version: 1, backend: this.backend, order: this.order, transitions };
  }
}

export class MarkovMovementBackend implements MovementModelBackend {
  readonly name = "markov";

  train(dataset: MovementDataset, options: TrainMovementModelOptions = {}): MarkovMovementModel {
    const order = Math.max(0, Math.trunc(options.order ?? DEFAULT_ORDER));
    const transitions = new Map<string, Map<MovementToken, number>>();

    const record = (context: MovementToken[], next: MovementToken): void => {
      const key = contextKey(context);
      let counts = transitions.get(key);
      if (!counts) {
        counts = new Map<MovementToken, number>();
        transitions.set(key, counts);
      }
      counts.set(next, (counts.get(next) ?? 0) + 1);
    };

    for (const sequence of dataset.sequences) {
      const padded = [
        ...Array.from({ length: order }, () => MOVEMENT_START_TOKEN),
        ...sequence.tokens,
        MOVEMENT_END_TOKEN,
      ];
      for (let index = order; index < padded.length; index += 1) {
        const next = padded[index]!;
        const history = padded.slice(0, index);
        for (let length = 0; length <= order; length += 1) {
          record(history.slice(history.length - length), next);
        }
      }
    }

    return new MarkovMovementModel(order, transitions);
  }

  static deserialize(serialized: SerializedMovementModel): MarkovMovementModel {
    const transitions = new Map<string, Map<MovementToken, number>>();
    for (const [key, counts] of Object.entries(serialized.transitions)) {
      transitions.set(key, new Map(Object.entries(counts)));
    }
    return new MarkovMovementModel(serialized.order, transitions);
  }
}
