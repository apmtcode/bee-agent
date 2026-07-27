/**
 * Local-movement learning subsystem — model layer.
 *
 * This module defines the *pluggable* local-model backend for the movement
 * subsystem (standing objective #2c/#2d): given a dataset of recorded
 * movements it post-trains a model that can (a) repeat the recorded movements
 * and (b) generalize to new-but-related movements.
 *
 * The default backend is a fully deterministic, in-process order-k Markov
 * model with Katz-style backoff. It requires no native code, no GPU and no
 * network, so it trains and infers in the cloud/CI — while the
 * {@link MovementModelBackend} seam lets a real on-device small model be
 * dropped in later without changing any call site.
 */

/** A single normalized movement extracted from capture (mouse/keyboard/UI). */
export type MovementEvent = {
  ts: number;
  /** "observation" (UI/OS state) or "action" (an input the operator made). */
  kind: "observation" | "action";
  /** Where it came from — an observation `source` or an action `tool`. */
  channel: string;
  /** Normalized leading verb of the summary (e.g. "clicked", "typed"). */
  verb: string;
  /** Optional concrete target/value, retained for replay (not used as token). */
  target?: string;
  /** Canonical vocabulary token: `${kind}:${channel}:${verb}`. */
  token: string;
};

/** One replayable movement sequence (typically one reviewed trajectory). */
export type MovementSequence = {
  id: string;
  sourceTrajectoryIds: string[];
  events: MovementEvent[];
  tokens: string[];
};

/** A structured, replayable dataset the backend trains on. */
export type MovementDataset = {
  version: 1;
  createdAt: string;
  /** Sorted unique tokens (plus the START/END sentinels). */
  vocabulary: string[];
  sequences: MovementSequence[];
  tokenCount: number;
};

/** Sentinels so a model can be seeded from nothing and know when to stop. */
export const MOVEMENT_START_TOKEN = "start";
export const MOVEMENT_END_TOKEN = "end";

export type MovementTrainingOptions = {
  /** Max context length of the Markov model (default 2). */
  order?: number;
};

export type MovementPredictionCandidate = {
  token: string;
  probability: number;
};

export type MovementPrediction = {
  token: string;
  probability: number;
  /** The context order actually used after backoff (k..0). */
  backoffOrder: number;
  alternatives: MovementPredictionCandidate[];
};

export type MovementGenerateOptions = {
  maxSteps?: number;
  /** Stop when this token is produced (default {@link MOVEMENT_END_TOKEN}). */
  stopToken?: string;
};

/** Serialized model — portable across processes, backend-agnostic envelope. */
export type SerializedMovementModel = {
  backendId: string;
  order: number;
  vocabulary: string[];
  /** grams[order] maps a joined context to token→count. */
  grams: Record<string, Record<string, number>>[];
};

/** A trained model that can repeat and generalize recorded movements. */
export interface TrainedMovementModel {
  readonly backendId: string;
  readonly order: number;
  readonly vocabulary: string[];
  /** Predict the next token given a context, or undefined if it can't. */
  predictNext(context: string[]): MovementPrediction | undefined;
  /** Roll out a full movement sequence from an optional seed context. */
  generate(seed?: string[], options?: MovementGenerateOptions): string[];
  toJSON(): SerializedMovementModel;
}

/** The pluggable training backend seam. */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainingOptions): Promise<TrainedMovementModel>;
}

const CONTEXT_SEPARATOR = " ";

function contextKey(context: string[]): string {
  return context.join(CONTEXT_SEPARATOR);
}

/**
 * Deterministic order-k Markov model with backoff.
 *
 * Backoff is what delivers generalization objective #2d: when the full-order
 * context has never been seen, the model falls back to shorter suffixes
 * (k-1, …, 1, 0), so a *new* movement sequence assembled from familiar
 * sub-steps still gets a sensible next-move prediction.
 */
export class MarkovMovementModel implements TrainedMovementModel {
  constructor(
    readonly backendId: string,
    readonly order: number,
    readonly vocabulary: string[],
    private readonly grams: Map<string, Map<string, number>>[],
  ) {}

  predictNext(context: string[]): MovementPrediction | undefined {
    for (let k = Math.min(this.order, context.length); k >= 0; k -= 1) {
      const suffix = k === 0 ? [] : context.slice(context.length - k);
      const table = this.grams[k]?.get(contextKey(suffix));
      if (!table || table.size === 0) {
        continue;
      }
      const total = [...table.values()].reduce((sum, count) => sum + count, 0);
      const ranked = [...table.entries()]
        .map(([token, count]) => ({ token, probability: count / total }))
        // Deterministic tie-break: higher probability first, then token order.
        .sort((a, b) => (b.probability - a.probability) || (a.token < b.token ? -1 : a.token > b.token ? 1 : 0));
      const best = ranked[0];
      if (!best) {
        continue;
      }
      return {
        token: best.token,
        probability: best.probability,
        backoffOrder: k,
        alternatives: ranked.slice(1, 5),
      };
    }
    return undefined;
  }

  generate(seed: string[] = [], options: MovementGenerateOptions = {}): string[] {
    const maxSteps = options.maxSteps ?? 64;
    const stopToken = options.stopToken ?? MOVEMENT_END_TOKEN;
    const context = [MOVEMENT_START_TOKEN, ...seed];
    const output: string[] = [...seed];
    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predictNext(context);
      if (!prediction || prediction.token === stopToken) {
        break;
      }
      output.push(prediction.token);
      context.push(prediction.token);
    }
    return output;
  }

  toJSON(): SerializedMovementModel {
    return {
      backendId: this.backendId,
      order: this.order,
      vocabulary: [...this.vocabulary],
      grams: this.grams.map((table) => {
        const record: Record<string, Record<string, number>> = {};
        for (const [key, counts] of table) {
          record[key] = Object.fromEntries(counts);
        }
        return record;
      }),
    };
  }

  static fromJSON(serialized: SerializedMovementModel): MarkovMovementModel {
    const grams = serialized.grams.map((record) => {
      const table = new Map<string, Map<string, number>>();
      for (const [key, counts] of Object.entries(record)) {
        table.set(key, new Map(Object.entries(counts)));
      }
      return table;
    });
    return new MarkovMovementModel(serialized.backendId, serialized.order, serialized.vocabulary, grams);
  }
}

/** The default deterministic mock backend (no native deps). */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov-mock";

  // eslint-disable-next-line @typescript-eslint/require-await -- async to satisfy the pluggable seam
  async train(dataset: MovementDataset, options: MovementTrainingOptions = {}): Promise<TrainedMovementModel> {
    const order = Math.max(0, options.order ?? 2);
    const grams: Map<string, Map<string, number>>[] = Array.from({ length: order + 1 }, () => new Map());

    for (const sequence of dataset.sequences) {
      const tokens = [MOVEMENT_START_TOKEN, ...sequence.tokens, MOVEMENT_END_TOKEN];
      for (let index = 1; index < tokens.length; index += 1) {
        const next = tokens[index]!;
        for (let k = 0; k <= order; k += 1) {
          if (index - k < 0) {
            break;
          }
          const suffix = tokens.slice(index - k, index);
          const key = contextKey(suffix);
          const table = grams[k]!;
          const counts = table.get(key) ?? new Map<string, number>();
          counts.set(next, (counts.get(next) ?? 0) + 1);
          table.set(key, counts);
        }
      }
    }

    return new MarkovMovementModel(this.id, order, [...dataset.vocabulary], grams);
  }
}

/** Registry making the model backend pluggable at runtime. */
export class MovementBackendRegistry {
  private readonly backends = new Map<string, MovementModelBackend>();

  register(backend: MovementModelBackend): this {
    this.backends.set(backend.id, backend);
    return this;
  }

  resolve(id: string): MovementModelBackend {
    const backend = this.backends.get(id);
    if (!backend) {
      throw new Error(`unknown movement model backend: ${id} (have: ${[...this.backends.keys()].join(", ") || "none"})`);
    }
    return backend;
  }

  list(): string[] {
    return [...this.backends.keys()].sort();
  }
}

/** Registry seeded with the deterministic mock — safe in cloud/CI. */
export function createDefaultMovementBackendRegistry(): MovementBackendRegistry {
  return new MovementBackendRegistry().register(new MarkovMovementBackend());
}
