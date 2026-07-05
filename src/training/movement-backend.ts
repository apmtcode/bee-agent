import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";

/**
 * Local-movement learning backend.
 *
 * The engine's standing objective #2 is a subsystem that can (a) record
 * movements, (b) persist them to a replayable dataset, (c) post-train a *local*
 * model to repeat the recorded movements, and (d) generalize to new-but-related
 * movements. Pieces (a)/(b) live in `src/capture`; the real on-device trainer
 * (MLX / Axolotl) is described by {@link LocalAppleSiliconTrainingRunner}, which
 * only emits a launch plan — it cannot run in the cloud.
 *
 * This module supplies the missing seam: a *pluggable* backend interface plus a
 * deterministic, dependency-free reference backend (an n-gram movement model)
 * that trains and infers fully in-process. It lets the capture -> dataset ->
 * train -> infer round-trip be validated in CI with synthetic event streams,
 * while leaving a documented seam ({@link LocalMovementBackend}) for a real
 * small on-device model to drop in later.
 */

/** A single tokenized movement step — the atomic unit the model learns over. */
export type MovementToken = string;

/** An ordered movement sequence, typically derived from one trajectory replay. */
export type MovementSample = {
  trajectoryId?: string;
  sessionId?: string;
  tokens: MovementToken[];
};

/** A replayable, model-ready dataset assembled from recorded movements. */
export type MovementDataset = {
  version: 1;
  samples: MovementSample[];
  /** Sorted, de-duplicated set of every token that appears in `samples`. */
  vocabulary: MovementToken[];
};

export type MovementTrainOptions = {
  /** Maximum context length (n-gram order). Defaults to 2. */
  order?: number;
};

/** Result of asking a trained model for the next movement token. */
export type MovementPrediction = {
  token: MovementToken;
  /** Conditional probability under the matched context. */
  probability: number;
  /** Context length actually used after back-off (0 = unigram prior). */
  order: number;
};

/**
 * Pluggable backend contract. A real implementation would shell out to an
 * on-device trainer and load the resulting weights; the reference
 * {@link NGramMovementBackend} implements it in pure TypeScript so tests pass in
 * the cloud with no native deps.
 */
export interface LocalMovementBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainOptions): Promise<TrainedMovementModel>;
  /** Rehydrate a previously trained model from its serialized form. */
  load(serialized: SerializedMovementModel): TrainedMovementModel;
}

export type SerializedMovementModel = {
  version: 1;
  backendId: string;
  order: number;
  vocabulary: MovementToken[];
  /** `[contextKey, [[token, count], ...]]` transition table. */
  transitions: Array<[string, Array<[MovementToken, number]>]>;
};

const START_TOKEN = "<s>";
const END_TOKEN = "</s>";
const CONTEXT_SEPARATOR = "";

function isSentinel(token: MovementToken): boolean {
  return token === START_TOKEN || token === END_TOKEN;
}

/** Deterministically map a replay timeline event to a movement token. */
export function tokenizeReplayEvent(event: ReplayTimelineEvent): MovementToken {
  switch (event.kind) {
    case "transcript":
      return `transcript:${event.role}`;
    case "observation":
      return `observation:${event.source}`;
    case "action":
      return `action:${event.tool}`;
  }
}

/** Turn a replay manifest into one ordered movement sample (events are pre-sorted). */
export function tokenizeReplayManifest(replay: ReplayManifest): MovementSample {
  return {
    sessionId: replay.sessionId,
    trajectoryId: replay.trajectoryIds[0],
    tokens: replay.events.map(tokenizeReplayEvent),
  };
}

/** Assemble a model-ready dataset from recorded replay manifests. */
export function buildMovementDataset(replays: ReplayManifest[]): MovementDataset {
  const samples = replays.map(tokenizeReplayManifest).filter((sample) => sample.tokens.length > 0);
  const vocabulary = [...new Set(samples.flatMap((sample) => sample.tokens))].sort();
  return { version: 1, samples, vocabulary };
}

/**
 * A trained, serializable n-gram movement model. It memorizes recorded
 * movements exactly and generalizes by composing overlapping sub-sequences from
 * different trajectories via stupid-backoff decoding.
 */
export class TrainedMovementModel {
  constructor(
    readonly backendId: string,
    readonly order: number,
    readonly vocabulary: MovementToken[],
    private readonly transitions: Map<string, Map<MovementToken, number>>,
  ) {}

  /** Most likely next token given a context, backing off to shorter contexts. */
  predictNext(context: MovementToken[]): MovementPrediction | undefined {
    const maxOrder = Math.min(this.order, context.length);
    for (let used = maxOrder; used >= 0; used -= 1) {
      const key = context.slice(context.length - used).join(CONTEXT_SEPARATOR);
      const distribution = this.transitions.get(key);
      if (!distribution || distribution.size === 0) {
        continue;
      }
      const best = argmax(distribution);
      if (best === undefined) {
        continue;
      }
      const total = [...distribution.values()].reduce((sum, count) => sum + count, 0);
      return { token: best.token, probability: best.count / total, order: used };
    }
    return undefined;
  }

  /**
   * Greedily generate a continuation for `prompt`, stopping at the learned end
   * marker or after `steps` tokens. Sentinel markers are stripped from output.
   */
  generate(prompt: MovementToken[], steps: number): MovementToken[] {
    const padding = Array.from({ length: this.order }, () => START_TOKEN);
    const history = [...padding, ...prompt];
    const generated: MovementToken[] = [];
    for (let step = 0; step < steps; step += 1) {
      const prediction = this.predictNext(history);
      if (!prediction || prediction.token === END_TOKEN) {
        break;
      }
      history.push(prediction.token);
      if (!isSentinel(prediction.token)) {
        generated.push(prediction.token);
      }
    }
    return generated;
  }

  toJSON(): SerializedMovementModel {
    return {
      version: 1,
      backendId: this.backendId,
      order: this.order,
      vocabulary: [...this.vocabulary],
      transitions: [...this.transitions.entries()].map(
        ([key, distribution]) => [key, [...distribution.entries()]] as [string, Array<[MovementToken, number]>],
      ),
    };
  }

  static fromJSON(serialized: SerializedMovementModel): TrainedMovementModel {
    const transitions = new Map<string, Map<MovementToken, number>>();
    for (const [key, entries] of serialized.transitions) {
      transitions.set(key, new Map(entries));
    }
    return new TrainedMovementModel(serialized.backendId, serialized.order, [...serialized.vocabulary], transitions);
  }
}

/**
 * Deterministic reference backend: an order-k Markov (n-gram) model with
 * stupid-backoff decoding and lexicographic tie-breaking, so identical datasets
 * always yield identical models and predictions. Zero native dependencies.
 */
export class NGramMovementBackend implements LocalMovementBackend {
  readonly id = "ngram";

  async train(dataset: MovementDataset, options: MovementTrainOptions = {}): Promise<TrainedMovementModel> {
    const order = Math.max(1, Math.floor(options.order ?? 2));
    const transitions = new Map<string, Map<MovementToken, number>>();

    for (const sample of dataset.samples) {
      const padded = [
        ...Array.from({ length: order }, () => START_TOKEN),
        ...sample.tokens,
        END_TOKEN,
      ];
      for (let index = 1; index < padded.length; index += 1) {
        const target = padded[index]!;
        const contextStart = Math.max(0, index - order);
        // Record every backoff context length (0..order) ending at `index`.
        for (let start = contextStart; start <= index; start += 1) {
          const key = padded.slice(start, index).join(CONTEXT_SEPARATOR);
          const distribution = transitions.get(key) ?? new Map<MovementToken, number>();
          distribution.set(target, (distribution.get(target) ?? 0) + 1);
          transitions.set(key, distribution);
        }
      }
    }

    return new TrainedMovementModel(this.id, order, [...dataset.vocabulary], transitions);
  }

  load(serialized: SerializedMovementModel): TrainedMovementModel {
    return TrainedMovementModel.fromJSON(serialized);
  }
}

/** Registry that makes the movement backend swappable by id. */
export class MovementBackendRegistry {
  private readonly backends = new Map<string, LocalMovementBackend>();

  constructor(backends: LocalMovementBackend[] = [new NGramMovementBackend()]) {
    for (const backend of backends) {
      this.register(backend);
    }
  }

  register(backend: LocalMovementBackend): void {
    this.backends.set(backend.id, backend);
  }

  get(id: string): LocalMovementBackend {
    const backend = this.backends.get(id);
    if (!backend) {
      throw new Error(`Unknown movement backend: ${id}`);
    }
    return backend;
  }

  list(): string[] {
    return [...this.backends.keys()].sort();
  }
}

export type ReplayFidelityReport = {
  sampleCount: number;
  /** Total next-token predictions attempted across held-out samples. */
  predictions: number;
  /** Predictions whose argmax matched the recorded next token. */
  correct: number;
  /** `correct / predictions`, or 1 when there was nothing to predict. */
  accuracy: number;
  /** Fraction of samples the model reproduced exactly from their first token. */
  exactReplayRate: number;
};

/**
 * Generalization eval harness: measure how well a trained model reproduces
 * held-out (but related) movement samples. Reports both teacher-forced
 * next-token accuracy and end-to-end exact-replay rate.
 */
export function evaluateReplayFidelity(
  model: TrainedMovementModel,
  heldOut: MovementSample[],
): ReplayFidelityReport {
  let predictions = 0;
  let correct = 0;
  let exactReplays = 0;
  const scorable = heldOut.filter((sample) => sample.tokens.length > 0);

  for (const sample of scorable) {
    const history = Array.from<unknown, MovementToken>({ length: model.order }, () => START_TOKEN);
    for (const expected of sample.tokens) {
      const prediction = model.predictNext(history);
      predictions += 1;
      if (prediction?.token === expected) {
        correct += 1;
      }
      history.push(expected);
    }
    const replayed = model.generate([sample.tokens[0]!], sample.tokens.length * 2);
    if (arraysEqual([sample.tokens[0]!, ...replayed], sample.tokens)) {
      exactReplays += 1;
    }
  }

  return {
    sampleCount: scorable.length,
    predictions,
    correct,
    accuracy: predictions === 0 ? 1 : correct / predictions,
    exactReplayRate: scorable.length === 0 ? 0 : exactReplays / scorable.length,
  };
}

function argmax(distribution: Map<MovementToken, number>): { token: MovementToken; count: number } | undefined {
  let best: { token: MovementToken; count: number } | undefined;
  for (const [token, count] of distribution) {
    if (
      best === undefined ||
      count > best.count ||
      (count === best.count && token < best.token)
    ) {
      best = { token, count };
    }
  }
  return best;
}

function arraysEqual(a: MovementToken[], b: MovementToken[]): boolean {
  if (a.length !== b.length) {
    return false;
  }
  return a.every((value, index) => value === b[index]);
}
