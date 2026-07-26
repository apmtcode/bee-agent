// Local-movement learning subsystem — in-process trainable model backend.
//
// The reviewed-export pipeline (`exporter.ts`) and the external-training planner
// (`runner.ts`) produce *plans* that run on a real machine (mlx/axolotl on Apple
// Silicon). Those cannot run in the cloud/CI. This module provides the missing
// piece: a *pluggable, deterministic, in-process* movement model that can be
// trained on a recorded dataset and then (c) repeat and (d) generalize movements
// — with zero external dependencies, so it is fully exercised by the test suite.
//
// The `MovementModelBackend` interface is the pluggable seam: `NgramMovementBackend`
// is the deterministic reference/mock backend; a real on-device small model can
// implement the same interface behind the same call sites.

import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";
import type { ReplayManifest } from "../capture/replay.js";

/** A canonical, replayable movement token (e.g. `device/swipe/down`). */
export type MovementToken = string;

/** Sentinel emitted when the model predicts a sequence has ended. */
export const MOVEMENT_END: MovementToken = "<end>";
/** Internal sentinel marking sequence start (never emitted as a prediction). */
const MOVEMENT_START: MovementToken = "<start>";
const CONTEXT_SEPARATOR = "␟";

export type MovementSequence = {
  trajectoryId?: string;
  tokens: MovementToken[];
};

export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
  /** Sorted unique tokens observed across all sequences (excludes sentinels). */
  vocabulary: MovementToken[];
};

export type MovementModelConfig = {
  /** Maximum n-gram context length used for prediction. Default 3. */
  order?: number;
};

export type MovementDistributionEntry = {
  token: MovementToken;
  probability: number;
};

export type MovementPrediction = {
  /** Most likely next movement, or `undefined` when the model is empty. */
  token: MovementToken | undefined;
  probability: number;
  /**
   * Length of the context suffix that actually produced the prediction. A value
   * shorter than the supplied context means the model *backed off* — i.e. it
   * generalized from a shorter, previously-seen pattern.
   */
  backoffOrder: number;
  distribution: MovementDistributionEntry[];
};

export type SerializedMovementModel = {
  version: 1;
  backendId: string;
  order: number;
  vocabulary: MovementToken[];
  contexts: Array<{ context: MovementToken[]; nexts: Array<[MovementToken, number]> }>;
};

export interface TrainedMovementModel {
  readonly backendId: string;
  readonly order: number;
  readonly vocabulary: MovementToken[];
  /** Predict the next movement given a (possibly novel) context prefix. */
  predictNext(context: MovementToken[]): MovementPrediction;
  /** Greedily roll out up to `maxLength` new movements after `seed`. */
  generate(seed: MovementToken[], maxLength: number): MovementToken[];
  /** Average natural-log probability the model assigns to a full sequence. */
  sequenceLogProb(tokens: MovementToken[]): number;
  toJSON(): SerializedMovementModel;
}

export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, config?: MovementModelConfig): Promise<TrainedMovementModel>;
}

// --- dataset construction -------------------------------------------------

function slug(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "") || "movement";
}

/** Derive a stable, structured movement token from a recorded action. */
export function movementTokenFromAction(action: TrajectoryAction): MovementToken {
  const metadata = action.metadata ?? {};
  const parts: string[] = [slug(action.tool)];
  const gesture = metadata["gesture"];
  const direction = metadata["direction"];
  const target = metadata["target"];
  if (typeof gesture === "string") {
    parts.push(slug(gesture));
  }
  if (typeof direction === "string") {
    parts.push(slug(direction));
  } else if (typeof target === "string") {
    parts.push(slug(target));
  }
  if (parts.length === 1) {
    parts.push(slug(action.summary));
  }
  return parts.join("/");
}

function sequenceFromActions(id: string | undefined, actions: TrajectoryAction[]): MovementSequence {
  const tokens = [...actions]
    .sort((a, b) => a.ts - b.ts)
    .map((action) => movementTokenFromAction(action));
  return id === undefined ? { tokens } : { trajectoryId: id, tokens };
}

function datasetFromSequences(sequences: MovementSequence[]): MovementDataset {
  const nonEmpty = sequences.filter((sequence) => sequence.tokens.length > 0);
  const vocabulary = [...new Set(nonEmpty.flatMap((sequence) => sequence.tokens))].sort();
  return { version: 1, sequences: nonEmpty, vocabulary };
}

/** Build a training dataset from recorded trajectory spans. */
export function buildMovementDataset(trajectories: TrajectorySpan[]): MovementDataset {
  return datasetFromSequences(
    trajectories.map((trajectory) => sequenceFromActions(trajectory.id, trajectory.actions)),
  );
}

/** Build a training dataset from replay manifests (post-review/redaction). */
export function buildMovementDatasetFromReplays(replays: ReplayManifest[]): MovementDataset {
  const sequences: MovementSequence[] = replays.map((replay) => {
    const tokens = replay.events
      .filter((event): event is Extract<ReplayManifest["events"][number], { kind: "action" }> => event.kind === "action")
      .sort((a, b) => a.ts - b.ts)
      .map((event) => `${slug(event.tool)}/${slug(event.summary)}`);
    return { trajectoryId: replay.sessionId, tokens };
  });
  return datasetFromSequences(sequences);
}

// --- n-gram backend -------------------------------------------------------

function contextKey(context: MovementToken[]): string {
  return context.join(CONTEXT_SEPARATOR);
}

class NgramMovementModel implements TrainedMovementModel {
  constructor(
    public readonly backendId: string,
    public readonly order: number,
    public readonly vocabulary: MovementToken[],
    private readonly table: Map<string, Map<MovementToken, number>>,
  ) {}

  predictNext(context: MovementToken[]): MovementPrediction {
    const effective = [MOVEMENT_START, ...context];
    const maxK = Math.min(this.order, effective.length);
    for (let k = maxK; k >= 0; k -= 1) {
      const suffix = effective.slice(effective.length - k);
      const counts = this.table.get(contextKey(suffix));
      if (!counts || counts.size === 0) {
        continue;
      }
      const total = [...counts.values()].reduce((sum, count) => sum + count, 0);
      if (total === 0) {
        continue;
      }
      const distribution = [...counts.entries()]
        .map(([token, count]) => ({ token, probability: count / total }))
        .sort((a, b) => (b.probability - a.probability) || (a.token < b.token ? -1 : a.token > b.token ? 1 : 0));
      const top = distribution[0];
      if (top) {
        return { token: top.token, probability: top.probability, backoffOrder: k, distribution };
      }
    }
    return { token: undefined, probability: 0, backoffOrder: 0, distribution: [] };
  }

  generate(seed: MovementToken[], maxLength: number): MovementToken[] {
    const generated: MovementToken[] = [];
    let context = [...seed];
    for (let step = 0; step < maxLength; step += 1) {
      const prediction = this.predictNext(context);
      if (prediction.token === undefined || prediction.token === MOVEMENT_END) {
        break;
      }
      generated.push(prediction.token);
      context = [...context, prediction.token];
    }
    return generated;
  }

  sequenceLogProb(tokens: MovementToken[]): number {
    const targets = [...tokens, MOVEMENT_END];
    if (targets.length === 0) {
      return 0;
    }
    let total = 0;
    for (let index = 0; index < targets.length; index += 1) {
      const context = tokens.slice(0, index);
      const prediction = this.predictNext(context);
      const match = prediction.distribution.find((entry) => entry.token === targets[index]);
      const probability = match?.probability ?? 1e-9;
      total += Math.log(probability);
    }
    return total / targets.length;
  }

  toJSON(): SerializedMovementModel {
    const contexts = [...this.table.entries()].map(([key, counts]) => ({
      context: key === "" ? [] : key.split(CONTEXT_SEPARATOR),
      nexts: [...counts.entries()].sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0)),
    }));
    contexts.sort((a, b) => contextKey(a.context) < contextKey(b.context) ? -1 : 1);
    return { version: 1, backendId: this.backendId, order: this.order, vocabulary: this.vocabulary, contexts };
  }
}

/**
 * Deterministic n-gram backend with Katz-style backoff. Learns movement
 * transition statistics from recorded sequences; generalizes to unseen prefixes
 * by falling back to the longest previously-seen context suffix. No external
 * model, no randomness — identical dataset in, identical model out.
 */
export class NgramMovementBackend implements MovementModelBackend {
  readonly id = "ngram-v1";

  async train(dataset: MovementDataset, config?: MovementModelConfig): Promise<TrainedMovementModel> {
    const order = Math.max(1, Math.floor(config?.order ?? 3));
    const table = new Map<string, Map<MovementToken, number>>();

    const record = (context: MovementToken[], next: MovementToken): void => {
      const key = contextKey(context);
      let counts = table.get(key);
      if (!counts) {
        counts = new Map<MovementToken, number>();
        table.set(key, counts);
      }
      counts.set(next, (counts.get(next) ?? 0) + 1);
    };

    for (const sequence of dataset.sequences) {
      const framed = [MOVEMENT_START, ...sequence.tokens, MOVEMENT_END];
      for (let index = 1; index < framed.length; index += 1) {
        const next = framed[index]!;
        const maxK = Math.min(order, index);
        for (let k = 0; k <= maxK; k += 1) {
          record(framed.slice(index - k, index), next);
        }
      }
    }

    return new NgramMovementModel(this.id, order, [...dataset.vocabulary], table);
  }
}

/** Rehydrate a persisted n-gram model produced by `TrainedMovementModel.toJSON`. */
export function loadMovementModel(serialized: SerializedMovementModel): TrainedMovementModel {
  const table = new Map<string, Map<MovementToken, number>>();
  for (const entry of serialized.contexts) {
    table.set(contextKey(entry.context), new Map(entry.nexts));
  }
  return new NgramMovementModel(serialized.backendId, serialized.order, [...serialized.vocabulary], table);
}

export function createNgramMovementBackend(): MovementModelBackend {
  return new NgramMovementBackend();
}

// --- generalization / fidelity eval --------------------------------------

export type MovementEvalResult = {
  /** Positions evaluated (one per token, per held-out sequence). */
  totalPredictions: number;
  /** Fraction where the model's top prediction matched the actual next move. */
  nextTokenAccuracy: number;
  /** Mean natural-log probability the model assigned to the actual next move. */
  meanLogProb: number;
  /**
   * Fraction of correct predictions that required backoff to a *shorter*
   * context than the supplied prefix — evidence of generalization rather than
   * memorization.
   */
  generalizationRate: number;
};

/**
 * Measure next-movement prediction fidelity on held-out sequences. This is the
 * generalization eval harness: it reports both raw accuracy and how much of that
 * accuracy came from backed-off (generalized) predictions.
 */
export function evaluateMovementModel(
  model: TrainedMovementModel,
  heldOut: MovementSequence[],
): MovementEvalResult {
  let totalPredictions = 0;
  let correct = 0;
  let generalizedCorrect = 0;
  let logProbSum = 0;

  for (const sequence of heldOut) {
    for (let index = 0; index < sequence.tokens.length; index += 1) {
      const context = sequence.tokens.slice(0, index);
      const actual = sequence.tokens[index]!;
      const prediction = model.predictNext(context);
      totalPredictions += 1;
      const match = prediction.distribution.find((entry) => entry.token === actual);
      logProbSum += Math.log(match?.probability ?? 1e-9);
      if (prediction.token === actual) {
        correct += 1;
        // effective context = [START, ...context]; a match shorter than that
        // means the model generalized from a shorter seen pattern.
        if (prediction.backoffOrder < context.length + 1) {
          generalizedCorrect += 1;
        }
      }
    }
  }

  return {
    totalPredictions,
    nextTokenAccuracy: totalPredictions === 0 ? 0 : correct / totalPredictions,
    meanLogProb: totalPredictions === 0 ? 0 : logProbSum / totalPredictions,
    generalizationRate: correct === 0 ? 0 : generalizedCorrect / correct,
  };
}
