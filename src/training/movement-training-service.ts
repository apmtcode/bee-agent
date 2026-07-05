// Local-movement training service + generalization eval harness.
//
// Ties the pieces together end-to-end: build a dataset from reviewed
// trajectories/replays, train a pluggable backend on it, and measure both
// *replay fidelity* (can the model reproduce recorded movements?) and
// *generalization* (does it assign high probability / reproduce held-out but
// related movements?). All in-process and deterministic, so it validates the
// subsystem in the cloud against synthetic event streams.

import type { TrajectorySpan } from "../capture/trajectory.js";
import type { ReviewedExportManifest } from "./export-manifest.js";
import {
  MOVEMENT_START_TOKEN,
  buildMovementDatasetFromReplays,
  buildMovementDatasetFromTrajectories,
  createMovementBackend,
  splitMovementDataset,
  type MovementBackendKind,
  type MovementDataset,
  type MovementModelBackend,
  type MovementToken,
  type TrainedMovementModel,
} from "./movement-model.js";

/** Per-sequence fidelity/generalization measurement. */
export type MovementSequenceEvaluation = {
  sequenceId: string;
  expected: MovementToken[];
  generated: MovementToken[];
  /** Greedy generation from `<start>` exactly reproduced the sequence. */
  exactMatch: boolean;
  /** Fraction of teacher-forced next-token predictions that were correct. */
  tokenAccuracy: number;
  /** Mean log-probability the model assigns to the sequence. */
  logProb: number;
};

/** Aggregate evaluation across a set of sequences. */
export type MovementEvaluation = {
  sequenceCount: number;
  exactMatchRate: number;
  meanTokenAccuracy: number;
  meanLogProb: number;
  perSequence: MovementSequenceEvaluation[];
};

export type MovementTrainingResult = {
  dataset: MovementDataset;
  model: TrainedMovementModel;
  /** Fidelity on the training set (memorization / replay quality). */
  replayEvaluation: MovementEvaluation;
  /** Generalization on a deterministically held-out split (undefined if none). */
  generalizationEvaluation?: MovementEvaluation;
};

export type MovementTrainingOptions = {
  /** Hold out every Nth sequence for a generalization measurement (0 = none). */
  holdoutEvery?: number;
};

/**
 * Evaluate a trained model against a dataset: teacher-forced token accuracy,
 * greedy-generation exact match, and mean log-probability per sequence.
 */
export function evaluateMovementModel(
  model: TrainedMovementModel,
  dataset: MovementDataset,
): MovementEvaluation {
  const perSequence = dataset.sequences.map((sequence): MovementSequenceEvaluation => {
    // Greedy generation from the start sentinel = "what movement does the model
    // perform unprompted"; exactMatch checks it reproduces a recorded sequence.
    const generated = model.generate({ seed: [MOVEMENT_START_TOKEN] });
    // Teacher-forced accuracy: does argmax at each recorded prefix match?
    let correct = 0;
    const context: MovementToken[] = [MOVEMENT_START_TOKEN];
    for (const token of sequence.tokens) {
      const prediction = model.predictNext(context);
      if (prediction && prediction.token === token) {
        correct += 1;
      }
      context.push(token);
    }
    const tokenAccuracy = sequence.tokens.length === 0 ? 1 : correct / sequence.tokens.length;
    return {
      sequenceId: sequence.id,
      expected: sequence.tokens,
      generated,
      exactMatch: arraysEqual(generated, sequence.tokens),
      tokenAccuracy,
      logProb: model.scoreSequence(sequence.tokens),
    };
  });

  const sequenceCount = perSequence.length;
  const exactMatches = perSequence.filter((entry) => entry.exactMatch).length;
  return {
    sequenceCount,
    exactMatchRate: sequenceCount === 0 ? 0 : exactMatches / sequenceCount,
    meanTokenAccuracy: mean(perSequence.map((entry) => entry.tokenAccuracy)),
    meanLogProb: mean(perSequence.map((entry) => entry.logProb)),
    perSequence,
  };
}

/**
 * Trains a pluggable movement-model backend on a recorded movement dataset and
 * reports replay-fidelity + generalization. The default backend is the
 * deterministic in-process Markov model; pass a different `backend` (or use
 * `createMovementBackend`) to swap in a real on-device model later.
 */
export class MovementTrainingService {
  private readonly backend: MovementModelBackend;

  constructor(backend?: MovementModelBackend | MovementBackendKind) {
    this.backend =
      backend === undefined
        ? createMovementBackend("markov")
        : typeof backend === "string"
          ? createMovementBackend(backend)
          : backend;
  }

  /** Train directly from a prepared dataset. */
  trainFromDataset(dataset: MovementDataset, options: MovementTrainingOptions = {}): MovementTrainingResult {
    const holdoutEvery = options.holdoutEvery ?? 0;
    if (holdoutEvery > 0 && dataset.sequences.length > holdoutEvery) {
      const { train, holdout } = splitMovementDataset(dataset, holdoutEvery);
      const model = this.backend.train(train);
      return {
        dataset: train,
        model,
        replayEvaluation: evaluateMovementModel(model, train),
        generalizationEvaluation:
          holdout.sequences.length > 0 ? evaluateMovementModel(model, holdout) : undefined,
      };
    }
    const model = this.backend.train(dataset);
    return {
      dataset,
      model,
      replayEvaluation: evaluateMovementModel(model, dataset),
    };
  }

  /** Build a dataset from reviewed trajectory spans, then train. */
  trainFromTrajectories(
    trajectories: TrajectorySpan[],
    options: MovementTrainingOptions = {},
  ): MovementTrainingResult {
    return this.trainFromDataset(buildMovementDatasetFromTrajectories(trajectories), options);
  }

  /** Build a dataset from a reviewed export's replays, then train. */
  trainFromExport(
    manifest: ReviewedExportManifest,
    options: MovementTrainingOptions = {},
  ): MovementTrainingResult {
    return this.trainFromDataset(buildMovementDatasetFromReplays(manifest.replays), options);
  }
}

function arraysEqual(a: MovementToken[], b: MovementToken[]): boolean {
  if (a.length !== b.length) {
    return false;
  }
  for (let i = 0; i < a.length; i += 1) {
    if (a[i] !== b[i]) {
      return false;
    }
  }
  return true;
}

function mean(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}
