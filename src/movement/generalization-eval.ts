import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  buildMovementDataset,
  type MovementDataset,
  type MovementTokenOptions,
  type TrainedMovementModel,
} from "./movement-model.js";

/**
 * Generalization eval harness (standing objective #2, piece (d)). Measures how
 * faithfully a trained movement model reproduces held-out trajectories by
 * next-movement prediction, and — critically — how well it does so on contexts
 * it never saw verbatim (the `viaBackoff` steps), which is the concrete signal
 * that it *generalizes* rather than merely memorizes.
 */

export type MovementEvalOptions = {
  /** Consider a step correct if the true token is within the top-K ranked. */
  topK?: number;
  tokenOptions?: MovementTokenOptions;
};

export type SequenceEvalResult = {
  trajectoryId: string;
  steps: number;
  correct: number;
  accuracy: number;
};

export type MovementEvalReport = {
  sequenceCount: number;
  /** Total predicted steps (every token after the first in each sequence). */
  stepCount: number;
  correct: number;
  accuracy: number;
  topK: number;
  topKCorrect: number;
  topKAccuracy: number;
  /** Steps whose exact context was unseen in training (required back-off). */
  backoffSteps: number;
  /** Correct predictions among the back-off steps. */
  generalizedCorrect: number;
  /** Accuracy restricted to back-off steps — the generalization score. */
  generalizedAccuracy: number;
  perSequence: SequenceEvalResult[];
};

function topKContains(model: TrainedMovementModel, context: string[], truth: string, k: number): boolean {
  const ranked = model.rank(context);
  for (let index = 0; index < Math.min(k, ranked.length); index += 1) {
    if (ranked[index].token === truth) {
      return true;
    }
  }
  return false;
}

/** Evaluate a trained model against an already-tokenized dataset. */
export function evaluateMovementModel(
  model: TrainedMovementModel,
  dataset: MovementDataset,
  options: MovementEvalOptions = {},
): MovementEvalReport {
  const topK = Math.max(1, Math.floor(options.topK ?? 3));
  const contextWindow = model.order - 1;

  let stepCount = 0;
  let correct = 0;
  let topKCorrect = 0;
  let backoffSteps = 0;
  let generalizedCorrect = 0;
  const perSequence: SequenceEvalResult[] = [];

  for (const sequence of dataset.sequences) {
    const tokens = sequence.tokens;
    let seqSteps = 0;
    let seqCorrect = 0;

    for (let index = 1; index < tokens.length; index += 1) {
      const context = tokens.slice(Math.max(0, index - contextWindow), index);
      const truth = tokens[index];
      const prediction = model.predict(context);

      stepCount += 1;
      seqSteps += 1;
      const hit = prediction.token === truth;
      if (hit) {
        correct += 1;
        seqCorrect += 1;
      }
      if (topKContains(model, context, truth, topK)) {
        topKCorrect += 1;
      }
      if (prediction.viaBackoff) {
        backoffSteps += 1;
        if (hit) {
          generalizedCorrect += 1;
        }
      }
    }

    perSequence.push({
      trajectoryId: sequence.trajectoryId,
      steps: seqSteps,
      correct: seqCorrect,
      accuracy: seqSteps > 0 ? seqCorrect / seqSteps : 0,
    });
  }

  return {
    sequenceCount: dataset.sequences.length,
    stepCount,
    correct,
    accuracy: stepCount > 0 ? correct / stepCount : 0,
    topK,
    topKCorrect,
    topKAccuracy: stepCount > 0 ? topKCorrect / stepCount : 0,
    backoffSteps,
    generalizedCorrect,
    generalizedAccuracy: backoffSteps > 0 ? generalizedCorrect / backoffSteps : 0,
    perSequence,
  };
}

/** Evaluate a trained model directly against held-out captured trajectories. */
export function evaluateMovementModelOnTrajectories(
  model: TrainedMovementModel,
  trajectories: TrajectorySpan[],
  options: MovementEvalOptions = {},
): MovementEvalReport {
  const dataset = buildMovementDataset(trajectories, options.tokenOptions);
  return evaluateMovementModel(model, dataset, options);
}
