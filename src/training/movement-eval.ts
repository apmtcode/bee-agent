import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  movementTokenKey,
  tokenizeTrajectory,
  type MovementToken,
  type TrainedMovementModel,
} from "./movement-model.js";

/**
 * Generalization eval harness for movement models.
 *
 * Measures how well a trained {@link TrainedMovementModel} reproduces held-out
 * (but related) movement sequences — the objective #2 (d) requirement of
 * generalizing to "new but related movements". For each sequence we walk every
 * prefix, ask the model for the next token, and score it against the truth:
 *
 *  - next-token accuracy (top-1) and top-k accuracy,
 *  - replay fidelity: the longest correct prefix the model reproduces from the
 *    sequence start, normalized by sequence length, and
 *  - generalization rate: the share of correct predictions that required
 *    feature-similarity backoff (i.e. the context was never seen in training).
 */

export type MovementEvalOptions = {
  /** k for top-k accuracy. Defaults to 3. */
  topK?: number;
};

export type MovementSequenceEval = {
  length: number;
  top1Correct: number;
  topKCorrect: number;
  /** Longest correct run starting at the sequence head, normalized to [0,1]. */
  replayFidelity: number;
  /** Correct top-1 predictions that required similarity generalization. */
  generalizedCorrect: number;
};

export type MovementEvalReport = {
  sequenceCount: number;
  predictionCount: number;
  topK: number;
  /** Top-1 next-token accuracy across all predictions. */
  accuracy: number;
  /** Top-k next-token accuracy across all predictions. */
  topKAccuracy: number;
  /** Mean replay fidelity across evaluated sequences. */
  meanReplayFidelity: number;
  /** Share of correct top-1 predictions that came from generalization. */
  generalizationRate: number;
  perSequence: MovementSequenceEval[];
};

function evaluateSequence(
  model: TrainedMovementModel,
  sequence: MovementToken[],
  topK: number,
): MovementSequenceEval {
  let top1Correct = 0;
  let topKCorrect = 0;
  let generalizedCorrect = 0;
  let leadingCorrect = 0;
  let prefixStillCorrect = true;

  for (let i = 0; i < sequence.length; i += 1) {
    const context = sequence.slice(0, i);
    const expectedKey = movementTokenKey(sequence[i]!);
    const prediction = model.predictNext(context);

    const isTop1 = prediction.key === expectedKey;
    if (isTop1) {
      top1Correct += 1;
      if (prediction.generalized) {
        generalizedCorrect += 1;
      }
    }

    const inTopK = prediction.candidates
      .slice(0, topK)
      .some((candidate) => candidate.key === expectedKey);
    if (inTopK) {
      topKCorrect += 1;
    }

    if (prefixStillCorrect && isTop1) {
      leadingCorrect += 1;
    } else {
      prefixStillCorrect = false;
    }
  }

  return {
    length: sequence.length,
    top1Correct,
    topKCorrect,
    replayFidelity: sequence.length === 0 ? 1 : leadingCorrect / sequence.length,
    generalizedCorrect,
  };
}

/** Evaluate a trained model against held-out movement sequences. */
export function evaluateMovementModel(
  model: TrainedMovementModel,
  heldOut: MovementToken[][],
  options: MovementEvalOptions = {},
): MovementEvalReport {
  const topK = Math.max(1, Math.floor(options.topK ?? 3));
  const perSequence = heldOut
    .filter((sequence) => sequence.length > 0)
    .map((sequence) => evaluateSequence(model, sequence, topK));

  let predictionCount = 0;
  let top1 = 0;
  let topKTotal = 0;
  let generalizedCorrect = 0;
  let fidelitySum = 0;

  for (const result of perSequence) {
    predictionCount += result.length;
    top1 += result.top1Correct;
    topKTotal += result.topKCorrect;
    generalizedCorrect += result.generalizedCorrect;
    fidelitySum += result.replayFidelity;
  }

  return {
    sequenceCount: perSequence.length,
    predictionCount,
    topK,
    accuracy: predictionCount === 0 ? 0 : top1 / predictionCount,
    topKAccuracy: predictionCount === 0 ? 0 : topKTotal / predictionCount,
    meanReplayFidelity: perSequence.length === 0 ? 0 : fidelitySum / perSequence.length,
    generalizationRate: top1 === 0 ? 0 : generalizedCorrect / top1,
    perSequence,
  };
}

/** Convenience: evaluate directly against held-out trajectory spans. */
export function evaluateMovementModelOnTrajectories(
  model: TrainedMovementModel,
  heldOut: TrajectorySpan[],
  options?: MovementEvalOptions,
): MovementEvalReport {
  return evaluateMovementModel(model, heldOut.map(tokenizeTrajectory), options);
}
