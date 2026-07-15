import type { MovementDataset, MovementModel, MovementSequence } from "./movement-model.js";

/**
 * Generalization eval harness for movement models. Measures how well a trained
 * model predicts held-out (but related) trajectories it was not trained on —
 * the objective-2d "generalize to new but related movements" signal.
 */

export type MovementEvalResult = {
  /** Number of next-token predictions scored. */
  predictions: number;
  /** Top-1 next-token accuracy across all held-out steps (0..1). */
  top1Accuracy: number;
  /** Fraction of predictions where the truth appeared anywhere in the distribution. */
  coverage: number;
  /** Mean probability the model assigned to the true next token. */
  meanTrueProbability: number;
  /** Per-sequence breakdown for debugging. */
  perSequence: Array<{
    trajectoryId?: string;
    steps: number;
    correct: number;
  }>;
};

/**
 * Score a model on held-out sequences by walking each one and, at every step,
 * asking the model to predict the next token from the true prefix (teacher
 * forcing). Empty held-out sets yield a zeroed result rather than NaN.
 */
export function evaluateMovementModel(
  model: MovementModel,
  heldOut: MovementSequence[],
): MovementEvalResult {
  let predictions = 0;
  let correct = 0;
  let covered = 0;
  let probabilityMass = 0;
  const perSequence: MovementEvalResult["perSequence"] = [];

  for (const sequence of heldOut) {
    let seqSteps = 0;
    let seqCorrect = 0;
    for (let index = 0; index < sequence.tokens.length; index += 1) {
      const context = sequence.tokens.slice(0, index);
      const truth = sequence.tokens[index]!;
      const prediction = model.predictNext(context);
      predictions += 1;
      seqSteps += 1;
      if (prediction.token === truth) {
        correct += 1;
        seqCorrect += 1;
      }
      const match = prediction.distribution.find((candidate) => candidate.token === truth);
      if (match) {
        covered += 1;
        probabilityMass += match.probability;
      }
    }
    perSequence.push({ trajectoryId: sequence.trajectoryId, steps: seqSteps, correct: seqCorrect });
  }

  return {
    predictions,
    top1Accuracy: predictions === 0 ? 0 : correct / predictions,
    coverage: predictions === 0 ? 0 : covered / predictions,
    meanTrueProbability: predictions === 0 ? 0 : probabilityMass / predictions,
    perSequence,
  };
}

/**
 * Deterministically split a dataset into train/held-out folds by round-robin on
 * sequence index (no RNG — reproducible in CI). `holdOutEvery = 3` holds out
 * every third sequence.
 */
export function splitMovementDataset(
  dataset: MovementDataset,
  holdOutEvery = 3,
): { train: MovementDataset; heldOut: MovementSequence[] } {
  const divisor = Math.max(2, Math.floor(holdOutEvery));
  const train: MovementSequence[] = [];
  const heldOut: MovementSequence[] = [];
  dataset.sequences.forEach((sequence, index) => {
    if ((index + 1) % divisor === 0) {
      heldOut.push(sequence);
    } else {
      train.push(sequence);
    }
  });
  return { train: { version: 1, sequences: train }, heldOut };
}
