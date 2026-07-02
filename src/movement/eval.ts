// Generalization eval harness for the movement-learning subsystem.
//
// Measures how well a trained model reproduces held-out (but related) movement
// trajectories. Two complementary metrics:
//   - next-step accuracy: teacher-forced top-1 token prediction accuracy. Tells
//     us whether the model has learned local transition structure.
//   - replay fidelity: free-running rollout from the intent seed compared to the
//     ground-truth token sequence via longest-common-subsequence overlap. Tells
//     us whether the model can autonomously replay the whole movement.

import {
  MOVEMENT_BOS,
  labelToken,
  tokenizeTrajectory,
  type CoordinateQuantizer,
  type MovementTrajectory,
} from "./events.js";
import type { TrainedMovementModel } from "./backend.js";

export type NextStepEvalResult = {
  total: number;
  correct: number;
  accuracy: number;
};

/**
 * Teacher-forced next-token accuracy over held-out trajectories: for every
 * position, feed the true prefix and check the model's top-1 token equals the
 * actual next token.
 */
export function evaluateNextStepAccuracy(
  model: TrainedMovementModel,
  quantizer: CoordinateQuantizer,
  heldOut: MovementTrajectory[],
): NextStepEvalResult {
  let total = 0;
  let correct = 0;
  for (const trajectory of heldOut) {
    const tokens = tokenizeTrajectory(trajectory, quantizer);
    for (let i = 1; i < tokens.length; i += 1) {
      const prediction = model.predictNextToken(tokens.slice(0, i));
      total += 1;
      if (prediction?.token === tokens[i]) {
        correct += 1;
      }
    }
  }
  return { total, correct, accuracy: total > 0 ? correct / total : 0 };
}

export type ReplayFidelityResult = {
  /** Mean LCS-overlap ratio across held-out trajectories, in [0, 1]. */
  meanFidelity: number;
  perTrajectory: Array<{ id: string; fidelity: number; exact: boolean }>;
};

/**
 * Free-running replay fidelity: seed the model with just the intent
 * (`<bos> <label:…>`), let it generate autonomously, and compare the generated
 * token sequence against ground truth using longest-common-subsequence overlap.
 */
export function evaluateReplayFidelity(
  model: TrainedMovementModel,
  quantizer: CoordinateQuantizer,
  heldOut: MovementTrajectory[],
): ReplayFidelityResult {
  const perTrajectory = heldOut.map((trajectory) => {
    const truth = tokenizeTrajectory(trajectory, quantizer);
    const seed = [MOVEMENT_BOS, labelToken(trajectory.label)];
    const generated = model.generate(seed, truth.length + 4);
    const overlap = longestCommonSubsequence(generated, truth);
    const fidelity = truth.length > 0 ? overlap / truth.length : 0;
    return {
      id: trajectory.id,
      fidelity,
      exact: generated.length === truth.length && generated.every((token, index) => token === truth[index]),
    };
  });
  const meanFidelity =
    perTrajectory.length > 0
      ? perTrajectory.reduce((sum, entry) => sum + entry.fidelity, 0) / perTrajectory.length
      : 0;
  return { meanFidelity, perTrajectory };
}

export function longestCommonSubsequence(a: string[], b: string[]): number {
  const rows = a.length;
  const cols = b.length;
  if (rows === 0 || cols === 0) return 0;
  let previous = new Array<number>(cols + 1).fill(0);
  let current = new Array<number>(cols + 1).fill(0);
  for (let i = 1; i <= rows; i += 1) {
    for (let j = 1; j <= cols; j += 1) {
      if (a[i - 1] === b[j - 1]) {
        current[j] = previous[j - 1]! + 1;
      } else {
        current[j] = Math.max(previous[j]!, current[j - 1]!);
      }
    }
    [previous, current] = [current, previous];
    current.fill(0);
  }
  return previous[cols]!;
}
