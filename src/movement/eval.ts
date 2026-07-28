// Generalization / replay-fidelity evaluation for movement models.
//
// Measures how well a trained model reproduces a *held-out* trajectory: the
// fraction of the demonstrated action sequence the model recovers, plus whether
// it had to generalize (backoff) to do so. Used to prove the model repeats
// recorded movements and generalizes to new-but-related contexts.

import type { MovementModel, MovementModelBackend, MovementTrajectory } from "./movement-model.js";

export type ReplayFidelity = {
  /** Length-normalized longest-common-subsequence overlap of actions, 0..1. */
  actionOverlap: number;
  /** Fraction of predicted steps whose action AND target match in order. */
  exactStepAccuracy: number;
  predictedLength: number;
  expectedLength: number;
  /** Max backoff level the model used (0 = exact context, higher = generalized). */
  maxBackoffLevel: number;
};

function lcsLength(a: string[], b: string[]): number {
  const rows = a.length + 1;
  const cols = b.length + 1;
  const dp = new Array<number>(rows * cols).fill(0);
  for (let i = 1; i < rows; i += 1) {
    for (let j = 1; j < cols; j += 1) {
      dp[i * cols + j] = a[i - 1] === b[j - 1]
        ? dp[(i - 1) * cols + (j - 1)] + 1
        : Math.max(dp[(i - 1) * cols + j], dp[i * cols + (j - 1)]);
    }
  }
  return dp[a.length * cols + b.length];
}

function stepKey(step: { action: string; target?: string }): string {
  return step.target ? `${step.action}#${step.target}` : step.action;
}

/**
 * Roll the model out for a held-out trajectory's context and compare the
 * predicted sequence against the demonstrated one.
 */
export async function evaluateReplayFidelity(
  backend: MovementModelBackend,
  model: MovementModel,
  expected: MovementTrajectory,
): Promise<ReplayFidelity> {
  const prediction = await backend.predict(model, {
    context: expected.context,
    maxSteps: Math.max(expected.steps.length * 2, 8),
  });

  const expectedActions = expected.steps.map((step) => step.action);
  const predictedActions = prediction.steps.map((step) => step.action);
  const overlap = expectedActions.length === 0 ? 1 : lcsLength(expectedActions, predictedActions) / expectedActions.length;

  const expectedKeys = expected.steps.map(stepKey);
  const predictedKeys = prediction.steps.map(stepKey);
  const compared = Math.min(expectedKeys.length, predictedKeys.length);
  let exact = 0;
  for (let i = 0; i < compared; i += 1) {
    if (expectedKeys[i] === predictedKeys[i]) {
      exact += 1;
    }
  }
  const exactStepAccuracy = expectedKeys.length === 0 ? 1 : exact / expectedKeys.length;

  return {
    actionOverlap: overlap,
    exactStepAccuracy,
    predictedLength: prediction.steps.length,
    expectedLength: expected.steps.length,
    maxBackoffLevel: prediction.maxBackoffLevel,
  };
}
