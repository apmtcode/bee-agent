import {
  MovementPolicyModel,
  decodeMovementToken,
  encodeMovementStep,
  tokenizeSequence,
  type MovementSequence,
} from "./movement-model.js";

/**
 * Generalization / replay-fidelity eval harness (standing objective 2d +
 * roadmap "generalization eval harness"). Given a trained
 * {@link MovementPolicyModel} and a set of held-out sequences, it reports how
 * well the model predicts the next movement and whether a full greedy rollout
 * reproduces the recorded sequence.
 */
export type MovementEvalReport = {
  sequenceCount: number;
  stepCount: number;
  /** Teacher-forced top-1 next-step accuracy across every non-start token
   * (exact token match — penalizes novel slot values by design). */
  nextStepAccuracy: number;
  /** Teacher-forced top-1 accuracy on the *gesture* alone. This is the
   * structural-generalization signal: it stays high on held-out tasks whose
   * slot values are novel but whose movement shape was learned in training. */
  nextGestureAccuracy: number;
  /** Fraction of sequences whose greedy rollout equals the recorded steps. */
  exactRolloutRate: number;
  /** Mean per-token log-probability the model assigns to the held-out data. */
  meanLogProbability: number;
  /** Fraction of held-out steps whose predicted token was never seen as a
   * zero-probability event (i.e. the model had a non-trivial guess). */
  coverage: number;
};

function stepsEqual(a: MovementSequence["steps"], b: MovementSequence["steps"]): boolean {
  if (a.length !== b.length) return false;
  return a.every((step, i) => encodeMovementStep(step) === encodeMovementStep(b[i]!));
}

export function evaluateMovementModel(
  model: MovementPolicyModel,
  heldOut: MovementSequence[],
): MovementEvalReport {
  let stepCount = 0;
  let correct = 0;
  let gestureCorrect = 0;
  let covered = 0;
  let exactRollouts = 0;
  let logProbTotal = 0;

  for (const sequence of heldOut) {
    const tokens = tokenizeSequence(sequence);
    for (let i = 1; i < tokens.length; i += 1) {
      const context = tokens.slice(0, i);
      const expected = tokens[i]!;
      const ranked = model.predict(context);
      const top = ranked[0];
      stepCount += 1;
      if (top && top.token === expected) correct += 1;
      if (top && decodeMovementToken(top.token).gesture === decodeMovementToken(expected).gesture) {
        gestureCorrect += 1;
      }
      const prob = ranked.find((p) => p.token === expected)?.probability ?? 0;
      if (prob > 0) covered += 1;
      logProbTotal += Math.log(prob > 0 ? prob : 1e-9);
    }

    const rollout = model.generate({ maxSteps: sequence.steps.length + 4 });
    if (stepsEqual(rollout, sequence.steps)) exactRollouts += 1;
  }

  return {
    sequenceCount: heldOut.length,
    stepCount,
    nextStepAccuracy: stepCount === 0 ? 0 : correct / stepCount,
    nextGestureAccuracy: stepCount === 0 ? 0 : gestureCorrect / stepCount,
    exactRolloutRate: heldOut.length === 0 ? 0 : exactRollouts / heldOut.length,
    meanLogProbability: stepCount === 0 ? 0 : logProbTotal / stepCount,
    coverage: stepCount === 0 ? 0 : covered / stepCount,
  };
}
