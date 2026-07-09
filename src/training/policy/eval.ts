import { MOVEMENT_END_TOKEN, type TrainedMovementPolicy } from "./backend.js";
import type { MovementActionToken, MovementSequence } from "./movement-dataset.js";

/**
 * Generalization eval harness for trained movement policies. It measures how well
 * a policy reproduces held-out (but related) movement sequences it may not have
 * trained on, under two regimes:
 *
 * - **next-token accuracy** (teacher-forcing): given the true prefix and the
 *   step's observation, does the policy's top prediction match the recorded next
 *   action? Averaged over every step.
 * - **rollout fidelity**: seeding only the first action (and observation), does a
 *   greedy rollout reproduce the remainder of the sequence exactly, and what
 *   fraction of the steps line up?
 */
export type MovementEvalResult = {
  sequenceCount: number;
  stepCount: number;
  /** Fraction of steps whose top-1 next-token prediction matched (0..1). */
  nextTokenAccuracy: number;
  /** Fraction of held-out sequences reproduced exactly by greedy rollout (0..1). */
  rolloutExactMatch: number;
  /** Mean per-sequence fraction of correctly reproduced rollout steps (0..1). */
  rolloutStepFidelity: number;
  /** Mean confidence of the top prediction across all teacher-forced steps (0..1). */
  averageConfidence: number;
};

export type MovementEvalOptions = {
  /** Include the end-of-sequence marker as a predicted step in next-token scoring. */
  scoreEndToken?: boolean;
};

export function evaluateMovementPolicy(
  policy: TrainedMovementPolicy,
  heldOut: readonly MovementSequence[],
  options: MovementEvalOptions = {},
): MovementEvalResult {
  let steps = 0;
  let correct = 0;
  let confidenceTotal = 0;
  let rolloutExact = 0;
  let fidelityTotal = 0;
  let scoredSequences = 0;

  for (const sequence of heldOut) {
    const tokens = sequence.steps.map((step) => step.token);
    if (tokens.length === 0) {
      continue;
    }
    scoredSequences += 1;

    const target = options.scoreEndToken ? [...tokens, MOVEMENT_END_TOKEN] : tokens;
    for (let index = 0; index < target.length; index += 1) {
      const observation = index < sequence.steps.length ? sequence.steps[index]?.observation : undefined;
      const prediction = policy.predict({ prefix: tokens.slice(0, index), observation });
      steps += 1;
      confidenceTotal += prediction.confidence;
      if (prediction.token === target[index]) {
        correct += 1;
      }
    }

    const remainder = tokens.slice(1);
    const generated = policy.rollout({
      prefix: tokens.slice(0, 1),
      observation: sequence.steps[0]?.observation,
      maxSteps: remainder.length,
    });
    fidelityTotal += stepFidelity(remainder, generated);
    if (sequencesEqual(remainder, generated)) {
      rolloutExact += 1;
    }
  }

  return {
    sequenceCount: scoredSequences,
    stepCount: steps,
    nextTokenAccuracy: steps === 0 ? 0 : correct / steps,
    rolloutExactMatch: scoredSequences === 0 ? 0 : rolloutExact / scoredSequences,
    rolloutStepFidelity: scoredSequences === 0 ? 0 : fidelityTotal / scoredSequences,
    averageConfidence: steps === 0 ? 0 : confidenceTotal / steps,
  };
}

function stepFidelity(expected: readonly MovementActionToken[], actual: readonly MovementActionToken[]): number {
  if (expected.length === 0) {
    return actual.length === 0 ? 1 : 0;
  }
  let matched = 0;
  for (let index = 0; index < expected.length; index += 1) {
    if (expected[index] === actual[index]) {
      matched += 1;
    }
  }
  return matched / expected.length;
}

function sequencesEqual(a: readonly MovementActionToken[], b: readonly MovementActionToken[]): boolean {
  return a.length === b.length && a.every((token, index) => token === b[index]);
}
