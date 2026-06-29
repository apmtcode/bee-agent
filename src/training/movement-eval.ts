/**
 * Generalization eval harness for the local-movement learning subsystem.
 *
 * Two complementary fidelity measures over held-out (but related) movement
 * sequences:
 *
 *  - **next-token** (teacher-forced): at each position predict the true next
 *    movement from the true preceding context. Does not compound errors, so it
 *    isolates "did the model learn the local transition statistics".
 *  - **rollout** (free-running): prime the model with a short prefix and let it
 *    generate the remainder, then compare element-wise. This is the harder,
 *    replay-faithfulness measure — errors compound, and it rewards a model that
 *    generalizes across trajectories rather than memorising one.
 *
 * "Held-out but related" is the whole point of objective #2(d): evaluate on
 * sequences the model was not trained on but which share sub-movements with the
 * training set, and confirm the model can still reproduce/continue them.
 */

import {
  movementTokensEqual,
  type MovementSequence,
  type TrainedMovementModel,
} from "./movement-model.js";

/** Per-sequence next-token (teacher-forced) result. */
export type NextTokenSequenceResult = {
  trajectoryId: string;
  predictions: number;
  correct: number;
  accuracy: number;
};

/** Per-sequence rollout (free-running) result. */
export type RolloutSequenceResult = {
  trajectoryId: string;
  promptLength: number;
  expected: number;
  matched: number;
  fidelity: number;
};

/** Aggregate report across all held-out sequences. */
export type MovementEvalReport = {
  sequenceCount: number;
  nextToken: {
    predictions: number;
    correct: number;
    accuracy: number;
    perSequence: NextTokenSequenceResult[];
  };
  rollout: {
    expected: number;
    matched: number;
    fidelity: number;
    perSequence: RolloutSequenceResult[];
  };
};

export type MovementEvalOptions = {
  /**
   * How many leading movements prime the rollout. Sequences shorter than
   * `promptLength + 1` are skipped for rollout (no continuation to score).
   * Defaults to the model's Markov order.
   */
  promptLength?: number;
};

function evaluateNextTokenSequence(
  model: TrainedMovementModel,
  sequence: MovementSequence,
): NextTokenSequenceResult {
  let predictions = 0;
  let correct = 0;
  for (let i = 1; i < sequence.tokens.length; i += 1) {
    const context = sequence.tokens.slice(0, i);
    const expected = sequence.tokens[i];
    const prediction = model.predict(context);
    predictions += 1;
    if (prediction && movementTokensEqual(prediction.token, expected)) {
      correct += 1;
    }
  }
  return {
    trajectoryId: sequence.trajectoryId,
    predictions,
    correct,
    accuracy: predictions > 0 ? correct / predictions : 0,
  };
}

function evaluateRolloutSequence(
  model: TrainedMovementModel,
  sequence: MovementSequence,
  promptLength: number,
): RolloutSequenceResult | undefined {
  if (sequence.tokens.length <= promptLength) {
    return undefined;
  }
  const prefix = sequence.tokens.slice(0, promptLength);
  const expected = sequence.tokens.slice(promptLength);
  const generated = model.generate(prefix, expected.length);
  let matched = 0;
  for (let i = 0; i < expected.length; i += 1) {
    const produced = generated[i];
    if (produced && movementTokensEqual(produced, expected[i])) {
      matched += 1;
    }
  }
  return {
    trajectoryId: sequence.trajectoryId,
    promptLength,
    expected: expected.length,
    matched,
    fidelity: expected.length > 0 ? matched / expected.length : 0,
  };
}

/** Score a trained model against held-out movement sequences. */
export function evaluateMovementModel(
  model: TrainedMovementModel,
  heldOut: MovementSequence[],
  options?: MovementEvalOptions,
): MovementEvalReport {
  const promptLength = Math.max(1, Math.floor(options?.promptLength ?? model.order));

  const nextTokenPerSequence = heldOut.map((sequence) => evaluateNextTokenSequence(model, sequence));
  const nextTokenPredictions = nextTokenPerSequence.reduce((sum, result) => sum + result.predictions, 0);
  const nextTokenCorrect = nextTokenPerSequence.reduce((sum, result) => sum + result.correct, 0);

  const rolloutPerSequence = heldOut
    .map((sequence) => evaluateRolloutSequence(model, sequence, promptLength))
    .filter((result): result is RolloutSequenceResult => result !== undefined);
  const rolloutExpected = rolloutPerSequence.reduce((sum, result) => sum + result.expected, 0);
  const rolloutMatched = rolloutPerSequence.reduce((sum, result) => sum + result.matched, 0);

  return {
    sequenceCount: heldOut.length,
    nextToken: {
      predictions: nextTokenPredictions,
      correct: nextTokenCorrect,
      accuracy: nextTokenPredictions > 0 ? nextTokenCorrect / nextTokenPredictions : 0,
      perSequence: nextTokenPerSequence,
    },
    rollout: {
      expected: rolloutExpected,
      matched: rolloutMatched,
      fidelity: rolloutExpected > 0 ? rolloutMatched / rolloutExpected : 0,
      perSequence: rolloutPerSequence,
    },
  };
}
