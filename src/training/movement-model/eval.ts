import type {
  MovementDataset,
  MovementModelBackend,
  MovementSequence,
  MovementToken,
  TrainedMovementModel,
} from "./types.js";

/**
 * Generalization eval harness.
 *
 * Measures how well a trained movement model reproduces held-out (but related)
 * trajectories. Two complementary signals:
 *  - next-step accuracy: at each position, does the argmax prediction match the
 *    recorded next movement? (fidelity of local prediction)
 *  - replay fidelity: does an argmax rollout from the sequence's first step
 *    regenerate the recorded movement sequence? (end-to-end repeatability)
 */

export type MovementEvalResult = {
  sequenceCount: number;
  stepCount: number;
  correctSteps: number;
  nextStepAccuracy: number;
  exactReplaySequences: number;
  replayFidelity: number;
  /** Fraction of predictions that required backing off below the model order. */
  backoffRate: number;
};

export function evaluateMovementModel(
  backend: MovementModelBackend,
  model: TrainedMovementModel,
  dataset: MovementDataset,
): MovementEvalResult {
  let stepCount = 0;
  let correctSteps = 0;
  let backoffSteps = 0;
  let exactReplaySequences = 0;
  const order = model.metadata.order;

  for (const sequence of dataset.sequences) {
    const tokens = sequence.steps.map((step) => step.token);
    for (let i = 0; i < tokens.length; i += 1) {
      const context = tokens.slice(0, i);
      const [prediction] = backend.predict(model, context, { topK: 1 });
      stepCount += 1;
      if (prediction?.token === tokens[i]) {
        correctSteps += 1;
      }
      if (prediction && prediction.order < Math.min(order, context.length)) {
        backoffSteps += 1;
      }
    }
    if (isExactReplay(backend, model, sequence)) {
      exactReplaySequences += 1;
    }
  }

  return {
    sequenceCount: dataset.sequences.length,
    stepCount,
    correctSteps,
    nextStepAccuracy: stepCount === 0 ? 0 : correctSteps / stepCount,
    exactReplaySequences,
    replayFidelity:
      dataset.sequences.length === 0 ? 0 : exactReplaySequences / dataset.sequences.length,
    backoffRate: stepCount === 0 ? 0 : backoffSteps / stepCount,
  };
}

/**
 * True when an argmax rollout seeded with the sequence's first movement
 * regenerates the remaining recorded movements exactly.
 */
export function isExactReplay(
  backend: MovementModelBackend,
  model: TrainedMovementModel,
  sequence: MovementSequence,
): boolean {
  const tokens = sequence.steps.map((step) => step.token);
  if (tokens.length === 0) {
    return true;
  }
  const seed: MovementToken[] = tokens.slice(0, 1);
  const generated = backend.generate(model, seed, tokens.length + 1);
  const rollout = [...seed, ...generated];
  if (rollout.length !== tokens.length) {
    return false;
  }
  return rollout.every((token, index) => token === tokens[index]);
}
