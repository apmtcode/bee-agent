/**
 * Generalization eval harness for the local-movement model.
 *
 * Measures how well a trained {@link TrainedMovementModel} predicts the next
 * movement on held-out sequences. Two accuracies are reported:
 *   - `exactMatchAccuracy` — predicted movement equals the actual one (tool,
 *     gesture, direction, and target all match).
 *   - `gestureMatchAccuracy` — predicted *shape* (tool + gesture + direction)
 *     matches, ignoring the specific target. This is the meaningful signal for
 *     "generalize to new but related movements": the target label may be novel,
 *     but the model should still choose the right kind of movement.
 *
 * Evaluation uses teacher forcing — each prediction is made from the true
 * prefix — so accuracies are not compounded by earlier prediction errors.
 */

import {
  movementFeatureKey,
  movementShapeKey,
  type MovementPredictionStrategy,
  type MovementSequence,
  type TrainedMovementModel,
} from "./movement-model.js";

export type MovementEvalReport = {
  sequenceCount: number;
  predictionCount: number;
  exactMatches: number;
  gestureMatches: number;
  exactMatchAccuracy: number;
  gestureMatchAccuracy: number;
  /** Prediction counts bucketed by the backoff strategy the model used. */
  byStrategy: Record<MovementPredictionStrategy, number>;
};

export function evaluateMovementGeneralization(
  model: TrainedMovementModel,
  heldOut: MovementSequence[],
): MovementEvalReport {
  const byStrategy: Record<MovementPredictionStrategy, number> = {
    exact: 0,
    shape: 0,
    unigram: 0,
    empty: 0,
  };
  let predictionCount = 0;
  let exactMatches = 0;
  let gestureMatches = 0;

  for (const sequence of heldOut) {
    for (let i = 1; i < sequence.features.length; i += 1) {
      const actual = sequence.features[i]!;
      const context = sequence.features.slice(0, i);
      const prediction = model.predictNext(context);
      predictionCount += 1;
      byStrategy[prediction.strategy] += 1;
      if (movementFeatureKey(prediction.feature) === movementFeatureKey(actual)) {
        exactMatches += 1;
      }
      if (movementShapeKey(prediction.feature) === movementShapeKey(actual)) {
        gestureMatches += 1;
      }
    }
  }

  return {
    sequenceCount: heldOut.length,
    predictionCount,
    exactMatches,
    gestureMatches,
    exactMatchAccuracy: predictionCount === 0 ? 0 : exactMatches / predictionCount,
    gestureMatchAccuracy: predictionCount === 0 ? 0 : gestureMatches / predictionCount,
    byStrategy,
  };
}
