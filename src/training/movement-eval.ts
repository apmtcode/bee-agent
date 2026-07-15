import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  extractMovementExamples,
  type MovementTrainingConfig,
  type TrainedMovementModel,
} from "./movement-model.js";

/**
 * Generalization eval harness for movement models.
 *
 * Replaying recorded movements is easy to fake by memorization; the real test
 * of the objective's part (d) — "generalize to new but related movements" — is
 * whether the model predicts the correct next action on *held-out* trajectories
 * whose context suffixes it never saw verbatim. This harness reports overall
 * top-1 accuracy and, crucially, splits correct predictions into `exact` (long
 * context memorized) vs `backoff` (generalized from a shorter shared suffix).
 */

export type MovementEvalCase = {
  context: string[];
  expectedTool: string;
  predictedTool?: string;
  correct: boolean;
  matchedOrder: number;
  backoff: boolean;
  confidence: number;
};

export type MovementEvalResult = {
  /** Number of (context -> action) decisions evaluated. */
  total: number;
  /** Correct top-1 predictions. */
  correct: number;
  /** correct / total (0 when total is 0). */
  accuracy: number;
  /** Correct predictions made from a fully-matched (non-backoff) context. */
  exactCorrect: number;
  /** Correct predictions that required backoff — i.e. genuine generalization. */
  backoffCorrect: number;
  /** Mean confidence across all evaluated decisions. */
  averageConfidence: number;
  cases: MovementEvalCase[];
};

/**
 * Evaluate a trained model against held-out trajectory spans.
 *
 * The held-out spans are flattened into the same (context -> next action) pairs
 * the model trained on, then each is scored by whether the model's top-1 tool
 * matches the recorded one.
 */
export function evaluateMovementModel(
  model: TrainedMovementModel,
  heldOutSpans: TrajectorySpan[],
  config: MovementTrainingConfig = {},
): MovementEvalResult {
  const examples = extractMovementExamples(heldOutSpans, config);
  const cases: MovementEvalCase[] = [];
  let correct = 0;
  let exactCorrect = 0;
  let backoffCorrect = 0;
  let confidenceSum = 0;

  for (const example of examples) {
    const prediction = model.predict(example.context);
    const predictedTool = prediction?.action.tool;
    const isCorrect = predictedTool === example.action.tool;
    const backoff = prediction?.backoff ?? false;
    const confidence = prediction?.confidence ?? 0;
    confidenceSum += confidence;
    if (isCorrect) {
      correct += 1;
      if (backoff) {
        backoffCorrect += 1;
      } else {
        exactCorrect += 1;
      }
    }
    cases.push({
      context: example.context,
      expectedTool: example.action.tool,
      predictedTool,
      correct: isCorrect,
      matchedOrder: prediction?.matchedOrder ?? -1,
      backoff,
      confidence,
    });
  }

  const total = examples.length;
  return {
    total,
    correct,
    accuracy: total === 0 ? 0 : correct / total,
    exactCorrect,
    backoffCorrect,
    averageConfidence: total === 0 ? 0 : confidenceSum / total,
    cases,
  };
}
