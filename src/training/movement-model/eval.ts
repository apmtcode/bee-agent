import type { TrajectorySpan } from "../../capture/trajectory.js";
import { buildMovementDataset, type BuildMovementDatasetOptions } from "./dataset.js";
import type { TrainedMovementModel } from "./backend.js";

export type MovementEvalResult = {
  /** Number of predictions scored. */
  total: number;
  /** Times the top-1 prediction matched the recorded movement. */
  correct: number;
  /** Times the recorded movement appeared in the top-k ranked candidates. */
  topKCorrect: number;
  /** Top-1 accuracy in [0, 1]. */
  accuracy: number;
  /** Top-k accuracy in [0, 1]. */
  topKAccuracy: number;
  /**
   * Fraction of predictions that required backing off to a shorter context than
   * requested — a direct measure of how much the model had to generalise rather
   * than reproduce a verbatim prefix.
   */
  generalizationRate: number;
};

export type EvaluateMovementModelOptions = BuildMovementDatasetOptions & {
  /** k for the top-k accuracy metric. Defaults to 3. */
  topK?: number;
};

/**
 * Score a trained model against held-out trajectories. Reproduction fidelity is
 * top-1 accuracy on trajectories the model trained on; generalization is the
 * same metric on held-out-but-related trajectories, where `generalizationRate`
 * shows how often the model had to fall back to a shorter context to answer.
 */
export function evaluateMovementModel(
  model: TrainedMovementModel,
  heldOut: TrajectorySpan[],
  options: EvaluateMovementModelOptions = {},
): MovementEvalResult {
  const topK = Math.max(1, options.topK ?? 3);
  const dataset = buildMovementDataset(heldOut, options);

  let total = 0;
  let correct = 0;
  let topKCorrect = 0;
  let backedOff = 0;

  for (const sample of dataset.samples) {
    const prediction = model.predict({ history: sample.context, appContext: sample.appContext });
    if (!prediction) {
      total += 1;
      continue;
    }
    total += 1;
    if (prediction.action === sample.action) {
      correct += 1;
    }
    if (prediction.ranked.slice(0, topK).some((candidate) => candidate.action === sample.action)) {
      topKCorrect += 1;
    }
    if (prediction.backoffOrder < Math.min(dataset.order, sample.context.length)) {
      backedOff += 1;
    }
  }

  return {
    total,
    correct,
    topKCorrect,
    accuracy: total > 0 ? correct / total : 0,
    topKAccuracy: total > 0 ? topKCorrect / total : 0,
    generalizationRate: total > 0 ? backedOff / total : 0,
  };
}
