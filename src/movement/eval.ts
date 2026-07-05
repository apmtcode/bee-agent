import { sequenceTokens, type MovementDataset, type MovementSequence } from "./movement-event.js";
import type { MovementModel } from "./model-backend.js";

/**
 * Result of evaluating a trained movement model against held-out sequences. The
 * harness measures how well the policy predicts the *next* movement given the
 * true prefix — the operational question for objective 2c/2d (repeat recorded
 * movements, and generalize to related ones).
 */
export type MovementEvalReport = {
  sequenceCount: number;
  /** Total next-token predictions scored across all sequences. */
  predictionCount: number;
  /** Fraction of predictions whose top-1 token matched the actual next token. */
  top1Accuracy: number;
  /** Fraction where the actual token appeared anywhere in the ranked alternatives. */
  recall: number;
  /**
   * Distribution of the backoff order used per correct prediction. A healthy
   * generalizing model answers many held-out steps at reduced order (< train
   * order) rather than only when the full context was memorized.
   */
  backoffOrderHistogram: Record<number, number>;
  /** Mean backoff order across all scored predictions (lower ⇒ more general). */
  meanOrder: number;
  perSequence: Array<{ id: string; predictionCount: number; correct: number; accuracy: number }>;
};

export type MovementEvalOptions = {
  /**
   * When true, score a prediction as correct if the actual token is present in
   * the ranked alternatives (top-k recall) in addition to strict top-1.
   */
  includeRecall?: boolean;
};

function evaluateSequence(
  model: MovementModel,
  sequence: MovementSequence,
): { predictionCount: number; correct: number; recallHits: number; orders: number[] } {
  const tokens = sequenceTokens(sequence);
  let correct = 0;
  let recallHits = 0;
  const orders: number[] = [];

  for (let index = 0; index < tokens.length; index += 1) {
    const context = tokens.slice(0, index);
    const actual = tokens[index];
    const prediction = model.predictNext(context);
    orders.push(prediction.order);
    if (prediction.token === actual) {
      correct += 1;
    }
    if (prediction.alternatives.some((alternative) => alternative.token === actual)) {
      recallHits += 1;
    }
  }

  return { predictionCount: tokens.length, correct, recallHits, orders };
}

/**
 * Evaluate `model` against a held-out dataset. Sequences the model was NOT
 * trained on measure generalization; sequences it was trained on measure
 * fidelity/replay accuracy.
 */
export function evaluateMovementModel(
  model: MovementModel,
  heldOut: MovementDataset,
  _options?: MovementEvalOptions,
): MovementEvalReport {
  let predictionCount = 0;
  let correct = 0;
  let recallHits = 0;
  let orderSum = 0;
  const backoffOrderHistogram: Record<number, number> = {};
  const perSequence: MovementEvalReport["perSequence"] = [];

  for (const sequence of heldOut.sequences) {
    const result = evaluateSequence(model, sequence);
    predictionCount += result.predictionCount;
    correct += result.correct;
    recallHits += result.recallHits;
    for (const order of result.orders) {
      orderSum += order;
      backoffOrderHistogram[order] = (backoffOrderHistogram[order] ?? 0) + 1;
    }
    perSequence.push({
      id: sequence.id,
      predictionCount: result.predictionCount,
      correct: result.correct,
      accuracy: result.predictionCount === 0 ? 0 : result.correct / result.predictionCount,
    });
  }

  return {
    sequenceCount: heldOut.sequences.length,
    predictionCount,
    top1Accuracy: predictionCount === 0 ? 0 : correct / predictionCount,
    recall: predictionCount === 0 ? 0 : recallHits / predictionCount,
    backoffOrderHistogram,
    meanOrder: predictionCount === 0 ? 0 : orderSum / predictionCount,
    perSequence,
  };
}

/**
 * Deterministically split a dataset into train / held-out partitions by a
 * stride, so callers can measure generalization without shuffling. `holdoutEvery
 * = 4` holds out every 4th sequence (25%).
 */
export function splitMovementDataset(
  dataset: MovementDataset,
  holdoutEvery: number,
): { train: MovementDataset; holdout: MovementDataset } {
  const stride = Math.max(2, Math.floor(holdoutEvery));
  const train: MovementSequence[] = [];
  const holdout: MovementSequence[] = [];
  dataset.sequences.forEach((sequence, index) => {
    if ((index + 1) % stride === 0) {
      holdout.push(sequence);
    } else {
      train.push(sequence);
    }
  });
  return { train: { sequences: train }, holdout: { sequences: holdout } };
}
