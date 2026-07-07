import { MOVEMENT_END, type MovementModel, type MovementSequence } from "./movement-model.js";

/**
 * Generalization eval harness for the movement subsystem.
 *
 * Measures how well a trained {@link MovementModel} predicts the *next*
 * movement on held-out episodes — the concrete proxy for objective 2(d),
 * "generalize to perform new but related movements". Beyond raw accuracy it
 * isolates *generalization* accuracy: prediction quality on the subset of
 * contexts the model had never seen at full order and therefore had to back off
 * on. A model that only memorizes scores well on the train split but collapses
 * on that subset; a model that truly generalizes holds up.
 */

export type MovementEvalOptions = {
  /** How many candidates count as a top-k hit. Default 3. */
  topK?: number;
};

export type MovementEvalResult = {
  sequences: number;
  predictions: number;
  correct: number;
  accuracy: number;
  topKCorrect: number;
  topKAccuracy: number;
  /** Predictions where the model backed off below its full order (unseen context). */
  generalizedPredictions: number;
  generalizedCorrect: number;
  generalizationAccuracy: number;
  /** Per-backoff-order breakdown, keyed by the order actually used. */
  perOrder: Record<number, { predictions: number; correct: number }>;
};

/**
 * Slide across each held-out sequence predicting every next token (including
 * the terminal `<end>`), comparing against the model's argmax and top-k.
 */
export function evaluateMovementModel(
  model: MovementModel,
  heldOut: MovementSequence[],
  options: MovementEvalOptions = {},
): MovementEvalResult {
  const topK = Math.max(1, Math.floor(options.topK ?? 3));
  let predictions = 0;
  let correct = 0;
  let topKCorrect = 0;
  let generalizedPredictions = 0;
  let generalizedCorrect = 0;
  const perOrder: Record<number, { predictions: number; correct: number }> = {};

  for (const sequence of heldOut) {
    // Targets are each token plus the terminal end marker.
    const targets = [...sequence.tokens, MOVEMENT_END];
    for (let i = 0; i < targets.length; i += 1) {
      const context = sequence.tokens.slice(0, i);
      const expected = targets[i]!;
      const prediction = model.predictNext(context);
      const predictedToken = prediction.token ?? MOVEMENT_END;

      predictions += 1;
      const isCorrect = predictedToken === expected;
      const bucket = (perOrder[prediction.order] ??= { predictions: 0, correct: 0 });
      bucket.predictions += 1;

      if (isCorrect) {
        correct += 1;
        bucket.correct += 1;
      }

      const inTopK = prediction.candidates.slice(0, topK).some((candidate) => {
        const token = candidate.token === MOVEMENT_END ? MOVEMENT_END : candidate.token;
        return token === expected;
      });
      if (inTopK) {
        topKCorrect += 1;
      }

      if (prediction.order < model.order) {
        generalizedPredictions += 1;
        if (isCorrect) {
          generalizedCorrect += 1;
        }
      }
    }
  }

  return {
    sequences: heldOut.length,
    predictions,
    correct,
    accuracy: ratio(correct, predictions),
    topKCorrect,
    topKAccuracy: ratio(topKCorrect, predictions),
    generalizedPredictions,
    generalizedCorrect,
    generalizationAccuracy: ratio(generalizedCorrect, generalizedPredictions),
    perOrder,
  };
}

/**
 * Deterministic train/holdout split by sequence index (every `1/holdoutRatio`th
 * sequence is held out). No randomness, so evals are reproducible run to run.
 */
export function splitMovementDataset(
  dataset: MovementSequence[],
  holdoutRatio = 0.25,
): { train: MovementSequence[]; holdout: MovementSequence[] } {
  const ratio = Math.min(0.9, Math.max(0.05, holdoutRatio));
  const stride = Math.max(2, Math.round(1 / ratio));
  const train: MovementSequence[] = [];
  const holdout: MovementSequence[] = [];
  dataset.forEach((sequence, index) => {
    if (index % stride === stride - 1) {
      holdout.push(sequence);
    } else {
      train.push(sequence);
    }
  });
  return { train, holdout };
}

function ratio(numerator: number, denominator: number): number {
  return denominator === 0 ? 0 : numerator / denominator;
}
