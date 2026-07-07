import type { MovementModel, MovementSequence, MovementToken } from "./movement-model.js";

/**
 * Generalization / replay-fidelity eval harness for movement models.
 *
 * Given a trained {@link MovementModel} and a set of held-out sequences, it
 * measures how well the model's greedy next-token prediction reproduces each
 * sequence step-for-step. Used to (a) prove exact replay of memorized
 * sequences and (b) quantify generalization to new-but-related sequences.
 */

export type MovementEvalStep = {
  index: number;
  context: MovementToken[];
  expected: MovementToken;
  predicted: MovementToken | undefined;
  correct: boolean;
};

export type MovementSequenceEval = {
  id: string;
  steps: MovementEvalStep[];
  correct: number;
  total: number;
  accuracy: number;
};

export type MovementEvalReport = {
  sequences: MovementSequenceEval[];
  correct: number;
  total: number;
  /** Micro-averaged next-token accuracy across every step of every sequence. */
  accuracy: number;
  /** Macro-averaged accuracy (mean of per-sequence accuracies). */
  macroAccuracy: number;
};

export type EvaluateMovementOptions = {
  /**
   * How much prior context to condition on when predicting each step.
   * Defaults to the model's order.
   */
  contextWindow?: number;
};

export function evaluateMovementModel(
  model: MovementModel,
  sequences: MovementSequence[],
  options: EvaluateMovementOptions = {},
): MovementEvalReport {
  const window = Math.max(1, Math.floor(options.contextWindow ?? model.order));
  const evaluated = sequences.map<MovementSequenceEval>((sequence) => {
    const steps: MovementEvalStep[] = [];
    for (let index = 0; index < sequence.tokens.length; index += 1) {
      const context = sequence.tokens.slice(Math.max(0, index - window), index);
      const predicted = model.predictNext(context)?.token;
      const expected = sequence.tokens[index];
      steps.push({ index, context, expected, predicted, correct: predicted === expected });
    }
    const correct = steps.filter((step) => step.correct).length;
    const total = steps.length;
    return {
      id: sequence.id,
      steps,
      correct,
      total,
      accuracy: total === 0 ? 1 : correct / total,
    };
  });

  const correct = evaluated.reduce((sum, entry) => sum + entry.correct, 0);
  const total = evaluated.reduce((sum, entry) => sum + entry.total, 0);
  const macroAccuracy =
    evaluated.length === 0 ? 1 : evaluated.reduce((sum, entry) => sum + entry.accuracy, 0) / evaluated.length;

  return {
    sequences: evaluated,
    correct,
    total,
    accuracy: total === 0 ? 1 : correct / total,
    macroAccuracy,
  };
}
