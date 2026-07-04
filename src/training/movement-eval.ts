/**
 * Generalization eval harness for movement models.
 *
 * Measures how faithfully a trained {@link MovementModel} reproduces held-out
 * (unseen) movement sequences by walking each sequence and asking the model to
 * predict the next token from the true prefix — top-1 next-token accuracy plus
 * a whole-sequence greedy-replay fidelity score.
 */
import {
  MOVEMENT_END_TOKEN,
  type MovementDataset,
  type MovementModel,
  type MovementToken,
} from "./movement-model.js";

export type MovementEvalResult = {
  sequences: number;
  /** Total (prefix -> next-token) prediction points scored. */
  predictions: number;
  /** Correct greedy next-token predictions over all prefixes. */
  correct: number;
  /** correct / predictions, in [0, 1]; 1 when there is nothing to score. */
  nextTokenAccuracy: number;
  /**
   * Mean over sequences of the longest correct greedy continuation from the
   * empty seed, normalized by sequence length — how far the model replays each
   * held-out trajectory before diverging.
   */
  replayFidelity: number;
};

function scoreSequence(model: MovementModel, tokens: MovementToken[]): { correct: number; prefixMatch: number } {
  if (tokens.length === 0) {
    return { correct: 0, prefixMatch: 0 };
  }
  let correct = 0;
  let prefixMatch = 0;
  let prefixIntact = true;
  for (let index = 0; index < tokens.length; index += 1) {
    const context = tokens.slice(0, index);
    const predicted = model.predictNext(context) ?? MOVEMENT_END_TOKEN;
    const expected = tokens[index]!;
    if (predicted === expected) {
      correct += 1;
      if (prefixIntact) {
        prefixMatch += 1;
      }
    } else {
      prefixIntact = false;
    }
  }
  return { correct, prefixMatch };
}

export function evaluateMovementModel(model: MovementModel, dataset: MovementDataset): MovementEvalResult {
  let predictions = 0;
  let correct = 0;
  let fidelitySum = 0;
  let scoredSequences = 0;

  for (const sequence of dataset.sequences) {
    if (sequence.tokens.length === 0) {
      continue;
    }
    scoredSequences += 1;
    predictions += sequence.tokens.length;
    const { correct: sequenceCorrect, prefixMatch } = scoreSequence(model, sequence.tokens);
    correct += sequenceCorrect;
    fidelitySum += prefixMatch / sequence.tokens.length;
  }

  return {
    sequences: dataset.sequences.length,
    predictions,
    correct,
    nextTokenAccuracy: predictions === 0 ? 1 : correct / predictions,
    replayFidelity: scoredSequences === 0 ? 1 : fidelitySum / scoredSequences,
  };
}
