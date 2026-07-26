import {
  movementAbstractKey,
  movementTokenKey,
  type MovementModel,
  type MovementSequence,
  type MovementToken,
} from "./movement-model.js";

/**
 * Generalization eval harness for the movement model.
 *
 * Two complementary metrics:
 *  - Next-action prediction (teacher-forced): given the true prefix, how often
 *    does the model predict the correct next movement? Reported both
 *    structurally (tool+action, target-agnostic) and exactly (incl. target).
 *  - Replay fidelity (free rollout): generate a sequence from scratch and
 *    measure structural overlap with the reference via normalized LCS.
 *
 * Structural accuracy on a held-out set with novel targets is the headline
 * generalization signal: it shows the model performs the same movements against
 * targets it never saw in training.
 */

export type NextActionEvalResult = {
  totalPredictions: number;
  structuralMatches: number;
  exactMatches: number;
  structuralAccuracy: number;
  exactAccuracy: number;
  /** fraction of correct structural predictions that relied on abstraction backoff. */
  abstractionRate: number;
};

export function evaluateNextActionPrediction(
  model: MovementModel,
  testSequences: MovementSequence[],
): NextActionEvalResult {
  let total = 0;
  let structural = 0;
  let exact = 0;
  let abstractedHits = 0;

  for (const sequence of testSequences) {
    for (let i = 0; i < sequence.tokens.length; i += 1) {
      const context = sequence.tokens.slice(0, i);
      const expected = sequence.tokens[i];
      const prediction = model.predict(context);
      total += 1;
      if (prediction.token && movementAbstractKey(prediction.token) === movementAbstractKey(expected)) {
        structural += 1;
        if (prediction.abstracted) {
          abstractedHits += 1;
        }
        if (movementTokenKey(prediction.token) === movementTokenKey(expected)) {
          exact += 1;
        }
      }
    }
  }

  return {
    totalPredictions: total,
    structuralMatches: structural,
    exactMatches: exact,
    structuralAccuracy: total > 0 ? structural / total : 0,
    exactAccuracy: total > 0 ? exact / total : 0,
    abstractionRate: structural > 0 ? abstractedHits / structural : 0,
  };
}

export type ReplayFidelityResult = {
  sequenceCount: number;
  averageStructuralOverlap: number;
  perSequence: Array<{ id: string; overlap: number; generatedLength: number; referenceLength: number }>;
};

/**
 * Free-rollout replay fidelity: generate from an empty seed and compare the
 * generated structural key stream to the reference via normalized LCS.
 */
export function evaluateReplayFidelity(
  model: MovementModel,
  testSequences: MovementSequence[],
  options: { seedTokens?: number } = {},
): ReplayFidelityResult {
  const seedTokens = options.seedTokens ?? 0;
  const perSequence = testSequences.map((sequence) => {
    const seed = sequence.tokens.slice(0, seedTokens);
    const generated = model.generate({ seed, maxSteps: sequence.tokens.length + 4 });
    const overlap = normalizedLcs(
      generated.map((token) => movementAbstractKey(token)),
      sequence.tokens.map((token) => movementAbstractKey(token)),
    );
    return {
      id: sequence.id,
      overlap,
      generatedLength: generated.length,
      referenceLength: sequence.tokens.length,
    };
  });

  const averageStructuralOverlap =
    perSequence.length > 0
      ? perSequence.reduce((sum, entry) => sum + entry.overlap, 0) / perSequence.length
      : 0;

  return {
    sequenceCount: perSequence.length,
    averageStructuralOverlap,
    perSequence,
  };
}

/** Longest common subsequence length normalized by the reference length. */
function normalizedLcs(a: string[], b: string[]): number {
  if (b.length === 0) {
    return a.length === 0 ? 1 : 0;
  }
  const rows = a.length + 1;
  const cols = b.length + 1;
  const table = new Array<number>(rows * cols).fill(0);
  for (let i = 1; i < rows; i += 1) {
    for (let j = 1; j < cols; j += 1) {
      if (a[i - 1] === b[j - 1]) {
        table[i * cols + j] = table[(i - 1) * cols + (j - 1)] + 1;
      } else {
        table[i * cols + j] = Math.max(table[(i - 1) * cols + j], table[i * cols + (j - 1)]);
      }
    }
  }
  return table[a.length * cols + b.length] / b.length;
}

/** Convenience: dataset sequences keyed for quick comparison in tests/reports. */
export function structuralSignature(tokens: MovementToken[]): string {
  return tokens.map((token) => movementAbstractKey(token)).join(" -> ");
}
