import { MOVEMENT_END, type MovementModel, type MovementSequence } from "./movement-model.js";

/**
 * Generalization eval harness for the movement-learning subsystem (objective #2,
 * part d). Measures how well a trained {@link MovementModel} both *repeats*
 * recorded movements (replay fidelity) and *predicts* next actions on held-out
 * trajectories (next-token accuracy) — the two behaviours the objective requires.
 */

export type MovementEvalOptions = {
  /** Rank considered a "hit" for top-K accuracy (default 3). */
  topK?: number;
  /** How many leading tokens to seed replay rollouts with (default 1). */
  replaySeedLength?: number;
};

export type SequenceEvalResult = {
  length: number;
  correctNextTokens: number;
  predictions: number;
  topKHits: number;
  /** Whether a greedy rollout from the seed exactly reproduces the sequence. */
  exactReplay: boolean;
  /** Fraction of positions where the rollout token matches the recorded token. */
  replayOverlap: number;
};

export type MovementEvalMetrics = {
  sequenceCount: number;
  predictionCount: number;
  /** Fraction of positions where the top-1 prediction matched the actual token. */
  nextTokenAccuracy: number;
  /** Fraction of positions where the actual token was within the top-K ranks. */
  topKAccuracy: number;
  topK: number;
  /** Fraction of sequences reproduced exactly by a greedy rollout from the seed. */
  exactReplayRate: number;
  /** Mean per-sequence token overlap between rollout and recorded sequence. */
  meanReplayOverlap: number;
  perSequence: SequenceEvalResult[];
};

/** Evaluate a trained model against a set of (held-out) movement sequences. */
export function evaluateMovementModel(
  model: MovementModel,
  sequences: MovementSequence[],
  options: MovementEvalOptions = {},
): MovementEvalMetrics {
  const topK = Math.max(1, Math.floor(options.topK ?? 3));
  const seedLength = Math.max(1, Math.floor(options.replaySeedLength ?? 1));

  const perSequence: SequenceEvalResult[] = [];
  let totalPredictions = 0;
  let totalCorrect = 0;
  let totalTopKHits = 0;
  let exactReplays = 0;
  let overlapSum = 0;

  for (const sequence of sequences) {
    let correctNextTokens = 0;
    let topKHits = 0;
    let predictions = 0;

    // Next-token prediction over every position, plus the END transition.
    const target = [...sequence, MOVEMENT_END];
    for (let i = 1; i < target.length; i += 1) {
      const context = sequence.slice(0, i);
      const actual = target[i]!;
      const ranked = model.predict(context, { topK });
      predictions += 1;
      if (ranked[0]?.token === actual) {
        correctNextTokens += 1;
      }
      if (ranked.some((prediction) => prediction.token === actual)) {
        topKHits += 1;
      }
    }

    const seed = sequence.slice(0, Math.min(seedLength, sequence.length));
    const rollout = model.generate(seed, { maxSteps: sequence.length });
    const exactReplay = arraysEqual(rollout, sequence);
    const replayOverlap = sequence.length === 0 ? 1 : tokenOverlap(rollout, sequence);

    perSequence.push({
      length: sequence.length,
      correctNextTokens,
      predictions,
      topKHits,
      exactReplay,
      replayOverlap,
    });

    totalPredictions += predictions;
    totalCorrect += correctNextTokens;
    totalTopKHits += topKHits;
    if (exactReplay) {
      exactReplays += 1;
    }
    overlapSum += replayOverlap;
  }

  const sequenceCount = sequences.length;
  return {
    sequenceCount,
    predictionCount: totalPredictions,
    nextTokenAccuracy: totalPredictions === 0 ? 0 : totalCorrect / totalPredictions,
    topKAccuracy: totalPredictions === 0 ? 0 : totalTopKHits / totalPredictions,
    topK,
    exactReplayRate: sequenceCount === 0 ? 0 : exactReplays / sequenceCount,
    meanReplayOverlap: sequenceCount === 0 ? 0 : overlapSum / sequenceCount,
    perSequence,
  };
}

function arraysEqual(a: MovementSequence, b: MovementSequence): boolean {
  if (a.length !== b.length) {
    return false;
  }
  return a.every((token, index) => token === b[index]);
}

function tokenOverlap(rollout: MovementSequence, expected: MovementSequence): number {
  if (expected.length === 0) {
    return 1;
  }
  let matches = 0;
  for (let i = 0; i < expected.length; i += 1) {
    if (rollout[i] === expected[i]) {
      matches += 1;
    }
  }
  return matches / expected.length;
}
