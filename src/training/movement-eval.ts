// Generalization eval harness for the local-movement model.
//
// Measures two things on held-out sequences:
//   1. Next-token accuracy (teacher-forced): given the true prefix, how often
//      does the model's top prediction match the actual next movement?
//   2. Replay fidelity: rolled out from just the first movement, how closely
//      does the generated sequence reproduce the real one (token overlap, and
//      exact-match rate)?
// It also reports the average backoff order used, which indicates how much the
// model is relying on memorised long contexts vs. generalising from short ones.

import { movementToken, type MovementEvent, type MovementModel, type MovementSequence } from "./movement-model.js";

export type MovementEvalResult = {
  sequenceCount: number;
  predictedSteps: number;
  correctNextTokens: number;
  nextTokenAccuracy: number;
  exactReplayMatches: number;
  replayFidelity: number;
  averageTokenOverlap: number;
  averageBackoffOrder: number;
};

export type MovementEvalOptions = {
  /** Cap generation length relative to the reference sequence. */
  maxStepFactor?: number;
};

function tokensOf(events: MovementEvent[]): string[] {
  return events.map((event) => movementToken(event));
}

/** Ordered longest-common-subsequence length between two token lists. */
function orderedOverlap(a: string[], b: string[]): number {
  const dp: number[] = new Array(b.length + 1).fill(0);
  for (let i = 1; i <= a.length; i += 1) {
    let prevDiagonal = 0;
    for (let j = 1; j <= b.length; j += 1) {
      const current = dp[j]!;
      dp[j] = a[i - 1] === b[j - 1] ? prevDiagonal + 1 : Math.max(dp[j]!, dp[j - 1]!);
      prevDiagonal = current;
    }
  }
  return dp[b.length]!;
}

export function evaluateMovementModel(
  model: MovementModel,
  sequences: MovementSequence[],
  options: MovementEvalOptions = {},
): MovementEvalResult {
  let predictedSteps = 0;
  let correctNextTokens = 0;
  let exactReplayMatches = 0;
  let overlapRatioSum = 0;
  let replayableSequences = 0;
  let backoffOrderSum = 0;
  let backoffSamples = 0;

  for (const sequence of sequences) {
    const events = sequence.events;
    if (events.length === 0) {
      continue;
    }

    // Teacher-forced next-token accuracy.
    for (let i = 1; i < events.length; i += 1) {
      const prediction = model.predictNext(events.slice(0, i));
      predictedSteps += 1;
      backoffSamples += 1;
      if (prediction) {
        backoffOrderSum += prediction.contextOrderUsed;
        if (prediction.token === movementToken(events[i]!)) {
          correctNextTokens += 1;
        }
      }
    }

    // Free-running replay fidelity from the first movement only.
    const reference = tokensOf(events.slice(1));
    if (reference.length > 0) {
      replayableSequences += 1;
      const maxSteps = Math.ceil(reference.length * (options.maxStepFactor ?? 2));
      const generated = tokensOf(model.generate([events[0]!], { maxSteps }));
      const overlap = orderedOverlap(generated, reference);
      overlapRatioSum += overlap / reference.length;
      if (generated.length === reference.length && overlap === reference.length) {
        exactReplayMatches += 1;
      }
    }
  }

  return {
    sequenceCount: sequences.length,
    predictedSteps,
    correctNextTokens,
    nextTokenAccuracy: predictedSteps > 0 ? correctNextTokens / predictedSteps : 0,
    exactReplayMatches,
    replayFidelity: replayableSequences > 0 ? exactReplayMatches / replayableSequences : 0,
    averageTokenOverlap: replayableSequences > 0 ? overlapRatioSum / replayableSequences : 0,
    averageBackoffOrder: backoffSamples > 0 ? backoffOrderSum / backoffSamples : 0,
  };
}
