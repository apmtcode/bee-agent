import { MOVEMENT_START_TOKEN, tokenizeSequence, type MovementSequence } from "./event.js";
import type {
  MovementModelArtifact,
  MovementModelBackend,
} from "./backend.js";

/**
 * Generalization eval harness for the movement subsystem.
 *
 * Two complementary measures:
 *  - {@link evaluateNextEventAccuracy}: for every prefix of a (typically
 *    held-out) sequence, ask the model for the next event and score whether the
 *    real next token is the top-1 (repeat/generalize) or within top-k. This is
 *    the core "does it generalize to new-but-related movements" metric.
 *  - {@link evaluateReplayFidelity}: roll the model forward from a seed prefix
 *    and measure how much of the recorded sequence it reproduces — the "repeat
 *    the recorded movements" metric.
 */

export type NextEventAccuracyReport = {
  sequenceCount: number;
  predictionCount: number;
  /** Fraction where the true next token was the model's top-1. */
  top1Accuracy: number;
  /** Fraction where the true next token was within the top-k. */
  topKAccuracy: number;
  k: number;
  /** Mean n-gram order used after backoff (higher = more specific context hit). */
  meanBackoffOrder: number;
};

export async function evaluateNextEventAccuracy(
  backend: MovementModelBackend,
  model: MovementModelArtifact,
  sequences: MovementSequence[],
  options: { k?: number } = {},
): Promise<NextEventAccuracyReport> {
  const k = options.k ?? 3;
  let predictionCount = 0;
  let top1 = 0;
  let topK = 0;
  let backoffSum = 0;

  for (const sequence of sequences) {
    const tokens = tokenizeSequence(sequence, model.tokenize);
    for (let i = 0; i < sequence.events.length; i += 1) {
      const trueToken = tokens[i]!;
      const prediction = await backend.predict(model, { history: sequence.events.slice(0, i) });
      predictionCount += 1;
      backoffSum += prediction.backoffOrder;
      if (prediction.token === trueToken) {
        top1 += 1;
      }
      if (prediction.distribution.slice(0, k).some((entry) => entry.token === trueToken)) {
        topK += 1;
      }
    }
  }

  return {
    sequenceCount: sequences.length,
    predictionCount,
    top1Accuracy: predictionCount === 0 ? 0 : top1 / predictionCount,
    topKAccuracy: predictionCount === 0 ? 0 : topK / predictionCount,
    k,
    meanBackoffOrder: predictionCount === 0 ? 0 : backoffSum / predictionCount,
  };
}

export type ReplayFidelityReport = {
  /** Tokens the rollout reproduced in order, from the seed onward. */
  matchedTokens: number;
  expectedTokens: number;
  /** matchedTokens / expectedTokens. 1.0 = a perfect replay. */
  fidelity: number;
  /** Index of the first divergence (=== expectedTokens when fully faithful). */
  firstDivergenceIndex: number;
};

/**
 * Seed the model with the first `seedLength` events of a recorded sequence and
 * roll it forward, comparing the generated token path to the recorded one. A
 * fidelity of 1.0 means the model repeats the recorded movements exactly.
 */
export async function evaluateReplayFidelity(
  backend: MovementModelBackend,
  model: MovementModelArtifact,
  sequence: MovementSequence,
  options: { seedLength?: number } = {},
): Promise<ReplayFidelityReport> {
  const seedLength = Math.max(0, options.seedLength ?? 1);
  const trueTokens = tokenizeSequence(sequence, model.tokenize);
  const expectedTail = trueTokens.slice(seedLength);

  const generated = await backend.generate(
    model,
    { history: sequence.events.slice(0, seedLength) },
    { maxSteps: expectedTail.length },
  );
  const generatedTokens = tokenizeSequence(
    { id: sequence.id, events: generated },
    model.tokenize,
  );

  let matched = 0;
  let firstDivergence = expectedTail.length;
  for (let i = 0; i < expectedTail.length; i += 1) {
    if (generatedTokens[i] === expectedTail[i]) {
      matched += 1;
    } else {
      firstDivergence = i;
      break;
    }
  }

  return {
    matchedTokens: matched,
    expectedTokens: expectedTail.length,
    fidelity: expectedTail.length === 0 ? 1 : matched / expectedTail.length,
    firstDivergenceIndex: firstDivergence,
  };
}

/**
 * Split a dataset's sequences into train/test partitions deterministically
 * (by index parity or an explicit ratio) so eval never leaks training data.
 */
export function splitSequences(
  sequences: MovementSequence[],
  trainRatio = 0.7,
): { train: MovementSequence[]; test: MovementSequence[] } {
  const cutoff = Math.max(1, Math.floor(sequences.length * trainRatio));
  return {
    train: sequences.slice(0, cutoff),
    test: sequences.slice(cutoff),
  };
}

/**
 * Convenience: the maximum n-gram order actually representable given a model,
 * used by tests/telemetry to sanity-check training config. Exposed here to keep
 * the START-token constant reachable through the eval surface.
 */
export function maxRepresentableOrder(model: MovementModelArtifact): number {
  return model.order;
}

export { MOVEMENT_START_TOKEN };
