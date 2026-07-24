/**
 * Generalization eval harness for the movement-learning subsystem.
 *
 * Objective #2(d) requires the trained model to *generalize* to new-but-related
 * movements, not just memorize the training set. This harness measures that on
 * held-out movement samples with two complementary metrics:
 *
 *   - **nextTokenAccuracy** — teacher-forced top-1 accuracy: at every position
 *     the true prefix is fed in and we check whether the model's single most
 *     likely next token matches the recorded one. Measures per-step fidelity.
 *   - **replayFidelity** — free-running rollout: the model is seeded with a short
 *     prefix and asked to regenerate the rest of the sequence on its own; we
 *     score how much of the recorded continuation it reproduces. Measures whether
 *     it can actually *repeat/complete a movement* unaided.
 *
 * The **backoffProfile** reports how much context the model matched on average —
 * a high share of full-order matches means memorization, a healthy share of
 * shorter-context matches on held-out data means genuine generalization.
 *
 * Everything here is pure and deterministic, so it runs in the cloud/CI.
 */

import type { MovementSample, TrainedMovementModel } from "./model-backend.js";

export type MovementEvalOptions = {
  /**
   * How many leading tokens to seed the free-running rollout with. Defaults to
   * the model's order (the minimum context it can condition on). Clamped to
   * `sequenceLength - 1` per sample.
   */
  seedLength?: number;
};

export type SampleEvalResult = {
  sourceSessionId?: string;
  tokenCount: number;
  /** Teacher-forced correct next-token predictions over this sample. */
  correctNextTokens: number;
  evaluatedTransitions: number;
  /** Free-running rollout matched tokens vs. the recorded continuation. */
  rolloutMatched: number;
  rolloutLength: number;
};

export type BackoffProfile = {
  /** contextOrder used → number of predictions made at that backoff depth. */
  byContextOrder: Record<number, number>;
  /** Fraction of teacher-forced predictions that matched the full model order. */
  fullOrderShare: number;
  /** Fraction that required backing off below the full order (generalization). */
  backoffShare: number;
  /** Fraction where the model had no context match at all (unigram prior). */
  unigramShare: number;
};

export type MovementEvalReport = {
  backendId: string;
  order: number;
  sampleCount: number;
  evaluatedTransitions: number;
  /** Top-1 teacher-forced next-token accuracy across all evaluated transitions. */
  nextTokenAccuracy: number;
  /** Average per-sample free-running rollout fidelity (matched / recorded). */
  replayFidelity: number;
  backoffProfile: BackoffProfile;
  perSample: SampleEvalResult[];
};

/** Evaluate a trained movement model on held-out samples. */
export function evaluateMovementModel(
  model: TrainedMovementModel,
  heldOut: MovementSample[],
  options: MovementEvalOptions = {},
): MovementEvalReport {
  const perSample: SampleEvalResult[] = [];
  const byContextOrder: Record<number, number> = {};

  let totalCorrect = 0;
  let totalTransitions = 0;
  let fidelitySum = 0;
  let scoredSamples = 0;

  for (const sample of heldOut) {
    const tokens = sample.tokens;
    let correctNextTokens = 0;
    let evaluatedTransitions = 0;

    // Teacher-forced next-token accuracy.
    for (let i = 1; i < tokens.length; i += 1) {
      const prefix = tokens.slice(0, i);
      const prediction = model.predictNext(prefix);
      evaluatedTransitions += 1;
      if (prediction) {
        byContextOrder[prediction.contextOrder] = (byContextOrder[prediction.contextOrder] ?? 0) + 1;
        if (prediction.token === tokens[i]) {
          correctNextTokens += 1;
        }
      } else {
        // No prediction still counts as an evaluated (missed) transition.
        byContextOrder[-1] = (byContextOrder[-1] ?? 0) + 1;
      }
    }

    // Free-running rollout fidelity.
    const seedLength = clampSeedLength(options.seedLength ?? model.order, tokens.length);
    let rolloutMatched = 0;
    let rolloutLength = 0;
    if (tokens.length > seedLength) {
      const seed = tokens.slice(0, seedLength);
      const expected = tokens.slice(seedLength);
      rolloutLength = expected.length;
      const generated = model.generate(seed, { maxSteps: expected.length });
      for (let i = 0; i < expected.length; i += 1) {
        if (generated[i] === expected[i]) {
          rolloutMatched += 1;
        }
      }
      fidelitySum += rolloutLength > 0 ? rolloutMatched / rolloutLength : 0;
      scoredSamples += 1;
    }

    totalCorrect += correctNextTokens;
    totalTransitions += evaluatedTransitions;

    perSample.push({
      ...(sample.sourceSessionId ? { sourceSessionId: sample.sourceSessionId } : {}),
      tokenCount: tokens.length,
      correctNextTokens,
      evaluatedTransitions,
      rolloutMatched,
      rolloutLength,
    });
  }

  return {
    backendId: model.backendId,
    order: model.order,
    sampleCount: heldOut.length,
    evaluatedTransitions: totalTransitions,
    nextTokenAccuracy: totalTransitions > 0 ? totalCorrect / totalTransitions : 0,
    replayFidelity: scoredSamples > 0 ? fidelitySum / scoredSamples : 0,
    backoffProfile: buildBackoffProfile(byContextOrder, model.order),
    perSample,
  };
}

function clampSeedLength(requested: number, sequenceLength: number): number {
  const max = Math.max(0, sequenceLength - 1);
  if (requested < 1) {
    return Math.min(1, max);
  }
  return Math.min(requested, max);
}

function buildBackoffProfile(byContextOrder: Record<number, number>, order: number): BackoffProfile {
  let total = 0;
  let fullOrder = 0;
  let unigram = 0;
  for (const [key, count] of Object.entries(byContextOrder)) {
    const contextOrder = Number(key);
    total += count;
    if (contextOrder === order) {
      fullOrder += count;
    }
    if (contextOrder === 0) {
      unigram += count;
    }
  }
  const backoff = total - fullOrder;
  return {
    byContextOrder: { ...byContextOrder },
    fullOrderShare: total > 0 ? fullOrder / total : 0,
    backoffShare: total > 0 ? backoff / total : 0,
    unigramShare: total > 0 ? unigram / total : 0,
  };
}
