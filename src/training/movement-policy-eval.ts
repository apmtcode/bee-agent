import {
  MOVEMENT_END_TOKEN,
  type MovementExample,
  type MovementPolicyModel,
  type MovementToken,
} from "./movement-policy.js";

/**
 * Generalization eval harness for movement policies (roadmap item).
 *
 * Measures how well a trained policy reproduces *held-out* movement sequences —
 * sequences drawn from the same generating process but never seen during
 * training. This is the on-device-learning success metric: the agent should
 * both repeat recorded movements (exact replay) and generalize to new-but-
 * related ones (next-step accuracy via context backoff).
 */
export type MovementPolicyEvalResult = {
  /** Held-out trajectories scored. */
  trajectoryCount: number;
  /** Total next-token predictions made across all trajectories. */
  predictionCount: number;
  /** Fraction of next-token predictions whose top-1 candidate was correct. */
  top1Accuracy: number;
  /** Fraction correct when the true token is anywhere in the top-`k`. */
  topKAccuracy: number;
  k: number;
  /** Fraction of held-out trajectories regenerated exactly from their seed. */
  exactReplayRate: number;
  /**
   * Of the *correct* top-1 predictions, the fraction that came from a backed-off
   * (shorter-than-full) context — i.e. predictions the model got right by
   * generalizing rather than by exact full-context recall.
   */
  generalizationRate: number;
  /** Mean context order matched across all predictions. */
  meanMatchedOrder: number;
};

export type MovementPolicyEvalOptions = {
  /** Top-`k` cutoff for `topKAccuracy`. Default 3. */
  k?: number;
};

/** Score a trained model against held-out movement examples. */
export function evaluateMovementPolicy(
  model: MovementPolicyModel,
  heldOut: MovementExample[],
  options?: MovementPolicyEvalOptions,
): MovementPolicyEvalResult {
  const k = Math.max(1, Math.trunc(options?.k ?? 3));
  let predictionCount = 0;
  let top1Correct = 0;
  let topKCorrect = 0;
  let generalizedCorrect = 0;
  let orderSum = 0;
  let exactReplays = 0;
  let scoredTrajectories = 0;

  for (const example of heldOut) {
    if (example.tokens.length === 0) {
      continue;
    }
    scoredTrajectories += 1;

    // Next-token prediction over each position, conditioning on the true prefix.
    const target = [...example.tokens, MOVEMENT_END_TOKEN];
    for (let i = 1; i < target.length; i += 1) {
      const context = target.slice(0, i);
      const fullOrder = Math.min(model.maxOrder, context.length);
      const ranked = model.predictTopK(context, k);
      const best = ranked[0];
      const truth = target[i];
      predictionCount += 1;
      orderSum += best?.order ?? 0;
      if (best?.token === truth) {
        top1Correct += 1;
        if ((best?.order ?? 0) < fullOrder) {
          generalizedCorrect += 1;
        }
      }
      if (ranked.some((candidate) => candidate.token === truth)) {
        topKCorrect += 1;
      }
    }

    // Exact-replay: regenerate from the first movement and compare.
    const seed = example.tokens.slice(0, 1);
    const regenerated = model.generate(seed, example.tokens.length + 4);
    if (sequencesEqual(regenerated, example.tokens)) {
      exactReplays += 1;
    }
  }

  return {
    trajectoryCount: scoredTrajectories,
    predictionCount,
    top1Accuracy: ratio(top1Correct, predictionCount),
    topKAccuracy: ratio(topKCorrect, predictionCount),
    k,
    exactReplayRate: ratio(exactReplays, scoredTrajectories),
    generalizationRate: ratio(generalizedCorrect, top1Correct),
    meanMatchedOrder: ratio(orderSum, predictionCount),
  };
}

function ratio(numerator: number, denominator: number): number {
  return denominator === 0 ? 0 : numerator / denominator;
}

function sequencesEqual(a: MovementToken[], b: MovementToken[]): boolean {
  return a.length === b.length && a.every((token, index) => token === b[index]);
}

/**
 * Deterministic, seeded synthetic movement-stream generator (roadmap item).
 *
 * Validates the capture->dataset->train->infer loop without real OS input. It
 * draws sequences from a small fixed Markov structure over reusable "movement
 * primitives", so trajectories share learnable structure: a model trained on
 * one split should predict a disjoint split well. Fully deterministic given a
 * seed (no `Math.random`), so eval results are reproducible.
 */
export type SyntheticMovementOptions = {
  seed: number;
  trajectoryCount: number;
  /** Distinct movement primitives (tool::summary tokens) to compose. */
  vocabulary?: MovementToken[];
  minLength?: number;
  maxLength?: number;
  /** Probability of following the "expected" next primitive vs. a detour. */
  coherence?: number;
};

const DEFAULT_SYNTHETIC_VOCABULARY: MovementToken[] = [
  "focus::window main",
  "move::pointer toolbar",
  "click::button new",
  "type::field title",
  "press::key tab",
  "click::button save",
  "wait::status saved",
];

/** Generate a deterministic synthetic movement dataset. */
export function generateSyntheticMovementExamples(options: SyntheticMovementOptions): MovementExample[] {
  const vocabulary = options.vocabulary && options.vocabulary.length > 0 ? options.vocabulary : DEFAULT_SYNTHETIC_VOCABULARY;
  const minLength = Math.max(1, Math.trunc(options.minLength ?? 4));
  const maxLength = Math.max(minLength, Math.trunc(options.maxLength ?? Math.max(minLength, vocabulary.length)));
  const coherence = clamp01(options.coherence ?? 0.85);
  const random = mulberry32(options.seed >>> 0);
  const examples: MovementExample[] = [];

  for (let index = 0; index < Math.max(0, Math.trunc(options.trajectoryCount)); index += 1) {
    const length = minLength + Math.floor(random() * (maxLength - minLength + 1));
    let cursor = Math.floor(random() * vocabulary.length);
    const tokens: MovementToken[] = [];
    for (let step = 0; step < length; step += 1) {
      tokens.push(vocabulary[cursor]!);
      // With `coherence` probability advance along the canonical chain; else detour.
      if (random() < coherence) {
        cursor = (cursor + 1) % vocabulary.length;
      } else {
        cursor = Math.floor(random() * vocabulary.length);
      }
    }
    examples.push({ trajectoryId: `synthetic-${options.seed}-${index}`, tokens });
  }

  return examples;
}

function clamp01(value: number): number {
  if (Number.isNaN(value)) {
    return 0;
  }
  return Math.min(1, Math.max(0, value));
}

/** Small, fast, deterministic PRNG (mulberry32) — no global RNG state. */
function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
