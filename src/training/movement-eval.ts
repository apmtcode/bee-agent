import {
  MOVEMENT_END_TOKEN,
  tokenizeSequence,
  type MovementSequence,
  type MovementStep,
  type MovementToken,
  type TrainedMovementModel,
} from "./movement-model.js";

/**
 * Synthetic movement-stream generation + a generalization eval harness.
 *
 * Because the engine runs in the cloud with no access to real mouse/keyboard/UI
 * input, we validate the capture→dataset→train→generalize pipeline with
 * *deterministic synthetic* data. A seeded PRNG (no `Math.random`) makes every
 * generated corpus and every metric reproducible in CI.
 */

// ---------------------------------------------------------------------------
// Deterministic PRNG (mulberry32) — reproducible synthetic streams
// ---------------------------------------------------------------------------

function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function pick<T>(rng: () => number, items: readonly T[]): T {
  return items[Math.min(items.length - 1, Math.floor(rng() * items.length))];
}

// ---------------------------------------------------------------------------
// Task templates — a small library of realistic multi-step flows with slots
// ---------------------------------------------------------------------------

/** A slot fills a step's `target` from a pool, so variants stay related but differ. */
export type MovementTemplateStep = MovementStep & {
  targetPool?: readonly string[];
  optional?: boolean;
};

export type MovementTemplate = {
  name: string;
  appId: string;
  steps: MovementTemplateStep[];
};

export const DEFAULT_MOVEMENT_TEMPLATES: readonly MovementTemplate[] = [
  {
    name: "login",
    appId: "com.example.app",
    steps: [
      { gesture: "tap", targetPool: ["username-field", "email-field"] },
      { gesture: "type", targetPool: ["username-field", "email-field"] },
      { gesture: "tap", target: "password-field" },
      { gesture: "type", target: "password-field" },
      { gesture: "tap", targetPool: ["remember-me"], optional: true },
      { gesture: "tap", target: "submit-button" },
    ],
  },
  {
    name: "search",
    appId: "com.example.app",
    steps: [
      { gesture: "tap", target: "search-box" },
      { gesture: "type", target: "search-box" },
      { gesture: "tap", target: "search-submit" },
      { gesture: "scroll", direction: "down" },
      { gesture: "tap", targetPool: ["result-1", "result-2", "result-3"] },
    ],
  },
  {
    name: "compose",
    appId: "com.example.mail",
    steps: [
      { gesture: "tap", target: "compose-button" },
      { gesture: "tap", target: "to-field" },
      { gesture: "type", target: "to-field" },
      { gesture: "tap", target: "subject-field" },
      { gesture: "type", target: "subject-field" },
      { gesture: "tap", target: "body-field" },
      { gesture: "type", target: "body-field" },
      { gesture: "tap", target: "send-button" },
    ],
  },
];

export type SyntheticCorpusOptions = {
  seed?: number;
  /** Number of sequences to emit. */
  count: number;
  templates?: readonly MovementTemplate[];
};

function instantiate(template: MovementTemplate, rng: () => number, index: number): MovementSequence {
  const steps: MovementStep[] = [];
  for (const templateStep of template.steps) {
    if (templateStep.optional && rng() < 0.5) {
      continue;
    }
    const { targetPool, optional: _optional, ...base } = templateStep;
    const target = targetPool ? pick(rng, targetPool) : base.target;
    steps.push({ ...base, appId: template.appId, ...(target ? { target } : {}) });
  }
  return { id: `${template.name}-${index}`, appId: template.appId, steps };
}

/** Generate a deterministic corpus of related movement sequences. */
export function generateSyntheticCorpus(options: SyntheticCorpusOptions): MovementSequence[] {
  const templates = options.templates ?? DEFAULT_MOVEMENT_TEMPLATES;
  const rng = mulberry32(options.seed ?? 0x51ed);
  const sequences: MovementSequence[] = [];
  for (let i = 0; i < options.count; i += 1) {
    const template = templates[i % templates.length];
    sequences.push(instantiate(template, rng, i));
  }
  return sequences;
}

/** Split a corpus into train / held-out sets deterministically by ratio. */
export function splitCorpus(
  corpus: MovementSequence[],
  trainRatio = 0.7,
): { train: MovementSequence[]; heldOut: MovementSequence[] } {
  const pivot = Math.max(1, Math.floor(corpus.length * trainRatio));
  return { train: corpus.slice(0, pivot), heldOut: corpus.slice(pivot) };
}

// ---------------------------------------------------------------------------
// Generalization eval harness
// ---------------------------------------------------------------------------

export type MovementEvalReport = {
  sequences: number;
  predictions: number;
  /** Fraction of next-step predictions whose top-1 token matched the truth. */
  top1Accuracy: number;
  /** Fraction where the truth appeared anywhere in the predicted distribution. */
  recall: number;
  /** Mean top-1 confidence over all predictions. */
  meanConfidence: number;
  /** Fraction of predictions that required backoff to a shorter context (generalization rate). */
  backoffRate: number;
};

/**
 * Teacher-forced next-step evaluation over held-out sequences: at each position
 * the model predicts the next token given the true prefix, scored against the
 * held-out truth. Measures how well a model generalizes to related-but-unseen flows.
 */
export function evaluateMovementModel(
  model: TrainedMovementModel,
  heldOut: MovementSequence[],
): MovementEvalReport {
  let predictions = 0;
  let correct = 0;
  let recalled = 0;
  let confidenceSum = 0;
  let backoffCount = 0;

  for (const sequence of heldOut) {
    const tokens = [...tokenizeSequence(sequence), MOVEMENT_END_TOKEN];
    const context: MovementToken[] = [];
    for (const truth of tokens) {
      const prediction = model.predictNext(context);
      predictions += 1;
      confidenceSum += prediction.confidence;

      const predictedToken = prediction.token ?? MOVEMENT_END_TOKEN;
      if (predictedToken === truth) {
        correct += 1;
      }
      const inDistribution =
        truth === MOVEMENT_END_TOKEN
          ? prediction.token === null
          : prediction.distribution.some((entry) => entry.token === truth);
      if (inDistribution) {
        recalled += 1;
      }
      if (prediction.backoffOrder >= 0 && prediction.backoffOrder < Math.min(context.length, model.config.order)) {
        backoffCount += 1;
      }
      if (truth !== MOVEMENT_END_TOKEN) {
        context.push(truth);
      }
    }
  }

  const safe = (value: number) => (predictions === 0 ? 0 : value / predictions);
  return {
    sequences: heldOut.length,
    predictions,
    top1Accuracy: safe(correct),
    recall: safe(recalled),
    meanConfidence: safe(confidenceSum),
    backoffRate: safe(backoffCount),
  };
}
