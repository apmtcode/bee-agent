import type { DeviceGestureKind } from "../capture/device-adapter.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Local-movement learning model (standing objective 2d).
 *
 * This module implements the "post-train a local model on the recorded dataset
 * to repeat the recorded movements, and generalize to new but related movements"
 * piece of the movement-learning subsystem. It is deliberately dependency-free
 * and deterministic so it can be trained and evaluated in the cloud/CI on
 * synthetic event streams; the {@link MovementModelBackend} seam lets a real
 * on-device model (MLX / axolotl / llama.cpp, etc.) be swapped in later without
 * touching call sites.
 */

/** A gesture plus the two sentinels that bound a rollout. */
export type MovementGesture = DeviceGestureKind | "start" | "end";

/** One recorded (or predicted) low-level movement. */
export type MovementStep = {
  gesture: MovementGesture;
  target?: string;
  direction?: "up" | "down" | "left" | "right";
  valueSummary?: string;
};

/** A goal-labelled ordered list of movements — one training/eval example. */
export type MovementSequence = {
  id: string;
  goal?: string;
  steps: MovementStep[];
};

/** Canonical serialized form of a movement step used as a model token. */
export type MovementToken = string;

export const MOVEMENT_START_TOKEN: MovementToken = "start";
export const MOVEMENT_END_TOKEN: MovementToken = "end";

const FIELD_SEP = "\u0001";
const CONTEXT_SEP = "\u0002";

function normalizeField(value: string | undefined): string {
  return (value ?? "").trim().toLowerCase();
}

/** Encode a step to a stable, decodable token. */
export function encodeMovementStep(step: MovementStep): MovementToken {
  if (step.gesture === "start") return MOVEMENT_START_TOKEN;
  if (step.gesture === "end") return MOVEMENT_END_TOKEN;
  return [
    step.gesture,
    normalizeField(step.target),
    normalizeField(step.direction),
    normalizeField(step.valueSummary),
  ].join(FIELD_SEP);
}

/** Inverse of {@link encodeMovementStep}. */
export function decodeMovementToken(token: MovementToken): MovementStep {
  if (token === MOVEMENT_START_TOKEN) return { gesture: "start" };
  if (token === MOVEMENT_END_TOKEN) return { gesture: "end" };
  const [gesture, target, direction, valueSummary] = token.split(FIELD_SEP);
  const step: MovementStep = { gesture: gesture as MovementGesture };
  if (target) step.target = target;
  if (direction) step.direction = direction as MovementStep["direction"];
  if (valueSummary) step.valueSummary = valueSummary;
  return step;
}

/** Wrap a sequence's steps in start/end sentinels and encode to tokens. */
export function tokenizeSequence(sequence: MovementSequence): MovementToken[] {
  return [
    MOVEMENT_START_TOKEN,
    ...sequence.steps.map(encodeMovementStep),
    MOVEMENT_END_TOKEN,
  ];
}

export type MovementPrediction = { token: MovementToken; probability: number };

export type MovementTrainOptions = {
  /** Maximum context length the model conditions on. Defaults to 2. */
  order?: number;
};

/**
 * Serializable trained weights. Backend-tagged so a loader can pick the right
 * backend; JSON-safe so a job store can persist them next to a manifest.
 */
export type MovementModelWeights = {
  backend: string;
  order: number;
  /** levels[k] maps a k-token context key -> { nextToken: count }. */
  levels: Array<Record<string, Record<string, number>>>;
  /** shapeLevels[k] maps a k-token *gesture-shape* context key -> { nextToken:
   * count }. Consulted when the exact-token context is unseen, so a novel slot
   * value in the context (e.g. a never-seen app icon) still matches the learned
   * movement shape instead of collapsing to the unigram distribution. */
  shapeLevels: Array<Record<string, Record<string, number>>>;
  vocabulary: MovementToken[];
  sequenceCount: number;
};

/** Reduce a token to its gesture "shape" (drops slot values). */
export function movementTokenShape(token: MovementToken): string {
  if (token === MOVEMENT_START_TOKEN || token === MOVEMENT_END_TOKEN) return token;
  return token.split(FIELD_SEP)[0] ?? token;
}

/**
 * Pluggable model backend. A real on-device backend implements the same three
 * methods over its own (opaque, still-JSON-serializable) weights.
 */
export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementSequence[], options?: MovementTrainOptions): MovementModelWeights;
  /** Ranked next-token distribution given the trailing context. */
  predictNext(weights: MovementModelWeights, context: MovementToken[]): MovementPrediction[];
}

function contextKey(tokens: MovementToken[]): string {
  return tokens.join(CONTEXT_SEP);
}

/**
 * Deterministic in-process backend: a back-off n-gram sequence model.
 *
 * - Exact recorded contexts reproduce recorded movements (replay fidelity).
 * - Unseen higher-order contexts fall back to the longest matching suffix seen
 *   in training (stupid-backoff), which is what yields generalization to new
 *   but related movements.
 *
 * All ranking ties break on the token string so results are fully reproducible.
 */
export class NgramMovementBackend implements MovementModelBackend {
  readonly name = "ngram";

  train(dataset: MovementSequence[], options: MovementTrainOptions = {}): MovementModelWeights {
    const order = Math.max(1, Math.floor(options.order ?? 2));
    const emptyLevels = (): Array<Record<string, Record<string, number>>> =>
      Array.from({ length: order + 1 }, () => ({}));
    const levels = emptyLevels();
    const shapeLevels = emptyLevels();
    const vocabulary = new Set<MovementToken>();

    const bump = (
      table: Record<string, Record<string, number>>,
      key: string,
      next: MovementToken,
    ): void => {
      const counts = (table[key] ??= {});
      counts[next] = (counts[next] ?? 0) + 1;
    };

    for (const sequence of dataset) {
      const tokens = tokenizeSequence(sequence);
      for (const token of tokens) vocabulary.add(token);
      for (let i = 1; i < tokens.length; i += 1) {
        const next = tokens[i]!;
        for (let k = 0; k <= order; k += 1) {
          if (i - k < 0) break;
          const window = tokens.slice(i - k, i);
          bump(levels[k]!, contextKey(window), next);
          bump(shapeLevels[k]!, contextKey(window.map(movementTokenShape)), next);
        }
      }
    }

    return {
      backend: this.name,
      order,
      levels,
      shapeLevels,
      vocabulary: [...vocabulary].sort(),
      sequenceCount: dataset.length,
    };
  }

  predictNext(weights: MovementModelWeights, context: MovementToken[]): MovementPrediction[] {
    const rank = (table: Record<string, number> | undefined): MovementPrediction[] | null => {
      if (!table) return null;
      const total = Object.values(table).reduce((sum, count) => sum + count, 0);
      if (total <= 0) return null;
      return Object.entries(table)
        .map(([token, count]) => ({ token, probability: count / total }))
        .sort((a, b) => b.probability - a.probability || (a.token < b.token ? -1 : 1));
    };

    const maxK = Math.min(weights.order, context.length);
    // 1) Exact-token backoff (k = maxK..1): highest replay fidelity.
    for (let k = maxK; k >= 1; k -= 1) {
      const ranked = rank(weights.levels[k]?.[contextKey(context.slice(context.length - k))]);
      if (ranked) return ranked;
    }
    // 2) Gesture-shape backoff (k = maxK..1): generalize past novel slot values.
    const shapeLevels = weights.shapeLevels ?? [];
    for (let k = maxK; k >= 1; k -= 1) {
      const shapeKey = contextKey(context.slice(context.length - k).map(movementTokenShape));
      const ranked = rank(shapeLevels[k]?.[shapeKey]);
      if (ranked) return ranked;
    }
    // 3) Unigram fallback.
    return rank(weights.levels[0]?.[""]) ?? [];
  }
}

export type MovementRolloutOptions = {
  /** Hard cap on generated steps (excludes the terminating end token). */
  maxSteps?: number;
  /** Seed context to condition on (defaults to the start sentinel). */
  startContext?: MovementToken[];
};

const DEFAULT_MAX_STEPS = 64;

/**
 * A trained movement policy: rolls out predicted movements and scores
 * sequences. Wraps any {@link MovementModelBackend}.
 */
export class MovementPolicyModel {
  constructor(
    private readonly backend: MovementModelBackend,
    readonly weights: MovementModelWeights,
  ) {}

  static train(
    backend: MovementModelBackend,
    dataset: MovementSequence[],
    options?: MovementTrainOptions,
  ): MovementPolicyModel {
    return new MovementPolicyModel(backend, backend.train(dataset, options));
  }

  /** Greedy (most-probable) rollout — reproduces training sequences exactly
   * when the next step is unambiguous, and generalizes via backoff otherwise. */
  generate(options: MovementRolloutOptions = {}): MovementStep[] {
    const maxSteps = options.maxSteps ?? DEFAULT_MAX_STEPS;
    const context: MovementToken[] = [...(options.startContext ?? [MOVEMENT_START_TOKEN])];
    const steps: MovementStep[] = [];
    while (steps.length < maxSteps) {
      const prediction = this.backend.predictNext(this.weights, context)[0];
      if (!prediction || prediction.token === MOVEMENT_END_TOKEN) break;
      context.push(prediction.token);
      steps.push(decodeMovementToken(prediction.token));
    }
    return steps;
  }

  /** Ranked next-token distribution given the trailing context. */
  predict(context: MovementToken[]): MovementPrediction[] {
    return this.backend.predictNext(this.weights, context);
  }

  /** Probability the model assigns to the observed next token (teacher forced). */
  nextStepProbability(context: MovementToken[], token: MovementToken): number {
    const match = this.backend.predictNext(this.weights, context).find((p) => p.token === token);
    return match?.probability ?? 0;
  }

  /** Mean per-token log-probability of a sequence (higher = better fit). */
  scoreSequence(sequence: MovementSequence): number {
    const tokens = tokenizeSequence(sequence);
    let logProb = 0;
    let count = 0;
    for (let i = 1; i < tokens.length; i += 1) {
      const prob = this.nextStepProbability(tokens.slice(0, i), tokens[i]!);
      logProb += Math.log(prob > 0 ? prob : 1e-9);
      count += 1;
    }
    return count === 0 ? 0 : logProb / count;
  }
}

/**
 * Derive a movement sequence from a recorded {@link TrajectorySpan}. Device
 * gestures captured by {@link DeviceCaptureAdapter} land in `action.metadata`,
 * so we reconstruct the low-level step stream the model trains on.
 */
export function trajectoryToMovementSequence(span: TrajectorySpan): MovementSequence {
  const steps: MovementStep[] = [];
  for (const action of [...span.actions].sort((a, b) => a.ts - b.ts)) {
    const meta = action.metadata ?? {};
    const gesture = typeof meta.gesture === "string" ? (meta.gesture as MovementGesture) : undefined;
    if (!gesture) continue;
    const step: MovementStep = { gesture };
    if (typeof meta.target === "string") step.target = meta.target;
    if (typeof meta.direction === "string") step.direction = meta.direction as MovementStep["direction"];
    if (typeof meta.valueSummary === "string") step.valueSummary = meta.valueSummary;
    steps.push(step);
  }
  return { id: span.id, goal: span.outcome?.summary, steps };
}

export function createDefaultMovementModel(
  dataset: MovementSequence[],
  options?: MovementTrainOptions,
): MovementPolicyModel {
  return MovementPolicyModel.train(new NgramMovementBackend(), dataset, options);
}
