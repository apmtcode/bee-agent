import type { MovementDataset, MovementToken } from "./movement-event.js";

/**
 * A single next-movement prediction. `order` reports how much context the model
 * actually consumed after backoff, which the eval harness uses to reason about
 * how much the prediction relied on memorization vs. generalization.
 */
export type MovementPrediction = {
  /** Predicted next token, or `null` when the model has no basis to predict. */
  token: MovementToken | null;
  /** Probability mass assigned to `token` within the consulted context. */
  probability: number;
  /** Context length (n-gram order) used to make the prediction after backoff. */
  order: number;
  /** Ranked alternatives (including `token`), most probable first. */
  alternatives: Array<{ token: MovementToken; probability: number }>;
};

export type MovementGenerateOptions = {
  /** Hard cap on generated steps (excludes the terminal END token). */
  maxSteps?: number;
};

/**
 * A trained, in-process movement policy. Deterministic given identical training
 * data + seed, so it is fully testable in CI without a real device or GPU.
 */
export interface MovementModel {
  readonly backend: string;
  /** Predict the next token following `context` (most-recent token last). */
  predictNext(context: MovementToken[]): MovementPrediction;
  /**
   * Roll out a full movement continuation from `seed`, stopping at a modeled
   * sequence boundary or `maxSteps`. Returns emitted tokens (no boundary token).
   */
  generate(seed: MovementToken[], options?: MovementGenerateOptions): MovementToken[];
  /** Serialize model parameters for persistence / later `restore`. */
  serialize(): unknown;
}

export type MovementTrainOptions = {
  /** Max n-gram order the backend may use; backends may clamp to their own max. */
  order?: number;
};

/**
 * Pluggable training backend. The real on-device path (mlx / axolotl) and the
 * in-process deterministic path both implement this seam, so callers — and the
 * generalization eval harness — are backend-agnostic.
 */
export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, options?: MovementTrainOptions): MovementModel;
  /** Rehydrate a model previously produced by this backend's `serialize()`. */
  restore(state: unknown): MovementModel;
}
