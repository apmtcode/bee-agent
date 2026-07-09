import type { MovementActionToken, MovementDataset } from "./movement-dataset.js";

/**
 * Pluggable local movement-policy backend. Implementations post-train a small,
 * on-device model on a reviewed {@link MovementDataset} and expose a trained
 * policy that can (a) reproduce recorded movements and (b) generalize to related
 * but unseen ones. The interface is intentionally backend-agnostic: the default
 * {@link NgramMovementBackend} is a deterministic, dependency-free reference
 * implementation for cloud/CI, while a real on-device small model (e.g. an MLX
 * LoRA policy head) can be dropped in behind the same seam.
 */

/** Sentinel token appended by backends to mark the end of a movement sequence. */
export const MOVEMENT_END_TOKEN: MovementActionToken = "<end>";

export type MovementPredictionContext = {
  /** Action tokens already performed, oldest first. */
  prefix: readonly MovementActionToken[];
  /** Current observation summary, if the caller has one. */
  observation?: string;
};

export type MovementCandidate = {
  token: MovementActionToken;
  probability: number;
  count: number;
};

/** How a prediction was derived, for observability and eval breakdowns. */
export type MovementPredictionSource = "context" | "observation" | "unigram" | "empty";

export type MovementPrediction = {
  /** Most likely next token, or undefined when the model has learned nothing. */
  token: MovementActionToken | undefined;
  /** Probability mass on the top token (0..1). */
  confidence: number;
  /** All candidates, most likely first, deterministically ordered. */
  candidates: MovementCandidate[];
  /**
   * Length of the action context actually matched: `order..1` for an n-gram
   * match, `0` for an observation-only match, `-1` for a unigram/empty fallback.
   */
  backoffOrder: number;
  source: MovementPredictionSource;
};

export type MovementRolloutSeed = {
  prefix?: readonly MovementActionToken[];
  observation?: string;
  maxSteps?: number;
};

export type MovementPolicyMetadata = {
  backendId: string;
  order: number;
  vocabularySize: number;
  sequenceCount: number;
  stepCount: number;
};

export interface TrainedMovementPolicy {
  readonly backendId: string;
  readonly metadata: MovementPolicyMetadata;
  /** Predict the next movement given prior actions and an optional observation. */
  predict(context: MovementPredictionContext): MovementPrediction;
  /**
   * Greedily generate a movement continuation from a seed, stopping at the
   * learned end-of-sequence marker or `maxSteps`. Returns only the newly
   * generated tokens (the seed prefix is not echoed back).
   */
  rollout(seed?: MovementRolloutSeed): MovementActionToken[];
}

export type MovementTrainingOptions = {
  /** Maximum action-context length the model conditions on (default 3). */
  order?: number;
  /** Default cap on generated tokens during {@link TrainedMovementPolicy.rollout}. */
  maxRolloutSteps?: number;
};

export interface MovementPolicyBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainingOptions): TrainedMovementPolicy;
}
