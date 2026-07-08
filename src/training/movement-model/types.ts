/**
 * Movement-model subsystem — types.
 *
 * This is the in-process, cloud-testable half of standing objective 2(c)/(d):
 * "post-train a local model on the recorded dataset to repeat the recorded
 * movements" and "generalize to perform new but related movements".
 *
 * The runner (`../runner.ts`) only *plans* external MLX/axolotl processes for a
 * real on-device job. This module provides a pluggable, fully deterministic
 * model backend that can train and infer entirely in-process, so the
 * capture -> dataset -> train -> replay -> generalize loop can be validated in
 * the cloud with synthetic event streams. A real on-device small-model backend
 * can be swapped in behind {@link MovementModelBackend} without touching the
 * tokenizer, dataset format, or eval harness.
 */

/** Canonical, discrete form of a single recorded movement step. */
export type MovementToken = string;

/** Sentinel token marking the start of a sequence (context padding). */
export const MOVEMENT_START_TOKEN: MovementToken = "start";

/** Sentinel token marking the end of a sequence (generation stop signal). */
export const MOVEMENT_END_TOKEN: MovementToken = "end";

/**
 * A single movement step. `token` is the learnable/discrete symbol; the
 * remaining fields carry enough structure to reconstruct a replayable action
 * (see {@link ../../capture/trajectory.TrajectoryAction}).
 */
export type MovementStep = {
  token: MovementToken;
  tool: string;
  ts: number;
  gesture?: string;
  target?: string;
  direction?: string;
};

/** An ordered run of movement steps captured within one trajectory/session. */
export type MovementSequence = {
  id: string;
  steps: MovementStep[];
};

/** The replayable dataset a backend trains on. */
export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

export type MovementModelMetadata = {
  backend: string;
  order: number;
  vocabularySize: number;
  sequenceCount: number;
  transitionCount: number;
  trainedAt?: string;
};

/**
 * Opaque, serializable trained-model handle. `state` is backend-private; only
 * the producing backend may interpret it.
 */
export type TrainedMovementModel = {
  metadata: MovementModelMetadata;
  state: unknown;
};

export type MovementPrediction = {
  token: MovementToken;
  probability: number;
  /** Context length (n-gram order) that produced this prediction after backoff. */
  order: number;
};

export type MovementPredictOptions = {
  /** Return at most this many candidates (highest probability first). */
  topK?: number;
  /** Exclude these tokens from the ranking (e.g. already-tried dead ends). */
  exclude?: readonly MovementToken[];
};

/**
 * Pluggable backend seam. The deterministic {@link MarkovMovementBackend} is the
 * default; a real on-device small model implements the same three methods.
 */
export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset): TrainedMovementModel;
  predict(
    model: TrainedMovementModel,
    context: readonly MovementToken[],
    options?: MovementPredictOptions,
  ): MovementPrediction[];
  /**
   * Deterministically roll the model forward from `seed`, taking the
   * highest-probability token at each step, until an end sentinel is produced
   * or `maxSteps` is reached. This is the "repeat the recorded movement" path.
   */
  generate(
    model: TrainedMovementModel,
    seed: readonly MovementToken[],
    maxSteps: number,
  ): MovementToken[];
}
