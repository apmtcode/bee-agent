import type { MovementActionToken, MovementDataset } from "./dataset.js";

/**
 * The runtime context a prediction is made in: the recent movement history
 * (oldest first) plus the ambient app/screen context.
 */
export type MovementContext = {
  history: MovementActionToken[];
  appContext?: string;
};

export type MovementCandidate = {
  action: MovementActionToken;
  probability: number;
};

export type MovementPrediction = {
  /** Most likely next movement. */
  action: MovementActionToken;
  probability: number;
  /** Full ranked distribution (deterministic ordering). */
  ranked: MovementCandidate[];
  /**
   * How much history context was actually usable for this prediction. Equal to
   * the requested order for exact reproduction; a smaller number means the
   * model generalised by backing off to a shorter, familiar context.
   */
  backoffOrder: number;
};

export type MovementTrainingConfig = {
  /** Context window; defaults to the dataset's order. */
  order?: number;
  /** Additive (Laplace) smoothing mass. Defaults to 0 (pure MLE). */
  smoothing?: number;
};

/**
 * A trained, in-memory movement policy. Backends produce this from a dataset;
 * it is serialisable so a trained model can be persisted as an artifact and
 * reloaded for inference/replay.
 */
export interface TrainedMovementModel {
  readonly backend: string;
  /** Predict the next movement, or undefined if the model is empty. */
  predict(context: MovementContext): MovementPrediction | undefined;
  /** A plain, JSON-serialisable snapshot of the trained parameters. */
  serialize(): SerializedMovementModel;
}

export type SerializedMovementModel = {
  backend: string;
  version: 1;
  [key: string]: unknown;
};

/**
 * Pluggable local-model backend. The cloud/CI ships a deterministic in-process
 * backend; a real on-device small model (e.g. an MLX/llama.cpp policy) can
 * implement this same seam without touching the capture/dataset pipeline.
 */
export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, config?: MovementTrainingConfig): Promise<TrainedMovementModel>;
  load(serialized: SerializedMovementModel): TrainedMovementModel;
}
