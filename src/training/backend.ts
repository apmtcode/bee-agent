import type { TrainingMode } from "./export-manifest.js";
import type { MovementTrajectory } from "./movement-policy.js";

/**
 * Pluggable training backend seam for the local-movement learning subsystem.
 *
 * The recorded-movement dataset (reviewed export → replay manifests) is the
 * common input; *how* a model is fit from it is backend-specific:
 *
 *   - The in-process {@link MockMovementTrainingBackend} (mock-backend.ts) fits
 *     a deterministic local movement policy with no native deps, so training is
 *     exercised end-to-end in the cloud/CI.
 *   - On-device backends (mlx SFT / axolotl RL, see runner.ts) shell out to a
 *     real trainer on Apple Silicon. They can implement this same interface so
 *     callers select a backend without branching on platform details.
 */

export type TrainingBackendKind = "in-process" | "external-process";

export type TrainingBackendRequest = {
  jobId: string;
  mode: TrainingMode;
  /** Directory (root-relative) where the backend writes its model artifact. */
  artifactDir: string;
  /** Normalized recorded-movement dataset to train on. */
  trajectories: MovementTrajectory[];
  /** Optional backend-specific knobs (order, learning-rate, …). */
  hyperparameters?: Record<string, number>;
};

export type TrainingBackendMetrics = {
  trainedTrajectories: number;
  trainedActions: number;
  vocabularySize: number;
  /** Fraction (0..1) of training trajectories the model reproduces exactly. */
  replayFidelity?: number;
};

export type TrainingBackendResult = {
  backendId: string;
  jobId: string;
  status: "completed" | "failed";
  /** Root-relative path to the written model artifact, when produced. */
  modelPath?: string;
  metrics: TrainingBackendMetrics;
  error?: string;
};

export interface TrainingBackend {
  readonly id: string;
  readonly kind: TrainingBackendKind;
  train(request: TrainingBackendRequest): Promise<TrainingBackendResult>;
}
