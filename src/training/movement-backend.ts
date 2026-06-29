import type { ReplayManifest } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Pluggable local-model training backend for the local-movement learning
 * subsystem (standing objective #2).
 *
 * The capture pipeline records movements into trajectories/replays; this module
 * turns those into a normalized {@link MovementDataset} and defines the seam a
 * backend implements to (a) post-train a model that repeats recorded movements
 * and (b) generalize to new-but-related movements.
 *
 * Backends are intentionally abstract: a real backend drives an on-device small
 * model (e.g. MLX on Apple Silicon — see {@link ./runner.ts}); the bundled
 * {@link ../training/mock-movement-backend.ts} is a deterministic in-process
 * learner so the pipeline is exercised in the cloud/CI with no OS access.
 */

/** A single learned movement step distilled from an action event. */
export type MovementStep = {
  ts: number;
  /** Tool/device channel that produced the action (e.g. "device"). */
  tool: string;
  /** Human-readable action summary (e.g. "tapped Submit"). */
  summary: string;
  /** Observation summary that immediately preceded the action, if any. */
  context?: string;
};

/** An ordered movement sequence belonging to one trajectory. */
export type MovementSequence = {
  trajectoryId: string;
  steps: MovementStep[];
};

/** Normalized training dataset built from replays/trajectories. */
export type MovementDataset = {
  version: 1;
  sequenceCount: number;
  stepCount: number;
  sequences: MovementSequence[];
};

/** Where a prediction came from, for explainability and eval. */
export type MovementPredictionSource = "transition" | "context" | "fallback";

/** A predicted next movement step plus calibrated confidence. */
export type MovementPrediction = {
  step: MovementStep;
  /** 0..1 — share of observed evidence supporting this prediction. */
  confidence: number;
  source: MovementPredictionSource;
};

/** Serialized, backend-tagged model payload for persistence/transport. */
export type SerializedMovementModel = {
  version: 1;
  backendId: string;
  sequenceCount: number;
  stepCount: number;
  /** Backend-specific opaque payload; only the producing backend reads it. */
  payload: unknown;
};

/** A trained model: repeats recorded movements and generalizes to related ones. */
export interface TrainedMovementModel {
  readonly backendId: string;
  readonly sequenceCount: number;
  readonly stepCount: number;
  /** Exactly reproduce the recorded steps for a trajectory (replay). */
  replay(trajectoryId: string): MovementStep[] | undefined;
  /** Predict the next step from the last action and/or a context cue. */
  predictNext(input: { lastSummary?: string; context?: string }): MovementPrediction | undefined;
  /** Generalize: synthesize a plan for a (possibly novel) goal context. */
  generate(input: { context: string; maxSteps?: number }): MovementStep[];
  serialize(): SerializedMovementModel;
}

export type TrainMovementOptions = {
  /** Optional label propagated into logs/serialized payloads. */
  label?: string;
};

/** Declared backend capabilities, for selection and eval gating. */
export type MovementBackendCapabilities = {
  /** Trains/infers on the local device (vs. emitting a launch plan only). */
  onDevice: boolean;
  /** Produces identical output for identical input (required for CI). */
  deterministic: boolean;
  /** Can produce plans for unseen-but-related contexts. */
  generalizes: boolean;
};

/** The pluggable backend seam. */
export interface MovementTrainingBackend {
  readonly id: string;
  readonly description: string;
  readonly capabilities: MovementBackendCapabilities;
  train(dataset: MovementDataset, options?: TrainMovementOptions): Promise<TrainedMovementModel>;
  /** Rehydrate a previously serialized model produced by this backend. */
  load(serialized: SerializedMovementModel): TrainedMovementModel;
}

/**
 * Build a normalized movement dataset from replay manifests. Actions are grouped
 * by trajectory, ordered by timestamp, and annotated with the most recent
 * preceding observation summary as their context.
 */
export function buildMovementDataset(manifests: ReplayManifest[]): MovementDataset {
  const stepsByTrajectory = new Map<string, MovementStep[]>();
  for (const manifest of manifests) {
    const lastContext = new Map<string, string>();
    // Manifest events are already ts-sorted by buildReplayManifest.
    for (const event of manifest.events) {
      if (event.kind === "observation") {
        lastContext.set(event.trajectoryId, event.summary);
      } else if (event.kind === "action") {
        const steps = stepsByTrajectory.get(event.trajectoryId) ?? [];
        const context = lastContext.get(event.trajectoryId);
        steps.push({
          ts: event.ts,
          tool: event.tool,
          summary: event.summary,
          ...(context !== undefined ? { context } : {}),
        });
        stepsByTrajectory.set(event.trajectoryId, steps);
      }
    }
  }
  return finalizeDataset(stepsByTrajectory);
}

/** Build a movement dataset directly from trajectory spans. */
export function buildMovementDatasetFromTrajectories(trajectories: TrajectorySpan[]): MovementDataset {
  const stepsByTrajectory = new Map<string, MovementStep[]>();
  for (const trajectory of trajectories) {
    const observations = [...trajectory.observations].sort((a, b) => a.ts - b.ts);
    const steps: MovementStep[] = trajectory.actions
      .slice()
      .sort((a, b) => a.ts - b.ts)
      .map((action) => {
        const context = latestBefore(observations, action.ts);
        return {
          ts: action.ts,
          tool: action.tool,
          summary: action.summary,
          ...(context !== undefined ? { context } : {}),
        };
      });
    if (steps.length > 0) {
      stepsByTrajectory.set(trajectory.id, steps);
    }
  }
  return finalizeDataset(stepsByTrajectory);
}

function latestBefore(
  observations: TrajectorySpan["observations"],
  ts: number,
): string | undefined {
  let summary: string | undefined;
  for (const observation of observations) {
    if (observation.ts <= ts) {
      summary = observation.summary;
    } else {
      break;
    }
  }
  return summary;
}

function finalizeDataset(stepsByTrajectory: Map<string, MovementStep[]>): MovementDataset {
  const sequences: MovementSequence[] = [...stepsByTrajectory.entries()]
    .map(([trajectoryId, steps]) => ({
      trajectoryId,
      steps: steps.slice().sort((a, b) => a.ts - b.ts),
    }))
    .filter((sequence) => sequence.steps.length > 0)
    .sort((a, b) => a.trajectoryId.localeCompare(b.trajectoryId));
  const stepCount = sequences.reduce((sum, sequence) => sum + sequence.steps.length, 0);
  return {
    version: 1,
    sequenceCount: sequences.length,
    stepCount,
    sequences,
  };
}

/** A small in-memory registry making backends pluggable/selectable by id. */
export class MovementBackendRegistry {
  private readonly backends = new Map<string, MovementTrainingBackend>();

  register(backend: MovementTrainingBackend): this {
    if (this.backends.has(backend.id)) {
      throw new Error(`movement backend already registered: ${backend.id}`);
    }
    this.backends.set(backend.id, backend);
    return this;
  }

  has(id: string): boolean {
    return this.backends.has(id);
  }

  get(id: string): MovementTrainingBackend | undefined {
    return this.backends.get(id);
  }

  require(id: string): MovementTrainingBackend {
    const backend = this.backends.get(id);
    if (!backend) {
      const known = this.list().map((entry) => entry.id).join(", ") || "<none>";
      throw new Error(`unknown movement backend: ${id} (registered: ${known})`);
    }
    return backend;
  }

  list(): MovementTrainingBackend[] {
    return [...this.backends.values()].sort((a, b) => a.id.localeCompare(b.id));
  }
}
