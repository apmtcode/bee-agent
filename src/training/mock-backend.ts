import path from "node:path";
import { writeJsonAtomic } from "../shared/fs.js";
import type { ReviewedExportManifest } from "./export-manifest.js";
import {
  movementTrajectoryFromReplay,
  replayMovement,
  serializeMovementPolicy,
  trainMovementPolicy,
  type MovementAction,
  type MovementObservation,
  type MovementTrajectory,
} from "./movement-policy.js";
import type {
  TrainingBackend,
  TrainingBackendKind,
  TrainingBackendRequest,
  TrainingBackendResult,
} from "./backend.js";

/**
 * In-process movement-training backend.
 *
 * Fits a {@link trainMovementPolicy} model from the recorded-movement dataset
 * with no native dependencies or child processes, writes the serialized policy
 * to the job's artifact directory, and reports replay fidelity. This is the
 * cloud/CI-friendly backend that lets the full capture → dataset → train →
 * infer loop run and be tested without a real machine or GPU.
 */
export class MockMovementTrainingBackend implements TrainingBackend {
  readonly id = "mock-movement";
  readonly kind: TrainingBackendKind = "in-process";

  constructor(
    private readonly rootDir: string,
    private readonly options: { modelFileName?: string } = {},
  ) {}

  async train(request: TrainingBackendRequest): Promise<TrainingBackendResult> {
    try {
      const order = request.hyperparameters?.order;
      const policy = trainMovementPolicy(request.trajectories, order ? { order } : {});
      const modelRelPath = path.posix.join(request.artifactDir, this.options.modelFileName ?? "movement-policy.json");
      await writeJsonAtomic(path.join(this.rootDir, modelRelPath), JSON.parse(serializeMovementPolicy(policy)));

      return {
        backendId: this.id,
        jobId: request.jobId,
        status: "completed",
        modelPath: modelRelPath,
        metrics: {
          trainedTrajectories: policy.trainedTrajectories,
          trainedActions: policy.trainedActions,
          vocabularySize: policy.vocabulary.length,
          replayFidelity: evaluateReplayFidelity(policy, request.trajectories),
        },
      };
    } catch (error) {
      return {
        backendId: this.id,
        jobId: request.jobId,
        status: "failed",
        metrics: { trainedTrajectories: 0, trainedActions: 0, vocabularySize: 0 },
        error: error instanceof Error ? error.message : String(error),
      };
    }
  }
}

/** Normalize a reviewed export's replay manifests into the training dataset. */
export function movementDatasetFromExport(manifest: ReviewedExportManifest): MovementTrajectory[] {
  return manifest.replays.map((replay, index) =>
    movementTrajectoryFromReplay(replay.trajectoryIds[0] ?? `replay-${index}`, replay.events),
  );
}

/**
 * Measure how faithfully the trained model reproduces its own training
 * trajectories: the fraction whose recorded action sequence is replayed exactly
 * from the trajectory's seed observation. A direct, deterministic proxy for
 * "can repeat recorded movements" (objective #2c).
 */
export function evaluateReplayFidelity(
  policy: Parameters<typeof replayMovement>[0],
  trajectories: MovementTrajectory[],
): number {
  const withActions = trajectories.filter((trajectory) =>
    trajectory.events.some((event) => event.kind === "action"),
  );
  if (withActions.length === 0) {
    return 0;
  }
  let reproduced = 0;
  for (const trajectory of withActions) {
    const expected = recordedActions(trajectory);
    const seed = firstObservation(trajectory);
    const replayed = replayMovement(policy, { observation: seed, maxSteps: expected.length + 4 });
    if (actionsEqual(expected, replayed)) {
      reproduced += 1;
    }
  }
  return reproduced / withActions.length;
}

function recordedActions(trajectory: MovementTrajectory): MovementAction[] {
  return trajectory.events.flatMap((event) =>
    event.kind === "action" ? [{ tool: event.tool, summary: event.summary }] : [],
  );
}

function firstObservation(trajectory: MovementTrajectory): MovementObservation | undefined {
  for (const event of trajectory.events) {
    if (event.kind === "observation") {
      return { source: event.source, summary: event.summary };
    }
  }
  return undefined;
}

function actionsEqual(a: MovementAction[], b: MovementAction[]): boolean {
  return a.length === b.length && a.every((action, index) => action.tool === b[index]?.tool && action.summary === b[index]?.summary);
}
