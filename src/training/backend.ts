/**
 * Pluggable training backend interface for the movement-learning subsystem.
 *
 * The existing `LocalAppleSiliconTrainingRunner` emits an MLX/Axolotl launch
 * plan that only executes on a real Apple-silicon machine. That is the right
 * production target, but it cannot run — let alone be validated — in the cloud.
 * This module introduces a backend seam so the runner is no longer hardwired to
 * one runtime: any `TrainingBackend` can be registered, and a deterministic,
 * in-process `MockMovementTrainingBackend` provides an executable train -> infer
 * path that works in CI with no OS access (standing objective #2, part c, and
 * the guardrail requiring a simulated implementation for OS-facing features).
 *
 * A real on-device small-model backend implements this same interface, reading
 * the same reviewed-export dataset and writing a model artifact the replay
 * engine can consume.
 */
import path from "node:path";
import { readJsonFile, writeJsonAtomic } from "../shared/fs.js";
import type { ExportedReplayManifest, ReviewedExportManifest, TrainingMode } from "./export-manifest.js";
import {
  evaluatePolicy,
  trainMovementPolicy,
  type MovementPolicyEvent,
  type MovementPolicyModel,
  type MovementReplay,
} from "./policy-model.js";

export type TrainingBackendExecutionKind = "local-process" | "in-process";

export type TrainingBackendDescriptor = {
  id: string;
  /** Human-facing runtime label (e.g. "mlx", "axolotl", "in-process-frequency"). */
  runtime: string;
  targetPlatform: string;
  kind: TrainingBackendExecutionKind;
  /** True when the backend can execute fully in the cloud/CI without OS access. */
  supportsCloudExecution: boolean;
  supportedModes: TrainingMode[];
};

export type TrainingBackendRequest = {
  jobId: string;
  mode: TrainingMode;
  /** Reviewed replay streams that form the dataset. */
  replays: MovementReplay[];
  /** Directory the backend writes artifacts into (created if absent). */
  outputDir: string;
};

export type TrainingBackendMetrics = {
  observationCount: number;
  actionCount: number;
  transitionCount: number;
  /** Train-on-train top-1 accuracy — a determinism/sanity signal, not a
   * generalization score. */
  selfConsistency: number;
};

export type TrainingBackendResult = {
  backendId: string;
  jobId: string;
  status: "completed" | "failed";
  /** Path to the serialized model artifact (relative to `outputDir`'s parent
   * expectations are the caller's; we return what we wrote). */
  modelPath?: string;
  metrics?: TrainingBackendMetrics;
  error?: string;
};

export interface TrainingBackend {
  readonly descriptor: TrainingBackendDescriptor;
  train(request: TrainingBackendRequest): Promise<TrainingBackendResult>;
}

/** Convert a reviewed-export replay manifest into policy training input,
 * dropping transcript events (the policy learns observation->action only). */
export function replaysFromExport(manifest: Pick<ReviewedExportManifest, "replays">): MovementReplay[] {
  return manifest.replays.map((replay) => replayFromManifest(replay));
}

function replayFromManifest(replay: ExportedReplayManifest): MovementReplay {
  const events: MovementPolicyEvent[] = [];
  for (const event of replay.events) {
    if (event.kind === "observation") {
      events.push({ kind: "observation", source: event.source, summary: event.summary });
    } else if (event.kind === "action") {
      events.push({ kind: "action", tool: event.tool, summary: event.summary });
    }
    // transcript events are intentionally ignored by the movement policy.
  }
  return { events };
}

export const MOCK_MOVEMENT_MODEL_FILENAME = "model.json";

/**
 * Deterministic in-process backend: trains the frequency next-action policy from
 * the request's replays and writes it as `model.json`. No child process, no
 * shell, no OS input — safe and reproducible in the cloud. Intended as the
 * default backend for tests and as a local dry-run before dispatching a real
 * on-device job.
 */
export class MockMovementTrainingBackend implements TrainingBackend {
  readonly descriptor: TrainingBackendDescriptor = {
    id: "mock-movement",
    runtime: "in-process-frequency",
    targetPlatform: "any",
    kind: "in-process",
    supportsCloudExecution: true,
    supportedModes: ["sft", "rl"],
  };

  async train(request: TrainingBackendRequest): Promise<TrainingBackendResult> {
    try {
      const model = trainMovementPolicy(request.replays);
      const evaluation = evaluatePolicy(model, request.replays);
      const modelPath = path.join(request.outputDir, MOCK_MOVEMENT_MODEL_FILENAME);
      await writeJsonAtomic(modelPath, model);
      return {
        backendId: this.descriptor.id,
        jobId: request.jobId,
        status: "completed",
        modelPath,
        metrics: {
          observationCount: model.observationCount,
          actionCount: model.actionCount,
          transitionCount: model.transitions.length,
          selfConsistency: evaluation.accuracy,
        },
      };
    } catch (error) {
      return {
        backendId: this.descriptor.id,
        jobId: request.jobId,
        status: "failed",
        error: error instanceof Error ? error.message : String(error),
      };
    }
  }

  /** Load a model artifact written by a previous `train` call. */
  async loadModel(modelPath: string): Promise<MovementPolicyModel | undefined> {
    return await readJsonFile<MovementPolicyModel | undefined>(modelPath, undefined);
  }
}

/**
 * Registry of available training backends. Lets the runtime pick a backend by id
 * (e.g. "mock-movement" in the cloud, an on-device backend locally) instead of
 * hardcoding one. Register defensively — duplicate ids throw.
 */
export class TrainingBackendRegistry {
  private readonly backends = new Map<string, TrainingBackend>();
  private defaultId: string | undefined;

  register(backend: TrainingBackend, options: { makeDefault?: boolean } = {}): this {
    if (this.backends.has(backend.descriptor.id)) {
      throw new Error(`Training backend "${backend.descriptor.id}" is already registered`);
    }
    this.backends.set(backend.descriptor.id, backend);
    if (options.makeDefault || this.defaultId === undefined) {
      this.defaultId = backend.descriptor.id;
    }
    return this;
  }

  has(id: string): boolean {
    return this.backends.has(id);
  }

  get(id: string): TrainingBackend {
    const backend = this.backends.get(id);
    if (!backend) {
      throw new Error(`Unknown training backend "${id}"`);
    }
    return backend;
  }

  getDefault(): TrainingBackend {
    if (!this.defaultId) {
      throw new Error("No training backends registered");
    }
    return this.get(this.defaultId);
  }

  /** Backends able to run in the cloud/CI, in registration order. */
  listCloudCapable(): TrainingBackend[] {
    return this.list().filter((backend) => backend.descriptor.supportsCloudExecution);
  }

  list(): TrainingBackend[] {
    return [...this.backends.values()];
  }

  describe(): TrainingBackendDescriptor[] {
    return this.list().map((backend) => backend.descriptor);
  }
}

/** A registry pre-seeded with the deterministic mock backend as default. */
export function createDefaultTrainingBackendRegistry(): TrainingBackendRegistry {
  return new TrainingBackendRegistry().register(new MockMovementTrainingBackend(), { makeDefault: true });
}
