/**
 * Pluggable training-backend seam for the local-movement learning subsystem.
 *
 * Historically the runner hard-wired Apple-Silicon runtimes (mlx for SFT,
 * axolotl for RL). That prevented (a) running the pipeline anywhere but a Mac
 * and (b) validating the loop in the cloud/CI. A `TrainingBackend` decouples the
 * *runtime-specific* slice of a plan (which command + env actually trains a
 * model, and what artifact it emits) from the runner's path/orchestration logic.
 *
 * Built-in backends:
 *   - `MlxSftBackend`     — SFT via mlx_lm.lora   (unchanged behavior)
 *   - `AxolotlRlBackend`  — GRPO/RL via axolotl    (unchanged behavior)
 *   - `MockLocalBackend`  — dependency-free, deterministic movement model
 *                           (`mock-model.ts`); runs anywhere, gates cloud tests.
 *
 * Third parties can register their own backend (e.g. a real small on-device
 * model) without touching the runner.
 */
import path from "node:path";
import {
  trainMockMovementModel,
  type MockMovementModel,
  type MovementDataset,
} from "./mock-model.js";
import { readJsonFile, writeJsonAtomic } from "../shared/fs.js";
import type { LocalTrainingExecution, LocalTrainingJobManifest } from "./job-manifest.js";
import type { RlTrainingConfig, SftTrainingConfig } from "./job-manifest.js";
import type { TrainingMode } from "./export-manifest.js";

/** The runtime-specific portion of a training plan produced by a backend. */
export type TrainingBackendPlan = {
  /** Backend runtime identifier surfaced on the plan (e.g. `mlx`, `mock`). */
  runtime: string;
  /** Platform the plan targets; `apple-silicon` for the built-ins, `portable` for mock. */
  targetPlatform: string;
  /** Artifact filename written under the execution's artifact directory. */
  outputArtifact: string;
  /** The command (argv) that performs training. */
  command: string[];
  /** Environment variables the launch script exports before running `command`. */
  environment: Record<string, string>;
};

export type TrainingBackendInput = {
  job: LocalTrainingJobManifest;
  execution: LocalTrainingExecution;
};

/** A pluggable training backend. Keep `buildPlan` pure (no clocks/RNG/IO). */
export interface TrainingBackend {
  readonly id: string;
  supportsMode(mode: TrainingMode): boolean;
  buildPlan(input: TrainingBackendInput): TrainingBackendPlan;
}

const SHARED_ENV = {
  OPENCLAW_REVIEWED_EXPORT_REQUIRED: "true",
  OPENCLAW_RAW_CAPTURE_ALLOWED: "false",
} as const;

function baseEnv(job: LocalTrainingJobManifest, runtime: string): Record<string, string> {
  return {
    OPENCLAW_TRAINING_JOB_ID: job.id,
    OPENCLAW_TRAINING_MODE: job.mode,
    OPENCLAW_TARGET_PLATFORM: job.targetPlatform,
    OPENCLAW_TRAINING_RUNTIME: runtime,
    ...SHARED_ENV,
  };
}

/** SFT via Apple-Silicon mlx_lm.lora. Reproduces the original runner behavior. */
export class MlxSftBackend implements TrainingBackend {
  readonly id = "mlx-sft";

  supportsMode(mode: TrainingMode): boolean {
    return mode === "sft";
  }

  buildPlan({ job, execution }: TrainingBackendInput): TrainingBackendPlan {
    const config = job.config as SftTrainingConfig;
    return {
      runtime: "mlx",
      targetPlatform: "apple-silicon",
      outputArtifact: "model.gguf",
      command: [
        "python3",
        "-m",
        "mlx_lm.lora",
        "--train",
        "--data",
        execution.datasetDir,
        "--adapter-path",
        execution.artifactDir,
        "--learning-rate",
        String(config.learningRate),
        "--batch-size",
        String(config.batchSize),
        "--iters",
        String(config.epochs * 1000),
      ],
      environment: baseEnv(job, "mlx"),
    };
  }
}

/** GRPO/RL via axolotl with a replay-manifest reward. Reproduces original behavior. */
export class AxolotlRlBackend implements TrainingBackend {
  readonly id = "axolotl-rl";

  supportsMode(mode: TrainingMode): boolean {
    return mode === "rl";
  }

  buildPlan({ job, execution }: TrainingBackendInput): TrainingBackendPlan {
    const config = job.config as RlTrainingConfig;
    return {
      runtime: "axolotl",
      targetPlatform: "apple-silicon",
      outputArtifact: "policy.gguf",
      command: [
        "python3",
        "-m",
        "axolotl.cli.train",
        execution.planFile,
        "--reward-model",
        "replay-manifest",
        "--rollouts",
        String(config.rolloutCount),
        "--kl-penalty",
        String(config.klPenalty),
      ],
      environment: baseEnv(job, "axolotl"),
    };
  }
}

/** Relative path (under the execution root) of the movement dataset the mock backend reads. */
export function mockDatasetFile(execution: LocalTrainingExecution): string {
  return path.posix.join(execution.datasetDir, "movement-dataset.json");
}

/**
 * Dependency-free, deterministic backend. Trains the Markov movement model from
 * a `movement-dataset.json` and writes the artifact — no mlx/axolotl/GPU needed,
 * so the full pipeline runs anywhere and gates cloud tests. The command invokes
 * `runMockTrainingJob` (exported from the package) via node.
 */
export class MockLocalBackend implements TrainingBackend {
  readonly id = "mock-local";

  supportsMode(): boolean {
    return true;
  }

  buildPlan({ job, execution }: TrainingBackendInput): TrainingBackendPlan {
    const outputArtifact = "movement-model.json";
    const datasetFile = mockDatasetFile(execution);
    const outputPath = path.posix.join(execution.artifactDir, outputArtifact);
    const bootstrap =
      "import('@openclaw/operator')" +
      ".then((m) => m.runMockTrainingJob({" +
      " datasetFile: process.env.OPENCLAW_MOCK_DATASET_FILE," +
      " outputPath: process.env.OPENCLAW_MOCK_OUTPUT_PATH }))" +
      ".then(() => process.exit(0))" +
      ".catch((error) => { console.error(error); process.exit(1); });";
    return {
      runtime: "mock",
      targetPlatform: "portable",
      outputArtifact,
      command: ["node", "--input-type=module", "-e", bootstrap],
      environment: {
        ...baseEnv(job, "mock"),
        OPENCLAW_MOCK_DATASET_FILE: datasetFile,
        OPENCLAW_MOCK_OUTPUT_PATH: outputPath,
      },
    };
  }
}

/** Ordered registry of backends; first supporting backend (or preferred id) wins. */
export class TrainingBackendRegistry {
  private readonly backends: TrainingBackend[] = [];

  constructor(backends: TrainingBackend[] = []) {
    for (const backend of backends) {
      this.register(backend);
    }
  }

  register(backend: TrainingBackend): this {
    if (this.backends.some((existing) => existing.id === backend.id)) {
      throw new Error(`training backend "${backend.id}" is already registered`);
    }
    this.backends.push(backend);
    return this;
  }

  list(): TrainingBackend[] {
    return [...this.backends];
  }

  get(id: string): TrainingBackend | undefined {
    return this.backends.find((backend) => backend.id === id);
  }

  /** Select a backend for `mode`, optionally forcing a specific backend id. */
  select(mode: TrainingMode, preferredId?: string): TrainingBackend {
    if (preferredId) {
      const preferred = this.get(preferredId);
      if (!preferred) {
        throw new Error(`unknown training backend "${preferredId}"`);
      }
      if (!preferred.supportsMode(mode)) {
        throw new Error(`training backend "${preferredId}" does not support mode "${mode}"`);
      }
      return preferred;
    }
    const match = this.backends.find((backend) => backend.supportsMode(mode));
    if (!match) {
      throw new Error(`no training backend supports mode "${mode}"`);
    }
    return match;
  }
}

/** Default registry reproducing the original Apple-Silicon behavior (mlx + axolotl). */
export function defaultTrainingBackendRegistry(): TrainingBackendRegistry {
  return new TrainingBackendRegistry([new MlxSftBackend(), new AxolotlRlBackend()]);
}

/** Registry that adds the portable mock backend (used for cloud/local dry runs). */
export function mockTrainingBackendRegistry(): TrainingBackendRegistry {
  return new TrainingBackendRegistry([
    new MockLocalBackend(),
    new MlxSftBackend(),
    new AxolotlRlBackend(),
  ]);
}

export type RunMockTrainingJobParams = {
  datasetFile?: string;
  outputPath?: string;
};

/**
 * Read a `MovementDataset`, train the mock movement model, and write the model
 * artifact atomically. This is what `MockLocalBackend`'s command invokes at run
 * time; it is also directly unit-testable in-process (no subprocess required).
 */
export async function runMockTrainingJob(
  params: RunMockTrainingJobParams,
): Promise<MockMovementModel> {
  if (!params.datasetFile) {
    throw new Error("runMockTrainingJob requires a datasetFile");
  }
  if (!params.outputPath) {
    throw new Error("runMockTrainingJob requires an outputPath");
  }
  const dataset = await readJsonFile<MovementDataset | undefined>(params.datasetFile, undefined);
  if (!dataset || !Array.isArray(dataset.sequences)) {
    throw new Error(`movement dataset not found or malformed at ${params.datasetFile}`);
  }
  const model = trainMockMovementModel(dataset.sequences);
  await writeJsonAtomic(params.outputPath, model);
  return model;
}
