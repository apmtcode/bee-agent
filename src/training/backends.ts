import path from "node:path";
import type { TrainingMode } from "./export-manifest.js";
import type {
  LocalTrainingExecution,
  LocalTrainingJobManifest,
  RlTrainingConfig,
  SftTrainingConfig,
} from "./job-manifest.js";

/**
 * Identifier for the local training runtime a {@link TrainingBackend} drives.
 *
 * - `mlx`     — Apple Silicon MLX LoRA fine-tuning (SFT).
 * - `axolotl` — GRPO/RL fine-tuning with a replay-manifest reward model.
 * - `mock`    — a deterministic, dependency-free trainer used for cloud/CI so
 *               the full plan → launch → artifact pipeline runs end-to-end
 *               without any ML toolchain installed.
 */
export type LocalTrainingRuntime = "mlx" | "axolotl" | "mock";

/**
 * A concrete, launchable local-training plan. Produced by a {@link TrainingBackend}
 * and persisted by the runner; the launch script executes `command` with
 * `environment` and transitions `statePath` through running → completed/failed.
 */
export type TrainingJobPlan = {
  version: 1;
  jobId: string;
  mode: LocalTrainingJobManifest["mode"];
  targetPlatform: "apple-silicon";
  runtime: LocalTrainingRuntime;
  datasetPath: string;
  outputPath: string;
  replayEvalPath: string;
  statePath: string;
  command: string[];
  environment: Record<string, string>;
};

/**
 * Pluggable local-model training backend. A backend owns the runtime-specific
 * command/environment construction for one or more training modes, keeping the
 * runner (I/O, launch-script rendering, state) independent of any particular
 * ML toolchain. This is the seam a real on-device small-model backend plugs
 * into — implement {@link buildPlan} and register it in a
 * {@link TrainingBackendRegistry}.
 */
export interface TrainingBackend {
  /** Runtime identifier stamped onto plans and env (`OPENCLAW_TRAINING_RUNTIME`). */
  readonly runtime: LocalTrainingRuntime;
  /** Modes this backend can plan. */
  readonly supportedModes: readonly TrainingMode[];
  /** Build a launchable plan for `job` using the paths in `execution`. */
  buildPlan(job: LocalTrainingJobManifest, execution: LocalTrainingExecution): TrainingJobPlan;
}

function openclawEnvironment(
  job: LocalTrainingJobManifest,
  runtime: LocalTrainingRuntime,
): Record<string, string> {
  return {
    OPENCLAW_TRAINING_JOB_ID: job.id,
    OPENCLAW_TRAINING_MODE: job.mode,
    OPENCLAW_TARGET_PLATFORM: job.targetPlatform,
    OPENCLAW_TRAINING_RUNTIME: runtime,
    OPENCLAW_REVIEWED_EXPORT_REQUIRED: "true",
    OPENCLAW_RAW_CAPTURE_ALLOWED: "false",
  };
}

/** MLX LoRA supervised fine-tuning backend (Apple Silicon). */
export class MlxTrainingBackend implements TrainingBackend {
  readonly runtime = "mlx" as const;
  readonly supportedModes = ["sft"] as const;

  buildPlan(job: LocalTrainingJobManifest, execution: LocalTrainingExecution): TrainingJobPlan {
    const config = job.config as SftTrainingConfig;
    return {
      version: 1,
      jobId: job.id,
      mode: job.mode,
      targetPlatform: "apple-silicon",
      runtime: this.runtime,
      datasetPath: execution.datasetDir,
      outputPath: path.posix.join(execution.artifactDir, "model.gguf"),
      replayEvalPath: execution.replayEvalFile,
      statePath: execution.stateFile,
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
      environment: openclawEnvironment(job, this.runtime),
    };
  }
}

/** Axolotl GRPO reinforcement-learning backend with replay-manifest rewards. */
export class AxolotlTrainingBackend implements TrainingBackend {
  readonly runtime = "axolotl" as const;
  readonly supportedModes = ["rl"] as const;

  buildPlan(job: LocalTrainingJobManifest, execution: LocalTrainingExecution): TrainingJobPlan {
    const config = job.config as RlTrainingConfig;
    return {
      version: 1,
      jobId: job.id,
      mode: job.mode,
      targetPlatform: "apple-silicon",
      runtime: this.runtime,
      datasetPath: execution.datasetDir,
      outputPath: path.posix.join(execution.artifactDir, "policy.gguf"),
      replayEvalPath: execution.replayEvalFile,
      statePath: execution.stateFile,
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
      environment: openclawEnvironment(job, this.runtime),
    };
  }
}

/**
 * Self-contained Node source for the mock trainer. Reads the reviewed dataset
 * manifest and emits a deterministic "model" artifact whose `weightsDigest` is
 * an FNV-1a hash of the dataset shape — so identical datasets always train to
 * an identical artifact, and any dataset change is observable in the output.
 * Requires no ML toolchain, so it runs anywhere Node runs (cloud/CI included).
 */
export const MOCK_TRAINER_SOURCE = [
  'const fs = require("node:fs");',
  'const path = require("node:path");',
  "const datasetDir = process.env.BEE_MOCK_DATASET_DIR;",
  "const outputPath = process.env.BEE_MOCK_OUTPUT_PATH;",
  "if (!datasetDir || !outputPath) { throw new Error(\"mock trainer: BEE_MOCK_DATASET_DIR/BEE_MOCK_OUTPUT_PATH required\"); }",
  'const manifest = JSON.parse(fs.readFileSync(path.join(datasetDir, "manifest.json"), "utf8"));',
  "const dataset = manifest.dataset || {};",
  "const serialized = JSON.stringify(dataset);",
  "let digest = 2166136261;",
  "for (let i = 0; i < serialized.length; i++) { digest ^= serialized.charCodeAt(i); digest = Math.imul(digest, 16777619); }",
  "const model = {",
  "  version: 1,",
  '  backend: "mock",',
  "  jobId: manifest.jobId,",
  "  mode: manifest.mode,",
  '  weightsDigest: (digest >>> 0).toString(16).padStart(8, "0"),',
  "  trainedFrom: dataset,",
  "  evaluatedReplays: typeof dataset.replayCount === \"number\" ? dataset.replayCount : 0,",
  "};",
  "fs.mkdirSync(path.dirname(outputPath), { recursive: true });",
  'fs.writeFileSync(outputPath, JSON.stringify(model, null, 2) + "\\n");',
].join("\n");

/**
 * Deterministic, dependency-free training backend. Produces a plan that a real
 * `node` process can execute to completion with no ML runtime installed, making
 * the entire local-training pipeline (plan → launch → state → artifact) runnable
 * and assertable in the cloud. The emitted artifact is a reproducible function
 * of the reviewed dataset, so replay evaluation and generalization harnesses can
 * exercise the full flow against synthetic data.
 */
export class MockTrainingBackend implements TrainingBackend {
  readonly runtime = "mock" as const;
  readonly supportedModes = ["sft", "rl"] as const;

  buildPlan(job: LocalTrainingJobManifest, execution: LocalTrainingExecution): TrainingJobPlan {
    const outputPath = path.posix.join(execution.artifactDir, "model.json");
    return {
      version: 1,
      jobId: job.id,
      mode: job.mode,
      targetPlatform: "apple-silicon",
      runtime: this.runtime,
      datasetPath: execution.datasetDir,
      outputPath,
      replayEvalPath: execution.replayEvalFile,
      statePath: execution.stateFile,
      command: ["node", "-e", MOCK_TRAINER_SOURCE],
      environment: {
        ...openclawEnvironment(job, this.runtime),
        BEE_MOCK_DATASET_DIR: execution.datasetDir,
        BEE_MOCK_OUTPUT_PATH: outputPath,
      },
    };
  }
}

/**
 * Resolves a {@link TrainingBackend} for a given training mode. Lets the runner
 * stay backend-agnostic: production uses {@link TrainingBackendRegistry.default}
 * (MLX for SFT, Axolotl for RL); cloud/CI and simulation use
 * {@link TrainingBackendRegistry.mock}. Custom backends (e.g. a real on-device
 * small model) can be supplied per mode.
 */
export class TrainingBackendRegistry {
  private readonly backends: Partial<Record<TrainingMode, TrainingBackend>>;

  constructor(backends: Partial<Record<TrainingMode, TrainingBackend>>) {
    this.backends = { ...backends };
  }

  /** Backend for `mode`, or throws if none is registered. */
  resolve(mode: TrainingMode): TrainingBackend {
    const backend = this.backends[mode];
    if (!backend) {
      throw new Error(`No training backend registered for mode "${mode}"`);
    }
    return backend;
  }

  /** Modes this registry can plan. */
  modes(): TrainingMode[] {
    return Object.keys(this.backends) as TrainingMode[];
  }

  /** Production default: MLX for SFT, Axolotl for RL. */
  static default(): TrainingBackendRegistry {
    return new TrainingBackendRegistry({
      sft: new MlxTrainingBackend(),
      rl: new AxolotlTrainingBackend(),
    });
  }

  /** Cloud/CI + simulation: the deterministic mock backend for every mode. */
  static mock(): TrainingBackendRegistry {
    const mock = new MockTrainingBackend();
    return new TrainingBackendRegistry({ sft: mock, rl: mock });
  }
}
