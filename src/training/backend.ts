import { createHash } from "node:crypto";
import path from "node:path";
import type { TrainingTargetPlatform } from "./export-manifest.js";
import type {
  LocalTrainingExecution,
  LocalTrainingJobManifest,
  RlTrainingConfig,
  SftTrainingConfig,
} from "./job-manifest.js";

/**
 * Runtime identifier for a produced training plan. `mlx`/`axolotl` are the
 * on-device Apple-Silicon runtimes; `mock` is a deterministic, dependency-free
 * runtime used to validate the training pipeline in the cloud/CI where no GPU
 * or python toolchain is available.
 */
export type LocalTrainingRuntime = "mlx" | "axolotl" | "mock";

export type TrainingJobPlan = {
  version: 1;
  jobId: string;
  mode: LocalTrainingJobManifest["mode"];
  targetPlatform: TrainingTargetPlatform | "portable";
  runtime: LocalTrainingRuntime;
  datasetPath: string;
  outputPath: string;
  replayEvalPath: string;
  statePath: string;
  command: string[];
  environment: Record<string, string>;
};

/**
 * A pluggable training backend. A backend knows how to turn a reviewed training
 * job + its prepared execution layout into a concrete, launchable
 * {@link TrainingJobPlan}. This is the seam that makes the local-model training
 * subsystem model-agnostic: swap the backend to target a different local
 * runtime (mlx, axolotl, llama.cpp, a mock, …) without touching the runner,
 * execution service, or job store.
 */
export interface TrainingBackend {
  /** Stable identifier, e.g. `"apple-silicon-mlx"` or `"mock"`. */
  readonly id: string;
  /** True when this backend can produce a plan for the given job. */
  supports(job: LocalTrainingJobManifest): boolean;
  /** Produce a concrete, launchable plan for the job. */
  buildPlan(job: LocalTrainingJobManifest, execution: LocalTrainingExecution): TrainingJobPlan;
}

/**
 * A deterministic model artifact produced by an in-process backend. Backends
 * that can "train" without external tooling (currently only the mock) implement
 * {@link InProcessTrainingBackend} so the full capture → dataset → train → infer
 * loop can be exercised end-to-end in tests.
 */
export type TrainedModelArtifact = {
  version: 1;
  backendId: string;
  jobId: string;
  mode: LocalTrainingJobManifest["mode"];
  /** Stable fingerprint of the inputs the model was trained on. */
  modelFingerprint: string;
  /** Deterministic per-input-signal weights the mock "learned". */
  weights: Record<string, number>;
  datasetSignature: {
    promotedSkillCount: number;
    memoryCount: number;
    trajectoryCount: number;
    replayCount: number;
  };
};

export interface InProcessTrainingBackend extends TrainingBackend {
  /**
   * Deterministically simulate training on the job's reviewed dataset and
   * return a reproducible model artifact. Same job → same artifact, always.
   */
  simulateTraining(job: LocalTrainingJobManifest): TrainedModelArtifact;
}

export function isInProcessTrainingBackend(
  backend: TrainingBackend,
): backend is InProcessTrainingBackend {
  return typeof (backend as InProcessTrainingBackend).simulateTraining === "function";
}

/**
 * The on-device Apple-Silicon backend. SFT jobs run through `mlx_lm.lora`; RL
 * jobs run through `axolotl` with a replay-manifest reward model. This is the
 * exact behaviour the runner shipped before backends were pluggable.
 */
export class AppleSiliconTrainingBackend implements TrainingBackend {
  readonly id = "apple-silicon-mlx";

  supports(job: LocalTrainingJobManifest): boolean {
    return job.targetPlatform === "apple-silicon";
  }

  buildPlan(job: LocalTrainingJobManifest, execution: LocalTrainingExecution): TrainingJobPlan {
    if (job.mode === "sft") {
      const config = job.config as SftTrainingConfig;
      return {
        version: 1,
        jobId: job.id,
        mode: job.mode,
        targetPlatform: "apple-silicon",
        runtime: "mlx",
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
        environment: {
          OPENCLAW_TRAINING_JOB_ID: job.id,
          OPENCLAW_TRAINING_MODE: job.mode,
          OPENCLAW_TARGET_PLATFORM: job.targetPlatform,
          OPENCLAW_TRAINING_RUNTIME: "mlx",
          OPENCLAW_REVIEWED_EXPORT_REQUIRED: "true",
          OPENCLAW_RAW_CAPTURE_ALLOWED: "false",
        },
      };
    }

    const config = job.config as RlTrainingConfig;
    return {
      version: 1,
      jobId: job.id,
      mode: job.mode,
      targetPlatform: "apple-silicon",
      runtime: "axolotl",
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
      environment: {
        OPENCLAW_TRAINING_JOB_ID: job.id,
        OPENCLAW_TRAINING_MODE: job.mode,
        OPENCLAW_TARGET_PLATFORM: job.targetPlatform,
        OPENCLAW_TRAINING_RUNTIME: "axolotl",
        OPENCLAW_REVIEWED_EXPORT_REQUIRED: "true",
        OPENCLAW_RAW_CAPTURE_ALLOWED: "false",
      },
    };
  }
}

/**
 * A deterministic, dependency-free backend. It produces a self-contained plan
 * whose launch command is a plain `node` one-liner that writes a reproducible
 * model artifact — no GPU, python, or network required — so the full training
 * pipeline can be exercised in the cloud/CI. It also implements
 * {@link InProcessTrainingBackend} so tests can "train" in-process and assert
 * the artifact is stable across runs.
 */
export class MockTrainingBackend implements InProcessTrainingBackend {
  readonly id = "mock";

  supports(_job: LocalTrainingJobManifest): boolean {
    return true;
  }

  simulateTraining(job: LocalTrainingJobManifest): TrainedModelArtifact {
    const fingerprint = fingerprintJob(job);
    return {
      version: 1,
      backendId: this.id,
      jobId: job.id,
      mode: job.mode,
      modelFingerprint: fingerprint,
      weights: deriveWeights(job, fingerprint),
      datasetSignature: {
        promotedSkillCount: job.dataset.promotedSkillCount,
        memoryCount: job.dataset.memoryCount,
        trajectoryCount: job.dataset.trajectoryCount,
        replayCount: job.dataset.replayCount,
      },
    };
  }

  buildPlan(job: LocalTrainingJobManifest, execution: LocalTrainingExecution): TrainingJobPlan {
    const artifact = this.simulateTraining(job);
    const outputPath = path.posix.join(execution.artifactDir, "model.mock.json");
    return {
      version: 1,
      jobId: job.id,
      mode: job.mode,
      targetPlatform: "portable",
      runtime: "mock",
      datasetPath: execution.datasetDir,
      outputPath,
      replayEvalPath: execution.replayEvalFile,
      statePath: execution.stateFile,
      command: ["node", "-e", MOCK_TRAIN_PROGRAM],
      environment: {
        OPENCLAW_TRAINING_JOB_ID: job.id,
        OPENCLAW_TRAINING_MODE: job.mode,
        OPENCLAW_TARGET_PLATFORM: job.targetPlatform,
        OPENCLAW_TRAINING_RUNTIME: "mock",
        OPENCLAW_REVIEWED_EXPORT_REQUIRED: "true",
        OPENCLAW_RAW_CAPTURE_ALLOWED: "false",
        OPENCLAW_MOCK_OUTPUT_PATH: outputPath,
        OPENCLAW_MOCK_MODEL_ARTIFACT: JSON.stringify(artifact),
      },
    };
  }
}

/**
 * Node one-liner run as the mock backend's launch command. It reads the
 * deterministic artifact and target path from the environment and writes the
 * artifact to disk — no external dependencies, works anywhere node runs.
 */
const MOCK_TRAIN_PROGRAM = [
  "const fs=require('node:fs');",
  "const path=require('node:path');",
  "const out=process.env.OPENCLAW_MOCK_OUTPUT_PATH;",
  "const body=process.env.OPENCLAW_MOCK_MODEL_ARTIFACT;",
  "if(!out||!body){throw new Error('mock training env not set');}",
  "fs.mkdirSync(path.dirname(out),{recursive:true});",
  "fs.writeFileSync(out,body);",
].join("");

/**
 * A registry of pluggable training backends. Resolution prefers an explicit id,
 * then the first backend whose `supports()` returns true. The default registry
 * ships the real Apple-Silicon backend plus the deterministic mock.
 */
export class TrainingBackendRegistry {
  private readonly backends = new Map<string, TrainingBackend>();
  private readonly order: string[] = [];

  register(backend: TrainingBackend): this {
    if (!this.backends.has(backend.id)) {
      this.order.push(backend.id);
    }
    this.backends.set(backend.id, backend);
    return this;
  }

  get(id: string): TrainingBackend | undefined {
    return this.backends.get(id);
  }

  list(): TrainingBackend[] {
    return this.order.map((id) => this.backends.get(id)!).filter((backend): backend is TrainingBackend => Boolean(backend));
  }

  /**
   * Resolve a backend for the job. If `preferredId` is given it must exist and
   * support the job; otherwise the first registered backend that supports the
   * job wins. Throws when nothing matches so misconfiguration fails loudly.
   */
  resolve(job: LocalTrainingJobManifest, preferredId?: string): TrainingBackend {
    if (preferredId) {
      const backend = this.backends.get(preferredId);
      if (!backend) {
        throw new Error(`Unknown training backend "${preferredId}"`);
      }
      if (!backend.supports(job)) {
        throw new Error(`Training backend "${preferredId}" does not support job ${job.id}`);
      }
      return backend;
    }
    for (const id of this.order) {
      const backend = this.backends.get(id);
      if (backend?.supports(job)) {
        return backend;
      }
    }
    throw new Error(`No registered training backend supports job ${job.id}`);
  }
}

export function createDefaultTrainingBackendRegistry(): TrainingBackendRegistry {
  return new TrainingBackendRegistry()
    .register(new AppleSiliconTrainingBackend())
    .register(new MockTrainingBackend());
}

function fingerprintJob(job: LocalTrainingJobManifest): string {
  const stable = JSON.stringify({
    mode: job.mode,
    targetPlatform: job.targetPlatform,
    config: job.config,
    dataset: job.dataset,
    reviewedExportCreatedAt: job.reviewedExportCreatedAt,
    reviewedBy: job.reviewedBy,
    purpose: job.purpose,
  });
  return createHash("sha256").update(stable).digest("hex");
}

/**
 * Deterministically derive per-signal "weights" from the job. This is a stand-in
 * for real gradient updates: it is stable, input-sensitive, and normalized, so
 * tests can assert the mock generalizes (different datasets → different weights)
 * without any real training toolchain.
 */
function deriveWeights(job: LocalTrainingJobManifest, fingerprint: string): Record<string, number> {
  const signals: Record<string, number> = {
    promotedSkill: job.dataset.promotedSkillCount,
    memory: job.dataset.memoryCount,
    trajectory: job.dataset.trajectoryCount,
    replay: job.dataset.replayCount,
  };
  const total = Object.values(signals).reduce((sum, value) => sum + value, 0) || 1;
  const seed = parseInt(fingerprint.slice(0, 8), 16);
  const weights: Record<string, number> = {};
  for (const [key, value] of Object.entries(signals)) {
    // Blend the empirical frequency with a deterministic per-signal jitter so
    // identical counts across different jobs still yield distinct fingerprints.
    const jitter = ((seed % 97) + key.length) / 1000;
    weights[key] = Number((value / total + jitter).toFixed(6));
  }
  return weights;
}
