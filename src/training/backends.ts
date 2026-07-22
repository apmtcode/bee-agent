import path from "node:path";
import { ensureParentDir, writeJsonAtomic } from "../shared/fs.js";
import type { TrainingMode } from "./export-manifest.js";
import type {
  LocalTrainingExecution,
  LocalTrainingJobManifest,
  RlTrainingConfig,
  SftTrainingConfig,
} from "./job-manifest.js";

/**
 * Pluggable training-backend seam.
 *
 * The runner no longer hardcodes the mlx/axolotl runtimes. A `TrainingBackend`
 * turns a (job, execution) pair into the runtime-specific part of a training
 * plan — the command to run, the artifact it produces, and any extra
 * environment. This lets bee-agent swap in different local-model backends
 * (a small on-device model, a remote trainer, or the dependency-free
 * `SimulatedTrainingBackend` used to validate the pipeline in the cloud)
 * without touching plan assembly, launch-script rendering, or the job store.
 */
export type TrainingBackendContext = {
  job: LocalTrainingJobManifest;
  execution: LocalTrainingExecution;
};

export type TrainingBackendPlan = {
  /** Identifier persisted into the plan + `OPENCLAW_TRAINING_RUNTIME`. */
  runtime: string;
  /** Platform label recorded in the plan (e.g. "apple-silicon", "simulated"). */
  targetPlatform: string;
  /** Basename of the model artifact produced under `execution.artifactDir`. */
  outputFileName: string;
  /** Executable command (argv) the launch script runs. */
  command: string[];
  /** Backend-specific environment merged into the shared training env. */
  extraEnvironment?: Record<string, string>;
};

export interface TrainingBackend {
  readonly id: string;
  readonly runtime: string;
  planExecution(context: TrainingBackendContext): TrainingBackendPlan;
}

export type TrainingSimulationResult = {
  jobId: string;
  mode: TrainingMode;
  runtime: string;
  artifactPath: string;
  statePath: string;
  fingerprint: string;
};

/**
 * A backend that can additionally run its "training" fully in-process, with no
 * external toolchain. Used so the capture→dataset→train→artifact loop can be
 * exercised deterministically in cloud/CI where Apple Silicon + mlx/axolotl are
 * unavailable.
 */
export interface SimulatableTrainingBackend extends TrainingBackend {
  simulate(rootDir: string, context: TrainingBackendContext): Promise<TrainingSimulationResult>;
}

export type TrainingBackendRegistry = Record<TrainingMode, TrainingBackend>;

/** MLX LoRA SFT backend for local Apple Silicon (the original behaviour). */
export class MlxSftBackend implements TrainingBackend {
  readonly id = "mlx-sft";
  readonly runtime = "mlx";

  planExecution({ job, execution }: TrainingBackendContext): TrainingBackendPlan {
    const config = job.config as SftTrainingConfig;
    return {
      runtime: this.runtime,
      targetPlatform: "apple-silicon",
      outputFileName: "model.gguf",
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
    };
  }
}

/** Axolotl GRPO RL backend rewarded by the replay manifest (original behaviour). */
export class AxolotlRlBackend implements TrainingBackend {
  readonly id = "axolotl-rl";
  readonly runtime = "axolotl";

  planExecution({ job, execution }: TrainingBackendContext): TrainingBackendPlan {
    const config = job.config as RlTrainingConfig;
    return {
      runtime: this.runtime,
      targetPlatform: "apple-silicon",
      outputFileName: "policy.gguf",
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
    };
  }
}

/**
 * Deterministic, dependency-free backend. Its plan runs a self-contained
 * `node` program (no python/mlx/axolotl), and `simulate()` performs the same
 * deterministic "training" in-process: it derives a stable fingerprint from the
 * reviewed dataset shape + config and writes a model artifact plus a completed
 * state file. Identical inputs always yield an identical fingerprint, so the
 * full pipeline can be asserted in cloud/CI.
 */
export class SimulatedTrainingBackend implements SimulatableTrainingBackend {
  readonly id = "simulated";
  readonly runtime = "simulated";

  planExecution({ job, execution }: TrainingBackendContext): TrainingBackendPlan {
    const outputFileName = job.mode === "sft" ? "model.sim.json" : "policy.sim.json";
    const artifactPath = path.posix.join(execution.artifactDir, outputFileName);
    return {
      runtime: this.runtime,
      targetPlatform: "simulated",
      outputFileName,
      command: ["node", "-e", renderSimulationProgram(), job.id, job.mode, artifactPath, canonicalTrainingInput(job)],
      extraEnvironment: {
        OPENCLAW_TRAINING_SIMULATED: "true",
      },
    };
  }

  async simulate(rootDir: string, { job, execution }: TrainingBackendContext): Promise<TrainingSimulationResult> {
    const plan = this.planExecution({ job, execution });
    const artifactPath = path.join(rootDir, execution.artifactDir, plan.outputFileName);
    const statePath = path.join(rootDir, execution.stateFile);
    const canonicalInput = canonicalTrainingInput(job);
    const fingerprint = fingerprintTrainingInput(canonicalInput);
    const trainedAt = job.reviewedExportCreatedAt;

    await ensureParentDir(artifactPath);
    await writeJsonAtomic(artifactPath, {
      version: 1,
      jobId: job.id,
      mode: job.mode,
      runtime: this.runtime,
      fingerprint,
      trainedOn: job.dataset,
      config: job.config,
      trainedAt,
    });

    await writeJsonAtomic(statePath, {
      version: 1,
      jobId: job.id,
      status: "completed",
      runtime: this.runtime,
      outputModelRef: path.posix.join(execution.artifactDir, plan.outputFileName),
      fingerprint,
      startedAt: trainedAt,
      updatedAt: trainedAt,
      completedAt: trainedAt,
      exitCode: 0,
      error: null,
    });

    return {
      jobId: job.id,
      mode: job.mode,
      runtime: this.runtime,
      artifactPath,
      statePath,
      fingerprint,
    };
  }
}

/** Default production registry: mlx for SFT, axolotl for RL (unchanged). */
export function createDefaultTrainingBackends(): TrainingBackendRegistry {
  return {
    sft: new MlxSftBackend(),
    rl: new AxolotlRlBackend(),
  };
}

/** Deterministic registry for cloud/CI: both modes use the simulated backend. */
export function createSimulatedTrainingBackends(): TrainingBackendRegistry {
  const backend = new SimulatedTrainingBackend();
  return { sft: backend, rl: backend };
}

/**
 * Canonical, stable-key serialization of the training inputs that determine the
 * model. Insertion order is fixed so the fingerprint is reproducible.
 */
export function canonicalTrainingInput(job: LocalTrainingJobManifest): string {
  return JSON.stringify({
    mode: job.mode,
    dataset: {
      promotedSkillCount: job.dataset.promotedSkillCount,
      memoryCount: job.dataset.memoryCount,
      trajectoryCount: job.dataset.trajectoryCount,
      replayCount: job.dataset.replayCount,
    },
    config: job.config,
  });
}

/** Deterministic djb2 fingerprint (hex) shared by `simulate()` and the plan program. */
export function fingerprintTrainingInput(canonicalInput: string): string {
  let hash = 5381;
  for (let index = 0; index < canonicalInput.length; index += 1) {
    hash = ((hash << 5) + hash + canonicalInput.charCodeAt(index)) >>> 0;
  }
  return hash.toString(16).padStart(8, "0");
}

/**
 * A self-contained node program (passed via `node -e`) mirroring `simulate()`.
 * It lets a real launch script produce the same artifact in environments
 * without any ML toolchain. Kept in sync with `fingerprintTrainingInput`.
 */
function renderSimulationProgram(): string {
  return [
    "const fs=require('fs'),path=require('path');",
    "const[jobId,mode,artifactPath,canonical]=process.argv.slice(1);",
    "let h=5381;for(let i=0;i<canonical.length;i++){h=((h<<5)+h+canonical.charCodeAt(i))>>>0;}",
    "const fingerprint=h.toString(16).padStart(8,'0');",
    "fs.mkdirSync(path.dirname(artifactPath),{recursive:true});",
    "const input=JSON.parse(canonical);",
    "fs.writeFileSync(artifactPath,JSON.stringify({version:1,jobId,mode,runtime:'simulated',fingerprint,trainedOn:input.dataset,config:input.config},null,2)+'\\n');",
  ].join("");
}
