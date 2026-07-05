import path from "node:path";
import type { LocalTrainingExecution, LocalTrainingJobManifest, RlTrainingConfig, SftTrainingConfig } from "./job-manifest.js";

/**
 * The runtime label for a training backend. `mlx`/`axolotl` drive real
 * on-device Apple Silicon training; `mock` is a deterministic, fully portable
 * backend that runs in the cloud/CI to validate the train → infer pipeline
 * end-to-end without any ML toolchain.
 */
export type LocalTrainingRuntime = "mlx" | "axolotl" | "mock";

/**
 * The runtime-specific slice of a training plan. A {@link TrainingBackend}
 * produces this from a job + its prepared execution layout; the runner then
 * composes it into the full, persisted {@link TrainingJobPlan}. Keeping this
 * narrow is what makes the backend pluggable — a backend only decides *how* the
 * model is trained (runtime, launch command, artifact name, extra env), not the
 * surrounding bookkeeping.
 */
export type TrainingBackendExecutionPlan = {
  runtime: LocalTrainingRuntime;
  /** Filename (within the execution's artifact dir) the trained model is written to. */
  outputArtifact: string;
  /** The command (argv) the launch script executes to run training. */
  command: string[];
  /** Backend-specific environment variables merged on top of the standard block. */
  extraEnvironment?: Record<string, string>;
};

/**
 * Pluggable training backend. Implementations map a reviewed export + prepared
 * execution layout onto a concrete launch command. The default
 * {@link AppleSiliconTrainingBackend} targets on-device Apple Silicon training;
 * {@link MockTrainingBackend} is a deterministic stand-in for cloud/CI.
 */
export interface TrainingBackend {
  /** Stable identifier, e.g. for logging / diagnostics. */
  readonly name: string;
  buildExecutionPlan(job: LocalTrainingJobManifest, execution: LocalTrainingExecution): TrainingBackendExecutionPlan;
}

/**
 * On-device Apple Silicon backend: SFT via `mlx_lm.lora`, RL via
 * `axolotl.cli.train` with a replay-manifest reward. This preserves the exact
 * behaviour the runner had before the backend seam was introduced.
 */
export class AppleSiliconTrainingBackend implements TrainingBackend {
  readonly name = "apple-silicon";

  buildExecutionPlan(job: LocalTrainingJobManifest, execution: LocalTrainingExecution): TrainingBackendExecutionPlan {
    if (job.mode === "sft") {
      const config = job.config as SftTrainingConfig;
      return {
        runtime: "mlx",
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
      };
    }

    const config = job.config as RlTrainingConfig;
    return {
      runtime: "axolotl",
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
    };
  }
}

/**
 * Deterministic mock backend. Its launch command is a self-contained Node
 * script (no Python, no ML libraries) that reads the reviewed dataset manifest,
 * derives a stable digest, and writes a stub "model" artifact plus a
 * replay-eval result. This lets the cloud engine and CI exercise the full
 * prepare → launch → produce-artifact → read-state pipeline for real, with a
 * predictable, reproducible output keyed only to the dataset contents.
 */
export class MockTrainingBackend implements TrainingBackend {
  readonly name = "mock";

  buildExecutionPlan(job: LocalTrainingJobManifest, execution: LocalTrainingExecution): TrainingBackendExecutionPlan {
    const outputArtifact = "model.mock.json";
    // Paths are relative to the runner root dir, which is also the launch cwd.
    const datasetManifest = path.posix.join(execution.datasetDir, "manifest.json");
    const outputPath = path.posix.join(execution.artifactDir, outputArtifact);
    const evalResultPath = path.posix.join(execution.artifactDir, "replay-eval-result.json");

    return {
      runtime: "mock",
      outputArtifact,
      command: ["node", "-e", MOCK_TRAINING_SCRIPT],
      extraEnvironment: {
        OPENCLAW_MOCK_DATASET: datasetManifest,
        OPENCLAW_MOCK_OUTPUT: outputPath,
        OPENCLAW_MOCK_REPLAY_EVAL: evalResultPath,
        OPENCLAW_MOCK_MODE: job.mode,
      },
    };
  }
}

/**
 * The inline Node program run by {@link MockTrainingBackend}. Deterministic:
 * the produced artifact/eval depend only on the reviewed dataset manifest, so
 * identical inputs yield byte-identical outputs. Kept dependency-free so it runs
 * anywhere Node runs.
 */
export const MOCK_TRAINING_SCRIPT = [
  'const fs=require("node:fs");',
  'const path=require("node:path");',
  'const crypto=require("node:crypto");',
  'const datasetPath=process.env.OPENCLAW_MOCK_DATASET;',
  'const outputPath=process.env.OPENCLAW_MOCK_OUTPUT;',
  'const evalPath=process.env.OPENCLAW_MOCK_REPLAY_EVAL;',
  'const manifest=JSON.parse(fs.readFileSync(datasetPath,"utf8"));',
  'const dataset=manifest.dataset||{};',
  'const digest=crypto.createHash("sha256").update(JSON.stringify(dataset)).digest("hex").slice(0,16);',
  'fs.mkdirSync(path.dirname(outputPath),{recursive:true});',
  'fs.writeFileSync(outputPath,JSON.stringify({version:1,backend:"mock",jobId:manifest.jobId,mode:manifest.mode,datasetDigest:digest,weights:"deterministic-stub"},null,2)+"\\n");',
  'fs.mkdirSync(path.dirname(evalPath),{recursive:true});',
  'fs.writeFileSync(evalPath,JSON.stringify({version:1,backend:"mock",jobId:manifest.jobId,mode:manifest.mode,replayCount:dataset.replayCount||0,passed:true,datasetDigest:digest},null,2)+"\\n");',
  'process.stdout.write("mock training complete for "+manifest.jobId+" ("+digest+")\\n");',
].join("");
