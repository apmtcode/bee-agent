import fs from "node:fs/promises";
import path from "node:path";
import { ensureParentDir, readJsonFile, writeJsonAtomic } from "../shared/fs.js";
import type { TrainingExecutionState } from "./execution-service.js";
import type {
  LocalTrainingExecution,
  LocalTrainingJobManifest,
  RlTrainingConfig,
  SftTrainingConfig,
} from "./job-manifest.js";

export type LocalTrainingRuntime = "mlx" | "axolotl";

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

export class LocalAppleSiliconTrainingRunner {
  constructor(private readonly rootDir: string) {}

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

  async writeArtifacts(job: LocalTrainingJobManifest, execution: LocalTrainingExecution): Promise<TrainingJobPlan> {
    const plan = this.buildPlan(job, execution);
    const planPath = path.join(this.rootDir, execution.planFile);
    const launchScriptPath = path.join(this.rootDir, execution.launchScript);
    const replayEvalPath = path.join(this.rootDir, execution.replayEvalFile);
    const datasetManifestPath = path.join(this.rootDir, execution.datasetDir, "manifest.json");

    await Promise.all([
      writeJsonAtomic(planPath, plan),
      writeJsonAtomic(replayEvalPath, {
        version: 1,
        jobId: job.id,
        mode: job.mode,
        replayCount: job.dataset.replayCount,
        required: true,
        source: "reviewed-export",
      }),
      writeJsonAtomic(datasetManifestPath, {
        version: 1,
        jobId: job.id,
        mode: job.mode,
        dataset: job.dataset,
        reviewedExportCreatedAt: job.reviewedExportCreatedAt,
        reviewedBy: job.reviewedBy,
        purpose: job.purpose,
      }),
    ]);

    await ensureParentDir(launchScriptPath);
    await fs.writeFile(launchScriptPath, renderLaunchScript(execution, plan), {
      encoding: "utf8",
      mode: 0o700,
    });

    return plan;
  }

  async readPlan(job: LocalTrainingJobManifest): Promise<TrainingJobPlan | undefined> {
    if (!job.execution) {
      return undefined;
    }
    return await readJsonFile<TrainingJobPlan | undefined>(path.join(this.rootDir, job.execution.planFile), undefined);
  }

  async readLaunchScript(job: LocalTrainingJobManifest): Promise<string | undefined> {
    if (!job.execution) {
      return undefined;
    }
    try {
      return await fs.readFile(path.join(this.rootDir, job.execution.launchScript), "utf8");
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") {
        return undefined;
      }
      throw error;
    }
  }
}

function renderLaunchScript(execution: LocalTrainingExecution, plan: TrainingJobPlan): string {
  const quotedStatePath = shellQuote(execution.stateFile);
  const quotedLogFile = shellQuote(execution.logFile);
  const quotedWorkingDirectory = shellQuote(execution.workingDirectory);
  const quotedCommand = `${shellQuote(plan.command[0] ?? "")}${plan.command.slice(1).map((arg) => ` ${shellQuote(arg)}`).join("")}`;
  // Write the base state verbatim (no shell string-templating of JSON — a
  // command arg containing quotes would otherwise corrupt the file), then let
  // python fill in the live pid + timestamps.
  const quotedStatePayload = shellQuote(
    JSON.stringify({
      version: 1,
      jobId: plan.jobId,
      status: "running",
      pid: 0,
      startedAt: "",
      updatedAt: "",
      logFile: execution.logFile,
      workingDirectory: execution.workingDirectory,
      command: plan.command,
    }),
  );

  return [
    "#!/usr/bin/env bash",
    "set -euo pipefail",
    `state_path=${quotedStatePath}`,
    `mkdir -p ${shellQuote(execution.artifactDir)} $(dirname ${quotedLogFile}) $(dirname "$state_path")`,
    "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    // Base state to a private ".base" temp; python publishes atomically so a
    // concurrent reader never observes a partially written state file.
    `printf '%s' ${quotedStatePayload} > "$state_path.base"`,
    `python3 - "$state_path" "$state_path.base" $$ "$started_at" <<'PY'`,
    ...renderInitStateWriterPython(),
    "PY",
    `printf '%s\n' "starting ${plan.mode} training for ${plan.jobId}" >> ${quotedLogFile}`,
    `if ${quotedCommand} >> ${quotedLogFile} 2>&1; then`,
    "  completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    `  python3 - "$state_path" $$ "$completed_at" 0 <<'PY'`,
    ...renderStateWriterPython("completed"),
    "PY",
    "else",
    "  exit_code=$?",
    "  completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    `  python3 - "$state_path" $$ "$completed_at" "$exit_code" <<'PY'`,
    ...renderStateWriterPython("failed"),
    "PY",
    '  exit "$exit_code"',
    "fi",
    "",
  ].join("\n");
}

function renderInitStateWriterPython(): string[] {
  return [
    "import json",
    "import os",
    "import pathlib",
    "import sys",
    "final_path = sys.argv[1]",
    "base_path = pathlib.Path(sys.argv[2])",
    "pid = int(sys.argv[3])",
    "timestamp = sys.argv[4]",
    "state = json.loads(base_path.read_text())",
    "state['pid'] = pid",
    "state['startedAt'] = timestamp",
    "state['updatedAt'] = timestamp",
    "tmp_path = final_path + '.tmp'",
    "pathlib.Path(tmp_path).write_text(json.dumps(state, indent=2) + '\\n')",
    "os.replace(tmp_path, final_path)",
    "base_path.unlink(missing_ok=True)",
  ];
}

function renderStateWriterPython(status: TrainingExecutionState["status"]): string[] {
  return [
    "import json",
    "import os",
    "import pathlib",
    "import sys",
    "final_path = sys.argv[1]",
    "pid = int(sys.argv[2])",
    "timestamp = sys.argv[3]",
    "exit_code = int(sys.argv[4])",
    "state = json.loads(pathlib.Path(final_path).read_text())",
    `state['status'] = '${status}'`,
    "state['pid'] = pid",
    "state['updatedAt'] = timestamp",
    "state['completedAt'] = timestamp",
    "state['exitCode'] = exit_code",
    `state['error'] = None if '${status}' == 'completed' else 'training process exited non-zero'`,
    "tmp_path = final_path + '.tmp'",
    "pathlib.Path(tmp_path).write_text(json.dumps(state, indent=2) + '\\n')",
    "os.replace(tmp_path, final_path)",
  ];
}

function shellQuote(value: string): string {
  return `'${value.replaceAll(`'`, `'"'"'`)}'`;
}
