import fs from "node:fs/promises";
import path from "node:path";
import { spawn } from "node:child_process";
import { ensureParentDir, readJsonFile, writeJsonAtomic } from "../shared/fs.js";
import type { LocalTrainingJobManifest } from "./job-manifest.js";

export type TrainingExecutionState = {
  version: 1;
  jobId: string;
  status: "running" | "completed" | "failed";
  pid: number;
  startedAt: string;
  updatedAt: string;
  completedAt?: string;
  exitCode?: number;
  signal?: string;
  error?: string;
  logFile: string;
  workingDirectory: string;
  command: string[];
};

export type ReadTrainingJobLogOptions = {
  lineLimit?: number;
};

export type SpawnProcess = (command: string, args: string[], options: {
  cwd: string;
  env: NodeJS.ProcessEnv;
  stdio: "ignore";
  detached: true;
}) => { pid?: number; unref(): void };

export class LocalTrainingExecutionService {
  constructor(
    private readonly rootDir: string,
    private readonly spawnProcess: SpawnProcess = (command, args, options) => spawn(command, args, options),
  ) {}

  async launch(job: LocalTrainingJobManifest): Promise<{ pid: number }> {
    if (!job.execution?.command?.length) {
      throw new Error(`Training job ${job.id} is not prepared`);
    }

    const launchScriptPath = path.join(this.rootDir, job.execution.launchScript);
    const child = this.spawnProcess(launchScriptPath, [], {
      cwd: this.rootDir,
      env: {
        ...process.env,
        ...job.execution.environment,
      },
      stdio: "ignore",
      detached: true,
    });

    child.unref();

    if (typeof child.pid !== "number") {
      throw new Error(`Failed to launch training job ${job.id}`);
    }

    return { pid: child.pid };
  }

  async writeState(job: LocalTrainingJobManifest, state: TrainingExecutionState): Promise<void> {
    if (!job.execution) {
      throw new Error(`Training job ${job.id} has no execution state`);
    }
    await writeJsonAtomic(path.join(this.rootDir, job.execution.stateFile), state);
  }

  async writeLog(job: LocalTrainingJobManifest, content: string): Promise<void> {
    if (!job.execution) {
      throw new Error(`Training job ${job.id} has no execution log`);
    }
    const logPath = path.join(this.rootDir, job.execution.logFile);
    await ensureParentDir(logPath);
    await fs.writeFile(logPath, content, "utf8");
  }

  async readState(job: LocalTrainingJobManifest): Promise<TrainingExecutionState | undefined> {
    if (!job.execution) {
      return undefined;
    }

    return await readJsonFile<TrainingExecutionState | undefined>(
      path.join(this.rootDir, job.execution.stateFile),
      undefined,
    );
  }

  async readLog(
    job: LocalTrainingJobManifest,
    options: ReadTrainingJobLogOptions = {},
  ): Promise<string | undefined> {
    if (!job.execution) {
      return undefined;
    }

    try {
      const content = await fs.readFile(path.join(this.rootDir, job.execution.logFile), "utf8");
      const lineLimit = options.lineLimit ?? 200;
      if (lineLimit <= 0) {
        return "";
      }
      const lines = content.split("\n");
      const trimmed = lines.at(-1) === "" ? lines.slice(0, -1) : lines;
      return trimmed.slice(-lineLimit).join("\n");
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") {
        return undefined;
      }
      throw error;
    }
  }
}
