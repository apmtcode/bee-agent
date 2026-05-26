import os from "node:os";
import fs from "node:fs/promises";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { createLocalTrainingExecution, createLocalTrainingJobManifest } from "./job-manifest.js";
import { LocalTrainingExecutionService, type TrainingExecutionState } from "./execution-service.js";
import type { ReviewedExportManifest } from "./export-manifest.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "training-execution-service-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })));
});

const exportManifest: ReviewedExportManifest = {
  version: 1,
  createdAt: "2026-01-01T00:00:00.000Z",
  reviewedBy: "operator",
  purpose: "local fine-tuning",
  targetPlatform: "apple-silicon",
  modes: ["sft", "rl"],
  rawCaptureIncluded: false,
  promotedSkills: [
    {
      id: "skill-1",
      title: "Deploy workflow",
      summary: "Deploy workflow repaired and ready to reuse",
      sourceCandidateId: "candidate-1",
      sourceTrajectoryIds: ["traj-1"],
      promotedAt: "2026-01-01T00:10:00.000Z",
      version: 1,
    },
  ],
  memories: [
    {
      id: "memory-1",
      type: "session-summary",
      summary: "Deploy workflow repaired and ready to reuse",
      sourceSessionId: "sess-1",
      sourceTrajectoryId: "traj-1",
      tags: ["deploy"],
    },
  ],
  trajectories: [
    {
      id: "traj-1",
      sessionId: "sess-1",
      createdAt: "2026-01-01T00:00:00.000Z",
      captureTier: "operator",
      observationCount: 1,
      actionCount: 1,
      outcomeStatus: "success",
      reward: 1,
    },
  ],
  replays: [
    {
      sessionId: "sess-1",
      trajectoryIds: ["traj-1"],
      eventCount: 2,
      events: [
        { kind: "observation", ts: 1, trajectoryId: "traj-1", source: "browser", summary: "opened deploy" },
        { kind: "action", ts: 2, trajectoryId: "traj-1", tool: "browser", summary: "clicked deploy" },
      ],
    },
  ],
};

describe("LocalTrainingExecutionService", () => {
  it("launches a prepared training script through the injected spawner", async () => {
    const rootDir = await makeTempDir();
    const spawned: Array<{ command: string; args: string[]; cwd: string }> = [];
    const service = new LocalTrainingExecutionService(rootDir, (command, args, options) => {
      spawned.push({ command, args, cwd: options.cwd });
      return {
        pid: 4242,
        unref() {},
      };
    });
    const job = createLocalTrainingJobManifest({ id: "job-sft", exportManifest, mode: "sft" });
    const execution = createLocalTrainingExecution({ jobId: job.id, mode: job.mode });
    const preparedJob = {
      ...job,
      execution: {
        ...execution,
        command: ["python3", "-m", "mlx_lm.lora"],
        environment: { OPENCLAW_TRAINING_RUNTIME: "mlx" },
      },
    };

    await expect(service.launch(preparedJob)).resolves.toEqual({ pid: 4242 });
    expect(spawned).toEqual([
      {
        command: path.join(rootDir, execution.launchScript),
        args: [],
        cwd: rootDir,
      },
    ]);
  });

  it("writes and reads persisted state and bounded logs", async () => {
    const rootDir = await makeTempDir();
    const service = new LocalTrainingExecutionService(rootDir);
    const job = createLocalTrainingJobManifest({ id: "job-sft", exportManifest, mode: "sft" });
    const execution = createLocalTrainingExecution({ jobId: job.id, mode: job.mode });
    const preparedJob = { ...job, execution };
    const state: TrainingExecutionState = {
      version: 1,
      jobId: job.id,
      status: "completed",
      pid: 4242,
      startedAt: "2026-01-01T00:00:00.000Z",
      updatedAt: "2026-01-01T00:10:00.000Z",
      completedAt: "2026-01-01T00:10:00.000Z",
      exitCode: 0,
      logFile: execution.logFile,
      workingDirectory: execution.workingDirectory,
      command: ["python3", "-m", "mlx_lm.lora"],
    };

    await service.writeState(preparedJob, state);
    await service.writeLog(preparedJob, "line-1\nline-2\nline-3\n");

    await expect(service.readState(preparedJob)).resolves.toMatchObject({
      jobId: job.id,
      status: "completed",
      pid: 4242,
    });
    await expect(service.readLog(preparedJob, { lineLimit: 2 })).resolves.toBe("line-2\nline-3");
  });
});
