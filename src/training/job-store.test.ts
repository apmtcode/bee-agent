import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { FileTrainingJobStore } from "./job-store.js";
import { LocalAppleSiliconTrainingRunner } from "./runner.js";
import type { TrainingExecutionState } from "./execution-service.js";
import type { ReviewedExportManifest } from "./export-manifest.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "training-jobs-"));
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
  executableSkills: [],
  executableSkillRuns: [],
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

describe("FileTrainingJobStore", () => {
  it("creates, prepares, runs, completes, and reloads training jobs", async () => {
    const filePath = path.join(await makeTempDir(), "training-jobs.json");
    const store = new FileTrainingJobStore(filePath);

    const sft = await store.create({ exportManifest, mode: "sft" });
    const rl = await store.create({ exportManifest, mode: "rl" });

    expect(sft.mode).toBe("sft");
    expect(rl.mode).toBe("rl");
    await expect(store.list()).resolves.toHaveLength(2);
    await expect(store.get(sft.id)).resolves.toMatchObject({ id: sft.id, status: "planned" });
    await expect(store.updateStatus(sft.id, "ready")).resolves.toMatchObject({ id: sft.id, status: "ready" });
    await expect(store.prepareExecution(sft.id)).resolves.toMatchObject({
      id: sft.id,
      status: "ready",
      execution: {
        workingDirectory: `training-jobs/${sft.id}`,
        datasetDir: `training-jobs/${sft.id}/dataset`,
        artifactDir: `training-jobs/${sft.id}/artifacts`,
        logFile: `training-jobs/${sft.id}/logs/sft.log`,
        stateFile: `training-jobs/${sft.id}/state.json`,
        planFile: `training-jobs/${sft.id}/plan.json`,
        launchScript: `training-jobs/${sft.id}/run-sft.sh`,
        replayEvalFile: `training-jobs/${sft.id}/replay-eval.json`,
        command: [
          "python3",
          "-m",
          "mlx_lm.lora",
          "--train",
          "--data",
          `training-jobs/${sft.id}/dataset`,
          "--adapter-path",
          `training-jobs/${sft.id}/artifacts`,
          "--learning-rate",
          "0.00001",
          "--batch-size",
          "1",
          "--iters",
          "3000",
        ],
        environment: {
          OPENCLAW_TRAINING_RUNTIME: "mlx",
          OPENCLAW_REVIEWED_EXPORT_REQUIRED: "true",
        },
        steps: [
          { id: "prepare-dataset", status: "pending" },
          { id: "launch-training", status: "pending" },
          { id: "collect-artifacts", status: "pending" },
          { id: "evaluate-replay", status: "pending" },
        ],
      },
    });
    await expect(store.markExecutionStarted(sft.id, { pid: 1234 })).resolves.toMatchObject({
      id: sft.id,
      status: "running",
      execution: {
        startedAt: expect.any(String),
        processId: 1234,
        steps: [
          { id: "prepare-dataset", status: "completed" },
          { id: "launch-training", status: "running" },
          { id: "collect-artifacts", status: "pending" },
          { id: "evaluate-replay", status: "pending" },
        ],
      },
    });
    await expect(store.completeExecution(sft.id, { outputModelRef: "artifacts/models/sft.gguf" })).resolves.toMatchObject({
      id: sft.id,
      status: "completed",
      execution: {
        completedAt: expect.any(String),
        outputModelRef: "artifacts/models/sft.gguf",
        steps: [
          { id: "prepare-dataset", status: "completed" },
          { id: "launch-training", status: "completed" },
          { id: "collect-artifacts", status: "completed" },
          { id: "evaluate-replay", status: "completed" },
        ],
      },
    });
    await expect(store.failExecution(rl.id, { reason: "trainer crashed" })).resolves.toMatchObject({
      id: rl.id,
      status: "failed",
      execution: {
        failedAt: expect.any(String),
        failureReason: "trainer crashed",
        logFile: `training-jobs/${rl.id}/logs/rl.log`,
        steps: [
          { id: "prepare-dataset", status: "running" },
          { id: "launch-training", status: "failed" },
          { id: "collect-artifacts", status: "pending" },
          { id: "evaluate-replay", status: "pending" },
        ],
      },
    });

    const reloaded = new FileTrainingJobStore(filePath);
    await expect(reloaded.get(sft.id)).resolves.toMatchObject({
      id: sft.id,
      status: "completed",
      execution: { outputModelRef: "artifacts/models/sft.gguf" },
    });
    await expect(reloaded.getExecutionPlan(sft.id)).resolves.toMatchObject({
      jobId: sft.id,
      runtime: "mlx",
      outputPath: `training-jobs/${sft.id}/artifacts/model.gguf`,
    });
    await expect(reloaded.getLaunchScript(sft.id)).resolves.toContain("mlx_lm.lora");
    await expect(reloaded.get(rl.id)).resolves.toMatchObject({
      id: rl.id,
      status: "failed",
      execution: { failureReason: "trainer crashed" },
    });
  });

  it("shares runner-compatible artifacts with the standalone runner", async () => {
    const filePath = path.join(await makeTempDir(), "training-jobs.json");
    const store = new FileTrainingJobStore(filePath);
    const runner = new LocalAppleSiliconTrainingRunner(path.dirname(filePath));
    const job = await store.create({ exportManifest, mode: "rl" });
    const prepared = await store.prepareExecution(job.id);

    expect(prepared).toBeDefined();
    if (!prepared?.execution) {
      throw new Error("expected prepared execution");
    }

    await expect(runner.readPlan(prepared)).resolves.toMatchObject({
      jobId: job.id,
      runtime: "axolotl",
      replayEvalPath: `training-jobs/${job.id}/replay-eval.json`,
      statePath: `training-jobs/${job.id}/state.json`,
    });
    await expect(runner.readLaunchScript(prepared)).resolves.toContain("axolotl.cli.train");
  });

  it("reads persisted execution state and log output", async () => {
    const rootDir = await makeTempDir();
    const filePath = path.join(rootDir, "training-jobs.json");
    const store = new FileTrainingJobStore(filePath);
    const job = await store.create({ exportManifest, mode: "sft" });
    const prepared = await store.prepareExecution(job.id);

    if (!prepared?.execution) {
      throw new Error("expected prepared execution");
    }

    const state: TrainingExecutionState = {
      version: 1,
      jobId: job.id,
      status: "completed",
      pid: 4321,
      startedAt: "2026-01-01T00:00:00.000Z",
      updatedAt: "2026-01-01T00:10:00.000Z",
      completedAt: "2026-01-01T00:10:00.000Z",
      exitCode: 0,
      logFile: prepared.execution.logFile,
      workingDirectory: prepared.execution.workingDirectory,
      command: prepared.execution.command ?? [],
    };
    await store.executionService.writeState(prepared, state);
    await store.executionService.writeLog(prepared, ["line-1", "line-2", "line-3"].join("\n") + "\n");

    await expect(store.getExecutionState(job.id)).resolves.toMatchObject({
      jobId: job.id,
      status: "completed",
      pid: 4321,
    });
    await expect(store.getExecutionLog(job.id, 2)).resolves.toBe("line-2\nline-3");
    await expect(store.syncExecution(job.id)).resolves.toMatchObject({
      id: job.id,
      status: "completed",
      execution: {
        outputModelRef: `training-jobs/${job.id}/artifacts/model.gguf`,
      },
    });
  });
});
