import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { createLocalTrainingExecution, createLocalTrainingJobManifest } from "./job-manifest.js";
import { LocalAppleSiliconTrainingRunner } from "./runner.js";
import type { ReviewedExportManifest } from "./export-manifest.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "training-runner-"));
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

describe("LocalAppleSiliconTrainingRunner", () => {
  it("builds and persists an SFT runner plan", async () => {
    const rootDir = await makeTempDir();
    const runner = new LocalAppleSiliconTrainingRunner(rootDir);
    const job = createLocalTrainingJobManifest({ id: "job-sft", exportManifest, mode: "sft" });
    const execution = createLocalTrainingExecution({ jobId: job.id, mode: job.mode });

    const plan = await runner.writeArtifacts(job, execution);
    expect(plan).toMatchObject({
      jobId: job.id,
      mode: "sft",
      runtime: "mlx",
      datasetPath: `training-jobs/${job.id}/dataset`,
      outputPath: `training-jobs/${job.id}/artifacts/model.gguf`,
      statePath: `training-jobs/${job.id}/state.json`,
      command: [
        "python3",
        "-m",
        "mlx_lm.lora",
        "--train",
        "--data",
        `training-jobs/${job.id}/dataset`,
        "--adapter-path",
        `training-jobs/${job.id}/artifacts`,
        "--learning-rate",
        "0.00001",
        "--batch-size",
        "1",
        "--iters",
        "3000",
      ],
    });

    await expect(runner.readPlan({ ...job, execution })).resolves.toMatchObject({
      jobId: job.id,
      runtime: "mlx",
      replayEvalPath: `training-jobs/${job.id}/replay-eval.json`,
    });
    await expect(runner.readLaunchScript({ ...job, execution })).resolves.toEqual(
      expect.stringContaining(`python3 - '${execution.stateFile}'`),
    );
    await expect(runner.readLaunchScript({ ...job, execution })).resolves.toContain("mlx_lm.lora");
    await expect(fs.readFile(path.join(rootDir, execution.datasetDir, "manifest.json"), "utf8")).resolves.toContain(
      '"reviewedBy": "operator"',
    );
  });

  it("builds an RL runner plan with replay-backed reward inputs", async () => {
    const rootDir = await makeTempDir();
    const runner = new LocalAppleSiliconTrainingRunner(rootDir);
    const job = createLocalTrainingJobManifest({ id: "job-rl", exportManifest, mode: "rl" });
    const execution = createLocalTrainingExecution({ jobId: job.id, mode: job.mode });

    const plan = runner.buildPlan(job, execution);
    expect(plan).toMatchObject({
      jobId: job.id,
      mode: "rl",
      runtime: "axolotl",
      outputPath: `training-jobs/${job.id}/artifacts/policy.gguf`,
      replayEvalPath: `training-jobs/${job.id}/replay-eval.json`,
      statePath: `training-jobs/${job.id}/state.json`,
      command: [
        "python3",
        "-m",
        "axolotl.cli.train",
        `training-jobs/${job.id}/plan.json`,
        "--reward-model",
        "replay-manifest",
        "--rollouts",
        "8",
        "--kl-penalty",
        "0.02",
      ],
      environment: {
        OPENCLAW_TRAINING_RUNTIME: "axolotl",
        OPENCLAW_REVIEWED_EXPORT_REQUIRED: "true",
      },
    });
  });
});
