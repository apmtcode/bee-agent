import { spawnSync } from "node:child_process";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  AppleSiliconTrainingBackend,
  MockTrainingBackend,
} from "./backend.js";
import { createLocalTrainingExecution, createLocalTrainingJobManifest } from "./job-manifest.js";
import { LocalAppleSiliconTrainingRunner } from "./runner.js";
import type { ReviewedExportManifest } from "./export-manifest.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "training-backend-"));
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
  promotedSkills: [],
  memories: [],
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

describe("training backend seam", () => {
  it("defaults to the Apple Silicon backend (behaviour unchanged)", () => {
    const runner = new LocalAppleSiliconTrainingRunner("/tmp/does-not-matter");
    const job = createLocalTrainingJobManifest({ id: "job-default", exportManifest, mode: "sft" });
    const execution = createLocalTrainingExecution({ jobId: job.id, mode: job.mode });

    const plan = runner.buildPlan(job, execution);
    expect(plan.runtime).toBe("mlx");
    expect(plan.command[2]).toBe("mlx_lm.lora");
    expect(plan.environment.OPENCLAW_TRAINING_RUNTIME).toBe("mlx");
  });

  it("accepts an injected backend and threads its runtime through the plan", () => {
    const runner = new LocalAppleSiliconTrainingRunner("/tmp/does-not-matter", new MockTrainingBackend());
    const job = createLocalTrainingJobManifest({ id: "job-mock", exportManifest, mode: "sft" });
    const execution = createLocalTrainingExecution({ jobId: job.id, mode: job.mode });

    const plan = runner.buildPlan(job, execution);
    expect(plan.runtime).toBe("mock");
    expect(plan.command[0]).toBe("node");
    expect(plan.outputPath).toBe(`training-jobs/${job.id}/artifacts/model.mock.json`);
    expect(plan.environment.OPENCLAW_TRAINING_RUNTIME).toBe("mock");
    expect(plan.environment.OPENCLAW_MOCK_DATASET).toBe(`training-jobs/${job.id}/dataset/manifest.json`);
    // Standard reviewed-export guarantees still apply to a pluggable backend.
    expect(plan.environment.OPENCLAW_REVIEWED_EXPORT_REQUIRED).toBe("true");
    expect(plan.environment.OPENCLAW_RAW_CAPTURE_ALLOWED).toBe("false");
  });

  it("mock backend runs the pipeline end-to-end and produces a deterministic artifact", async () => {
    const rootDir = await makeTempDir();
    const runner = new LocalAppleSiliconTrainingRunner(rootDir, new MockTrainingBackend());
    const job = createLocalTrainingJobManifest({ id: "job-e2e", exportManifest, mode: "sft" });
    const execution = createLocalTrainingExecution({ jobId: job.id, mode: job.mode });

    // Prepare on-disk artifacts (writes the dataset manifest the mock reads).
    const plan = await runner.writeArtifacts(job, execution);

    // Execute the training command exactly as the launch script would: cwd is
    // the runner root, env carries the plan's environment. No Python / ML deps.
    const result = spawnSync(plan.command[0], plan.command.slice(1), {
      cwd: rootDir,
      env: { ...process.env, ...plan.environment },
      encoding: "utf8",
    });
    expect(result.status).toBe(0);
    expect(result.stdout).toContain("mock training complete");

    // The trained "model" artifact exists and is keyed to the dataset.
    const modelRaw = await fs.readFile(path.join(rootDir, plan.outputPath), "utf8");
    const model = JSON.parse(modelRaw);
    expect(model).toMatchObject({ backend: "mock", jobId: job.id, mode: "sft", weights: "deterministic-stub" });
    expect(model.datasetDigest).toMatch(/^[0-9a-f]{16}$/);

    // A replay-eval result was produced alongside the model.
    const evalRaw = await fs.readFile(path.join(rootDir, execution.artifactDir, "replay-eval-result.json"), "utf8");
    const evalResult = JSON.parse(evalRaw);
    expect(evalResult).toMatchObject({ backend: "mock", jobId: job.id, passed: true, replayCount: 1 });
    expect(evalResult.datasetDigest).toBe(model.datasetDigest);
  });

  it("mock training is deterministic: identical datasets yield identical digests", async () => {
    async function trainDigest(id: string): Promise<string> {
      const rootDir = await makeTempDir();
      const runner = new LocalAppleSiliconTrainingRunner(rootDir, new MockTrainingBackend());
      const job = createLocalTrainingJobManifest({ id, exportManifest, mode: "sft" });
      const execution = createLocalTrainingExecution({ jobId: job.id, mode: job.mode });
      const plan = await runner.writeArtifacts(job, execution);
      const result = spawnSync(plan.command[0], plan.command.slice(1), {
        cwd: rootDir,
        env: { ...process.env, ...plan.environment },
        encoding: "utf8",
      });
      expect(result.status).toBe(0);
      const model = JSON.parse(await fs.readFile(path.join(rootDir, plan.outputPath), "utf8"));
      return model.datasetDigest;
    }

    // Same dataset shape (only jobId differs) → same digest, proving the mock
    // trains a function of the reviewed data rather than of run-to-run noise.
    const [a, b] = await Promise.all([trainDigest("job-a"), trainDigest("job-b")]);
    expect(a).toBe(b);
  });

  it("Apple Silicon backend still selects axolotl for RL jobs", () => {
    const backend = new AppleSiliconTrainingBackend();
    const job = createLocalTrainingJobManifest({ id: "job-rl", exportManifest, mode: "rl" });
    const execution = createLocalTrainingExecution({ jobId: job.id, mode: job.mode });
    const plan = backend.buildExecutionPlan(job, execution);
    expect(plan.runtime).toBe("axolotl");
    expect(plan.outputArtifact).toBe("policy.gguf");
    expect(plan.command).toContain("axolotl.cli.train");
  });
});
