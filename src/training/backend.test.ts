import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { afterEach, describe, expect, it } from "vitest";
import {
  AppleSiliconTrainingBackend,
  MockTrainingBackend,
  TrainingBackendRegistry,
  createDefaultTrainingBackendRegistry,
  isInProcessTrainingBackend,
} from "./backend.js";
import { createLocalTrainingExecution, createLocalTrainingJobManifest } from "./job-manifest.js";
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

describe("AppleSiliconTrainingBackend", () => {
  it("supports apple-silicon jobs and builds the mlx SFT plan", () => {
    const backend = new AppleSiliconTrainingBackend();
    const job = createLocalTrainingJobManifest({ id: "job-sft", exportManifest, mode: "sft" });
    const execution = createLocalTrainingExecution({ jobId: job.id, mode: job.mode });

    expect(backend.supports(job)).toBe(true);
    const plan = backend.buildPlan(job, execution);
    expect(plan).toMatchObject({
      runtime: "mlx",
      targetPlatform: "apple-silicon",
      command: expect.arrayContaining(["mlx_lm.lora"]),
    });
  });

  it("is not an in-process backend", () => {
    expect(isInProcessTrainingBackend(new AppleSiliconTrainingBackend())).toBe(false);
  });
});

describe("MockTrainingBackend", () => {
  it("produces a deterministic model artifact for identical jobs", () => {
    const backend = new MockTrainingBackend();
    const job = createLocalTrainingJobManifest({ id: "job-mock", exportManifest, mode: "sft" });

    const first = backend.simulateTraining(job);
    const second = backend.simulateTraining(job);
    expect(first).toEqual(second);
    expect(first.backendId).toBe("mock");
    expect(first.modelFingerprint).toMatch(/^[0-9a-f]{64}$/);
    expect(first.datasetSignature.promotedSkillCount).toBe(1);
  });

  it("generalizes: different datasets yield different fingerprints", () => {
    const backend = new MockTrainingBackend();
    const jobA = createLocalTrainingJobManifest({ id: "job-a", exportManifest, mode: "sft" });
    const richerManifest: ReviewedExportManifest = {
      ...exportManifest,
      trajectories: [...exportManifest.trajectories, { ...exportManifest.trajectories[0]!, id: "traj-2" }],
    };
    const jobB = createLocalTrainingJobManifest({ id: "job-b", exportManifest: richerManifest, mode: "sft" });

    expect(backend.simulateTraining(jobA).modelFingerprint).not.toBe(
      backend.simulateTraining(jobB).modelFingerprint,
    );
  });

  it("is an in-process backend and supports any target", () => {
    const backend = new MockTrainingBackend();
    expect(isInProcessTrainingBackend(backend)).toBe(true);
    const job = createLocalTrainingJobManifest({ id: "job-any", exportManifest, mode: "rl" });
    expect(backend.supports(job)).toBe(true);
  });

  it("builds a self-contained plan whose command actually writes the model artifact", async () => {
    const rootDir = await makeTempDir();
    const backend = new MockTrainingBackend();
    const job = createLocalTrainingJobManifest({ id: "job-run", exportManifest, mode: "sft" });
    const execution = createLocalTrainingExecution({ jobId: job.id, mode: job.mode });

    const plan = backend.buildPlan(job, execution);
    expect(plan.runtime).toBe("mock");
    expect(plan.command[0]).toBe("node");

    // Execute the mock command exactly as the launch script would: cwd=rootDir,
    // env carrying the deterministic artifact. This proves the pipeline runs in
    // an environment with nothing but node available (i.e. the cloud/CI).
    const result = spawnSync(plan.command[0], plan.command.slice(1), {
      cwd: rootDir,
      env: { ...process.env, ...plan.environment },
      encoding: "utf8",
    });
    expect(result.status).toBe(0);

    const written = await fs.readFile(path.join(rootDir, plan.outputPath), "utf8");
    expect(JSON.parse(written)).toEqual(backend.simulateTraining(job));
  });
});

describe("TrainingBackendRegistry", () => {
  it("resolves the first supporting backend by default and by explicit id", () => {
    const registry = createDefaultTrainingBackendRegistry();
    const job = createLocalTrainingJobManifest({ id: "job-reg", exportManifest, mode: "sft" });

    // Apple-silicon is registered first and supports the job, so it wins.
    expect(registry.resolve(job).id).toBe("apple-silicon-mlx");
    expect(registry.resolve(job, "mock").id).toBe("mock");
    expect(registry.list().map((b) => b.id)).toEqual(["apple-silicon-mlx", "mock"]);
  });

  it("throws on unknown or unsupporting backend ids", () => {
    const registry = createDefaultTrainingBackendRegistry();
    const job = createLocalTrainingJobManifest({ id: "job-reg2", exportManifest, mode: "sft" });
    expect(() => registry.resolve(job, "does-not-exist")).toThrow(/Unknown training backend/);
  });

  it("throws when no backend supports the job", () => {
    const registry = new TrainingBackendRegistry().register(new AppleSiliconTrainingBackend());
    const job = createLocalTrainingJobManifest({ id: "job-reg3", exportManifest, mode: "sft" });
    // Force an unsupported platform to exercise the no-match path.
    const unsupported = { ...job, targetPlatform: "other-platform" as never };
    expect(() => registry.resolve(unsupported)).toThrow(/No registered training backend/);
  });
});
