import { execFileSync } from "node:child_process";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  AxolotlTrainingBackend,
  MlxTrainingBackend,
  MockTrainingBackend,
  MOCK_TRAINER_SOURCE,
  TrainingBackendRegistry,
} from "./backends.js";
import { createLocalTrainingExecution, createLocalTrainingJobManifest } from "./job-manifest.js";
import type { ReviewedExportManifest } from "./export-manifest.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "training-backends-"));
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
  trajectories: [],
  replays: [],
};

function makeJob(mode: "sft" | "rl") {
  const job = createLocalTrainingJobManifest({ id: `job-${mode}`, exportManifest, mode });
  const execution = createLocalTrainingExecution({ jobId: job.id, mode: job.mode });
  return { job, execution };
}

describe("TrainingBackendRegistry.default", () => {
  it("resolves SFT to the MLX backend", () => {
    const registry = TrainingBackendRegistry.default();
    const backend = registry.resolve("sft");
    expect(backend).toBeInstanceOf(MlxTrainingBackend);
    expect(backend.runtime).toBe("mlx");

    const { job, execution } = makeJob("sft");
    const plan = backend.buildPlan(job, execution);
    expect(plan).toMatchObject({
      runtime: "mlx",
      outputPath: `training-jobs/${job.id}/artifacts/model.gguf`,
      environment: { OPENCLAW_TRAINING_RUNTIME: "mlx", OPENCLAW_RAW_CAPTURE_ALLOWED: "false" },
    });
    expect(plan.command).toContain("mlx_lm.lora");
  });

  it("resolves RL to the Axolotl backend", () => {
    const registry = TrainingBackendRegistry.default();
    const backend = registry.resolve("rl");
    expect(backend).toBeInstanceOf(AxolotlTrainingBackend);

    const { job, execution } = makeJob("rl");
    const plan = backend.buildPlan(job, execution);
    expect(plan).toMatchObject({
      runtime: "axolotl",
      outputPath: `training-jobs/${job.id}/artifacts/policy.gguf`,
    });
    expect(plan.command).toContain("axolotl.cli.train");
  });

  it("throws for a mode with no registered backend", () => {
    const registry = new TrainingBackendRegistry({ sft: new MlxTrainingBackend() });
    expect(registry.modes()).toEqual(["sft"]);
    expect(() => registry.resolve("rl")).toThrow(/No training backend registered/);
  });
});

describe("MockTrainingBackend", () => {
  it("plans a dependency-free node command for every mode", () => {
    const registry = TrainingBackendRegistry.mock();
    for (const mode of ["sft", "rl"] as const) {
      const { job, execution } = makeJob(mode);
      const plan = registry.resolve(mode).buildPlan(job, execution);
      expect(plan.runtime).toBe("mock");
      expect(plan.command[0]).toBe("node");
      expect(plan.command[1]).toBe("-e");
      expect(plan.outputPath).toBe(`training-jobs/${job.id}/artifacts/model.json`);
      expect(plan.environment.BEE_MOCK_DATASET_DIR).toBe(execution.datasetDir);
      expect(plan.environment.BEE_MOCK_OUTPUT_PATH).toBe(plan.outputPath);
      // No ML toolchain referenced anywhere in the mock plan.
      expect(plan.command.join(" ")).not.toContain("python3");
    }
  });

  it("runs end-to-end to a deterministic artifact with no ML toolchain", async () => {
    const rootDir = await makeTempDir();
    const backend = new MockTrainingBackend();
    const { job, execution } = makeJob("sft");
    const plan = backend.buildPlan(job, execution);

    // Materialize the reviewed dataset manifest the trainer reads.
    const datasetDir = path.join(rootDir, execution.datasetDir);
    await fs.mkdir(datasetDir, { recursive: true });
    await fs.writeFile(
      path.join(datasetDir, "manifest.json"),
      JSON.stringify({ jobId: job.id, mode: job.mode, dataset: job.dataset }),
    );

    // Actually execute the planned command — this must complete without mlx/axolotl.
    execFileSync("node", ["-e", MOCK_TRAINER_SOURCE], {
      cwd: rootDir,
      env: { ...process.env, ...plan.environment },
    });

    const artifact = JSON.parse(await fs.readFile(path.join(rootDir, plan.outputPath), "utf8"));
    expect(artifact).toMatchObject({ backend: "mock", jobId: job.id, mode: "sft" });
    expect(artifact.weightsDigest).toMatch(/^[0-9a-f]{8}$/);
    expect(artifact.trainedFrom).toEqual(job.dataset);

    // Determinism: re-running the same dataset yields the same weights digest.
    execFileSync("node", ["-e", MOCK_TRAINER_SOURCE], {
      cwd: rootDir,
      env: { ...process.env, ...plan.environment },
    });
    const rerun = JSON.parse(await fs.readFile(path.join(rootDir, plan.outputPath), "utf8"));
    expect(rerun.weightsDigest).toBe(artifact.weightsDigest);
  });
});
