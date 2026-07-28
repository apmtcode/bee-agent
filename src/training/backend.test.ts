import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  AxolotlRlBackend,
  MlxSftBackend,
  MockLocalBackend,
  TrainingBackendRegistry,
  defaultTrainingBackendRegistry,
  mockDatasetFile,
  mockTrainingBackendRegistry,
  runMockTrainingJob,
} from "./backend.js";
import { createLocalTrainingExecution, createLocalTrainingJobManifest } from "./job-manifest.js";
import type { MovementDataset } from "./mock-model.js";
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
  trajectories: [],
  replays: [],
};

function sftJob() {
  const job = createLocalTrainingJobManifest({ id: "job-sft", exportManifest, mode: "sft" });
  const execution = createLocalTrainingExecution({ jobId: job.id, mode: job.mode });
  return { job, execution };
}

describe("built-in backends", () => {
  it("MlxSftBackend supports only sft and emits the mlx command", () => {
    const backend = new MlxSftBackend();
    expect(backend.supportsMode("sft")).toBe(true);
    expect(backend.supportsMode("rl")).toBe(false);
    const { job, execution } = sftJob();
    const plan = backend.buildPlan({ job, execution });
    expect(plan.runtime).toBe("mlx");
    expect(plan.targetPlatform).toBe("apple-silicon");
    expect(plan.outputArtifact).toBe("model.gguf");
    expect(plan.command).toContain("mlx_lm.lora");
    expect(plan.environment.OPENCLAW_TRAINING_RUNTIME).toBe("mlx");
  });

  it("AxolotlRlBackend supports only rl", () => {
    const backend = new AxolotlRlBackend();
    expect(backend.supportsMode("rl")).toBe(true);
    expect(backend.supportsMode("sft")).toBe(false);
  });
});

describe("TrainingBackendRegistry", () => {
  it("selects by mode and rejects duplicate ids", () => {
    const registry = defaultTrainingBackendRegistry();
    expect(registry.select("sft").id).toBe("mlx-sft");
    expect(registry.select("rl").id).toBe("axolotl-rl");
    expect(() => registry.register(new MlxSftBackend())).toThrow(/already registered/);
  });

  it("honors a preferred backend id and validates support", () => {
    const registry = mockTrainingBackendRegistry();
    expect(registry.select("sft", "mock-local").id).toBe("mock-local");
    expect(() => registry.select("sft", "axolotl-rl")).toThrow(/does not support/);
    expect(() => registry.select("sft", "nope")).toThrow(/unknown training backend/);
  });

  it("throws when no backend supports the mode", () => {
    const registry = new TrainingBackendRegistry([new MlxSftBackend()]);
    expect(() => registry.select("rl")).toThrow(/no training backend/);
  });
});

describe("MockLocalBackend", () => {
  it("produces a portable, dependency-free plan", () => {
    const { job, execution } = sftJob();
    const plan = new MockLocalBackend().buildPlan({ job, execution });
    expect(plan.runtime).toBe("mock");
    expect(plan.targetPlatform).toBe("portable");
    expect(plan.outputArtifact).toBe("movement-model.json");
    expect(plan.command[0]).toBe("node");
    expect(plan.command).toContain("--input-type=module");
    expect(plan.environment.OPENCLAW_MOCK_DATASET_FILE).toBe(mockDatasetFile(execution));
    expect(plan.environment.OPENCLAW_MOCK_OUTPUT_PATH).toContain("movement-model.json");
  });
});

describe("runMockTrainingJob", () => {
  it("trains a movement model from a dataset file and writes the artifact", async () => {
    const dir = await makeTempDir();
    const datasetFile = path.join(dir, "movement-dataset.json");
    const outputPath = path.join(dir, "artifacts", "movement-model.json");
    const dataset: MovementDataset = {
      version: 1,
      sequences: [
        { sessionId: "s1", tokens: ["action:open", "action:type", "action:save"] },
        { sessionId: "s2", tokens: ["action:open", "action:type", "action:save"] },
      ],
    };
    await fs.writeFile(datasetFile, JSON.stringify(dataset), "utf8");

    const model = await runMockTrainingJob({ datasetFile, outputPath });
    expect(model.sequenceCount).toBe(2);
    expect(model.starts).toEqual({ "action:open": 2 });

    const written = JSON.parse(await fs.readFile(outputPath, "utf8"));
    expect(written).toEqual(model);
  });

  it("rejects missing params and malformed datasets", async () => {
    const dir = await makeTempDir();
    await expect(runMockTrainingJob({ outputPath: "x" })).rejects.toThrow(/datasetFile/);
    await expect(runMockTrainingJob({ datasetFile: "x" })).rejects.toThrow(/outputPath/);
    await expect(
      runMockTrainingJob({
        datasetFile: path.join(dir, "missing.json"),
        outputPath: path.join(dir, "out.json"),
      }),
    ).rejects.toThrow(/malformed/);
  });
});
