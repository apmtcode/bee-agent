import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { writeJsonAtomic, readJsonFile } from "../shared/fs.js";
import { MockLocalTrainingBackend, type MockMovementModel, replayMovementSequence } from "./backends.js";
import { createLocalTrainingExecution, createLocalTrainingJobManifest } from "./job-manifest.js";
import { LocalAppleSiliconTrainingRunner } from "./runner.js";
import { runMockTrainer, type MockMovementDataset } from "./mock-trainer.js";
import type { ReviewedExportManifest } from "./export-manifest.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "mock-trainer-"));
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
  modes: ["sft"],
  rawCaptureIncluded: false,
  executableSkills: [],
  executableSkillRuns: [],
  promotedSkills: [],
  memories: [],
  trajectories: [],
  replays: [],
};

describe("runner with the mock backend", () => {
  it("builds a portable plan and launch script that invokes node, not python", async () => {
    const rootDir = await makeTempDir();
    const runner = new LocalAppleSiliconTrainingRunner(rootDir, new MockLocalTrainingBackend());
    const job = createLocalTrainingJobManifest({ id: "job-mock", exportManifest, mode: "sft" });
    const execution = createLocalTrainingExecution({ jobId: job.id, mode: job.mode });

    const plan = await runner.writeArtifacts(job, execution);
    expect(plan).toMatchObject({
      runtime: "mock",
      targetPlatform: "portable",
      outputPath: `training-jobs/${job.id}/artifacts/model.json`,
    });
    expect(plan.command[0]).toBe("node");

    const script = await runner.readLaunchScript({ ...job, execution });
    expect(script).toContain("node");
    expect(script).not.toContain("python3 -m mlx_lm");
  });
});

describe("runMockTrainer", () => {
  it("trains a movement model from a movements.json dataset and persists it", async () => {
    const rootDir = await makeTempDir();
    const datasetDir = path.join(rootDir, "dataset");
    const dataset: MockMovementDataset = {
      version: 1,
      sequences: [
        ["open-app", "click-file", "click-save"],
        ["open-app", "click-file", "click-save"],
        ["open-app", "click-edit"],
      ],
    };
    await writeJsonAtomic(path.join(datasetDir, "movements.json"), dataset);
    const outPath = path.join(rootDir, "artifacts", "model.json");

    const result = await runMockTrainer({ datasetDir, outPath, mode: "sft" });
    expect(result.model.sequenceCount).toBe(3);

    const persisted = await readJsonFile<MockMovementModel | undefined>(outPath, undefined);
    expect(persisted).toBeDefined();
    expect(replayMovementSequence(persisted!, { start: "open-app" })).toEqual([
      "open-app",
      "click-file",
      "click-save",
    ]);
  });

  it("returns a valid empty model when the dataset is missing", async () => {
    const rootDir = await makeTempDir();
    const outPath = path.join(rootDir, "artifacts", "model.json");
    const result = await runMockTrainer({ datasetDir: path.join(rootDir, "dataset"), outPath });
    expect(result.model.sequenceCount).toBe(0);
    expect(result.model.transitionCount).toBe(0);
    expect(result.model.vocabulary).toEqual([]);
  });
});
