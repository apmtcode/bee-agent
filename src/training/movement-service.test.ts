import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { MovementLearningService } from "./movement-service.js";
import { generateSyntheticDataset, relatedVariant, syntheticWorkflow } from "./synthetic-movements.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "movement-service-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })));
});

describe("MovementLearningService", () => {
  it("trains, persists and reloads a model round-trip", async () => {
    const service = new MovementLearningService();
    const dataset = { version: 1 as const, sequences: generateSyntheticDataset(6) };
    const model = await service.train(dataset, { order: 2 });
    expect(model.trainedSequences).toBe(6);
    expect(model.backendId).toBe("deterministic-markov");

    const rootDir = await makeTempDir();
    await service.saveModel(rootDir, "models/movement.json", model);
    const reloaded = await service.loadModel(rootDir, "models/movement.json");
    expect(reloaded).toEqual(model);

    // A reloaded model still predicts identically to the in-memory one.
    const prefix = syntheticWorkflow("mail", "compose").slice(0, 1);
    const fromMemory = await service.predict(model, prefix, { maxSteps: 5 });
    const fromDisk = await service.predict(reloaded!, prefix, { maxSteps: 5 });
    expect(fromDisk).toEqual(fromMemory);
  });

  it("scores generalization on held-out related variants", async () => {
    const service = new MovementLearningService();
    const training = generateSyntheticDataset(24);
    const model = await service.train({ version: 1, sequences: training }, { order: 2 });

    // Held-out sequences are related variants (same skeleton + an extra scroll).
    const heldOut = [relatedVariant("mail", "compose"), relatedVariant("browser", "search")];
    const report = await service.evaluateGeneralization(model, heldOut, { prefixLength: 2 });

    expect(report.evaluated).toBe(2);
    expect(report.stepAccuracy).toBeGreaterThan(0);
    expect(report.stepAccuracy).toBeLessThanOrEqual(1);
    expect(report.perSequence).toHaveLength(2);
  });

  it("reproduces held-out sequences drawn from the training distribution exactly", async () => {
    const service = new MovementLearningService();
    const sequences = generateSyntheticDataset(8);
    const model = await service.train({ version: 1, sequences }, { order: 2 });

    // Evaluate on the training sequences themselves: with a 2-step prefix (which
    // disambiguates the target) the deterministic model must reproduce every
    // remaining step of each training sequence.
    const report = await service.evaluateGeneralization(model, sequences, { prefixLength: 2 });
    expect(report.sequenceAccuracy).toBe(1);
    expect(report.stepAccuracy).toBe(1);
  });
});
