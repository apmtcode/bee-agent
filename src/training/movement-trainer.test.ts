import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { buildMovementDataset } from "./movement-dataset.js";
import { createMovementBackend } from "./movement-model.js";
import {
  MovementModelTrainer,
  evaluateGeneralization,
  evaluateReplayFidelity,
} from "./movement-trainer.js";
import { synthesizeMovementDataset, synthesizeMovementTrajectories } from "./synthetic-movements.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "movement-trainer-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })));
});

describe("MovementModelTrainer", () => {
  it("trains, persists, and reloads a movement model", async () => {
    const dir = await makeTempDir();
    const trainer = new MovementModelTrainer(dir);
    const dataset = synthesizeMovementDataset({ seed: 1, variantsPerFamily: 8 });

    const result = await trainer.train("run-1", dataset);
    expect(result.model.trainedSequences).toBeGreaterThan(0);
    expect(result.fidelity.predictions).toBeGreaterThan(0);

    const persisted = JSON.parse(await fs.readFile(result.modelPath, "utf8"));
    expect(persisted.id).toBe("run-1");
    expect(persisted.backend).toBe("markov");

    const reloaded = await trainer.loadModel("run-1");
    expect(reloaded).toEqual(result.model);
  });

  it("returns undefined when loading a model that does not exist", async () => {
    const dir = await makeTempDir();
    const trainer = new MovementModelTrainer(dir);
    expect(await trainer.loadModel("missing")).toBeUndefined();
  });

  it("replays recorded movements with high fidelity", async () => {
    const dir = await makeTempDir();
    const trainer = new MovementModelTrainer(dir);
    const dataset = synthesizeMovementDataset({ seed: 1, variantsPerFamily: 8 });
    const result = await trainer.train("run-2", dataset);
    // Structural transitions are learned; only genuinely-random slots miss.
    expect(result.fidelity.accuracy).toBeGreaterThanOrEqual(0.6);
  });

  it("generalizes to held-out but related movement sequences", async () => {
    const dir = await makeTempDir();
    const backend = createMovementBackend("markov");
    const trainer = new MovementModelTrainer(dir, backend);

    const train = synthesizeMovementDataset({ seed: 1, variantsPerFamily: 10 });
    const heldOut = synthesizeMovementDataset({ seed: 99, variantsPerFamily: 4 });

    const { model } = await trainer.train("run-3", train);
    const generalization = evaluateGeneralization(backend, model, heldOut.sequences, 3);

    expect(generalization.predictions).toBeGreaterThan(0);
    expect(generalization.topKAccuracy).toBeGreaterThanOrEqual(generalization.top1Accuracy);
    // The model performs new-but-related movements well above chance.
    expect(generalization.topKAccuracy).toBeGreaterThanOrEqual(0.7);
  });

  it("trains end-to-end from captured trajectory spans", async () => {
    const dir = await makeTempDir();
    const backend = createMovementBackend("markov");
    const trainer = new MovementModelTrainer(dir, backend);

    const spans = synthesizeMovementTrajectories({ seed: 3, variantsPerFamily: 6 });
    const dataset = buildMovementDataset(spans);
    expect(dataset.sequences.length).toBe(spans.length);

    const { model } = await trainer.train("run-4", dataset);
    const fidelity = evaluateReplayFidelity(backend, model, dataset.sequences);
    expect(fidelity.accuracy).toBeGreaterThanOrEqual(0.6);
  });
});
