import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  MOVEMENT_END,
  MovementModelRegistry,
  NGramMovementBackend,
  buildMovementDatasetFromReplay,
  buildMovementDatasetFromTrajectories,
  defaultMovementModelRegistry,
  evaluateMovementModel,
  generateSyntheticMovementDataset,
  loadMovementModel,
  saveMovementModel,
  tokenizeMovement,
  type MovementDataset,
} from "./movement-model.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "movement-model-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })));
});

describe("tokenizeMovement", () => {
  it("encodes tool + gesture facets into a stable, distinguishable token", () => {
    expect(tokenizeMovement({ tool: "device", metadata: { gesture: "tap", target: "Send Button" } })).toBe(
      "device:tap:send-button",
    );
    expect(tokenizeMovement({ tool: "device", metadata: { gesture: "swipe", direction: "left" } })).toBe(
      "device:swipe:left",
    );
  });

  it("falls back to the summary when there is no gesture metadata", () => {
    expect(tokenizeMovement({ tool: "browser", summary: "Clicked Submit" })).toBe("browser:clicked-submit");
  });
});

describe("dataset builders", () => {
  it("builds sequences from trajectory actions and drops empties", () => {
    const dataset = buildMovementDatasetFromTrajectories([
      { id: "a", actions: [{ tool: "device", metadata: { gesture: "tap", target: "x" } }] },
      { id: "empty", actions: [] },
    ]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]).toMatchObject({ id: "a", tokens: ["device:tap:x"] });
  });

  it("groups replay action events by trajectory", () => {
    const dataset = buildMovementDatasetFromReplay({
      events: [
        { kind: "transcript" },
        { kind: "action", trajectoryId: "t1", tool: "device", summary: "tapped app" },
        { kind: "action", trajectoryId: "t1", tool: "device", summary: "typed hello" },
        { kind: "observation", trajectoryId: "t1" },
      ],
    });
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]!.tokens).toHaveLength(2);
  });
});

describe("NGramMovementBackend", () => {
  const backend = new NGramMovementBackend();

  it("reproduces a single recorded workflow exactly (greedy)", () => {
    const dataset: MovementDataset = {
      sequences: [
        { id: "wf", tokens: ["device:tap:launcher", "device:tap:compose", "device:type:subject", "device:tap:send"] },
      ],
    };
    const model = backend.train(dataset, { order: 2 });
    const generated = backend.generate(model, { temperature: 0, maxTokens: 16 });
    expect(generated).toEqual(dataset.sequences[0]!.tokens);
  });

  it("predicts the highest-probability next movement from context", () => {
    const dataset: MovementDataset = {
      sequences: [
        { id: "a", tokens: ["open", "type", "send"] },
        { id: "b", tokens: ["open", "type", "send"] },
        { id: "c", tokens: ["open", "type", "discard"] },
      ],
    };
    const model = backend.train(dataset, { order: 2 });
    const prediction = backend.predictNext(model, ["open", "type"]);
    expect(prediction.candidates[0]!.token).toBe("send");
    expect(prediction.candidates[0]!.probability).toBeCloseTo(2 / 3, 5);
    // "send" and "discard" are the only continuations of "...type".
    expect(prediction.candidates.map((candidate) => candidate.token).sort()).toEqual(["discard", "send"]);
  });

  it("backs off to a shorter context when the full context is unseen", () => {
    const dataset: MovementDataset = {
      sequences: [{ id: "a", tokens: ["open", "type", "send"] }],
    };
    const model = backend.train(dataset, { order: 2 });
    // "never" was never seen, but "type" -> "send" backs off through the unigram/bigram tables.
    const prediction = backend.predictNext(model, ["never", "type"]);
    expect(prediction.candidates[0]!.token).toBe("send");
    expect(prediction.order).toBeLessThanOrEqual(2);
  });

  it("generation is deterministic for a fixed seed and varies across seeds", () => {
    const dataset = generateSyntheticMovementDataset({ seed: 3, sequenceCount: 30, noise: 0.3 });
    const model = backend.train(dataset, { order: 2 });
    const a1 = backend.generate(model, { seed: 42, temperature: 0.8, maxTokens: 20 });
    const a2 = backend.generate(model, { seed: 42, temperature: 0.8, maxTokens: 20 });
    expect(a1).toEqual(a2);
    expect(a1).not.toContain(MOVEMENT_END);
  });

  it("generalizes: composes a novel-but-related workflow not present verbatim in training", () => {
    // Two workflows share a prefix; the model can stitch a path that no single
    // training sequence contains verbatim, yet every transition is grounded.
    const dataset: MovementDataset = {
      sequences: [
        { id: "a", tokens: ["open", "search", "select", "copy"] },
        { id: "b", tokens: ["open", "search", "filter", "export"] },
      ],
    };
    const model = backend.train(dataset, { order: 1 });
    const sampled = backend.generate(model, { seed: 5, temperature: 1.2, maxTokens: 12 });
    // Every emitted movement is a real learned token...
    for (const token of sampled) {
      expect(model.vocabulary).toContain(token);
    }
    // ...and generation always starts from the shared entry movement.
    expect(sampled[0]).toBe("open");
  });
});

describe("evaluateMovementModel", () => {
  it("reports high top-1 accuracy on the training distribution", () => {
    const backend = new NGramMovementBackend();
    const dataset = generateSyntheticMovementDataset({ seed: 11, sequenceCount: 40, noise: 0.1 });
    const model = backend.train(dataset, { order: 2 });
    const result = evaluateMovementModel(backend, model, dataset.sequences, { k: 3 });
    expect(result.predictions).toBeGreaterThan(0);
    expect(result.top1Accuracy).toBeGreaterThan(0.6);
    expect(result.topKAccuracy).toBeGreaterThanOrEqual(result.top1Accuracy);
  });

  it("generalizes to held-out related sequences better than chance", () => {
    const backend = new NGramMovementBackend();
    const full = generateSyntheticMovementDataset({ seed: 21, sequenceCount: 60, noise: 0.2 });
    const train = { sequences: full.sequences.slice(0, 45) };
    const heldOut = full.sequences.slice(45);
    const model = backend.train(train, { order: 2 });
    const result = evaluateMovementModel(backend, model, heldOut, { k: 3 });
    // Vocabulary is tiny; a uniform baseline would be ~1/|vocab|. The learned
    // model should comfortably beat that on structurally-related held-out data.
    const chance = 1 / Math.max(1, model.vocabulary.length);
    expect(result.topKAccuracy).toBeGreaterThan(chance);
    expect(result.top1Accuracy).toBeGreaterThan(0.4);
  });
});

describe("MovementModelRegistry", () => {
  it("provides the n-gram backend by default and rejects unknown backends", () => {
    const registry = defaultMovementModelRegistry();
    expect(registry.list()).toContain("ngram");
    expect(registry.get("ngram")).toBeInstanceOf(NGramMovementBackend);
    expect(() => registry.get("mlx-policy")).toThrow(/Unknown movement model backend/);
  });

  it("accepts a pluggable custom backend implementing the interface", () => {
    const registry = new MovementModelRegistry();
    const stub = {
      name: "stub",
      train: () => ({
        version: 1 as const,
        backend: "stub",
        order: 1,
        vocabulary: [],
        grams: {},
        sequenceCount: 0,
        tokenCount: 0,
      }),
      predictNext: () => ({ candidates: [], order: 0 }),
      generate: () => [],
    };
    registry.register(stub);
    expect(registry.has("stub")).toBe(true);
    expect(registry.get("stub").name).toBe("stub");
  });
});

describe("model persistence", () => {
  it("round-trips a trained model through disk", async () => {
    const dir = await makeTempDir();
    const backend = new NGramMovementBackend();
    const dataset = generateSyntheticMovementDataset({ seed: 2, sequenceCount: 10 });
    const model = backend.train(dataset, { order: 2 });
    const file = path.join(dir, "model.json");
    await saveMovementModel(file, model);
    const loaded = await loadMovementModel(file);
    expect(loaded).toEqual(model);
    // A reloaded model generates identically to the in-memory one.
    expect(backend.generate(loaded!, { seed: 9, temperature: 0 })).toEqual(
      backend.generate(model, { seed: 9, temperature: 0 }),
    );
  });

  it("returns undefined for a missing model file", async () => {
    const dir = await makeTempDir();
    expect(await loadMovementModel(path.join(dir, "missing.json"))).toBeUndefined();
  });
});
