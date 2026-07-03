import { describe, expect, it } from "vitest";
import { MarkovMovementBackend } from "./model-backend.js";
import {
  createSeededRng,
  evaluateGeneralization,
  generateSyntheticDataset,
  generateSyntheticMovementSequences,
} from "./movement-eval.js";

describe("seeded RNG", () => {
  it("is deterministic for a given seed and varies across seeds", () => {
    const a = createSeededRng(42);
    const b = createSeededRng(42);
    const c = createSeededRng(43);
    const drawA = [a(), a(), a()];
    const drawB = [b(), b(), b()];
    expect(drawA).toEqual(drawB);
    expect(drawA.every((value) => value >= 0 && value < 1)).toBe(true);
    expect([c(), c(), c()]).not.toEqual(drawA);
  });
});

describe("synthetic movement generator", () => {
  it("produces a deterministic, reproducible set of sequences", () => {
    const first = generateSyntheticMovementSequences({ seed: 7, count: 5 });
    const second = generateSyntheticMovementSequences({ seed: 7, count: 5 });
    expect(first).toEqual(second);
    expect(first).toHaveLength(5);
    expect(first.every((sequence) => sequence.tokens.length > 0)).toBe(true);
  });

  it("respects the max-length bound", () => {
    const sequences = generateSyntheticMovementSequences({ seed: 1, count: 20, maxLength: 4 });
    expect(sequences.every((sequence) => sequence.tokens.length <= 4)).toBe(true);
  });
});

describe("generalization eval", () => {
  it("reproduces the training draw with high fidelity", async () => {
    const dataset = generateSyntheticDataset({ seed: 11, count: 60 });
    const model = await new MarkovMovementBackend({ order: 3 }).train({ dataset });
    const report = evaluateGeneralization(model, dataset.sequences);
    expect(report.sequenceCount).toBe(60);
    expect(report.tokenCount).toBeGreaterThan(0);
    // The grammar is genuinely stochastic (branch factor ~10), so even a
    // perfect model is bounded by its inherent predictability; well above the
    // ~0.1 uniform-chance floor.
    expect(report.nextTokenAccuracy).toBeGreaterThan(0.45);
    expect(Number.isFinite(report.perplexity)).toBe(true);
  });

  it("generalizes to a held-out draw from the same grammar better than chance", async () => {
    const train = generateSyntheticDataset({ seed: 100, count: 120 });
    const heldOut = generateSyntheticMovementSequences({ seed: 999, count: 40, idPrefix: "heldout" });
    const model = await new MarkovMovementBackend({ order: 3 }).train({ dataset: train });

    const report = evaluateGeneralization(model, heldOut);
    // ~10 distinct tokens => chance next-token accuracy ~0.1. Structure learned
    // from the grammar should push held-out accuracy well above chance.
    expect(report.nextTokenAccuracy).toBeGreaterThan(0.35);
    // Held-out should be no easier than the training draw.
    const trainReport = evaluateGeneralization(model, train.sequences);
    expect(report.perplexity).toBeGreaterThanOrEqual(trainReport.perplexity * 0.5);
  });

  it("scores an in-grammar draw as less surprising than uniform random tokens", async () => {
    const train = generateSyntheticDataset({ seed: 5, count: 100 });
    const model = await new MarkovMovementBackend({ order: 3 }).train({ dataset: train });
    const inGrammar = generateSyntheticMovementSequences({ seed: 55, count: 30 });
    const rng = createSeededRng(7);
    const vocab = model.vocabulary();
    const randomSeqs = Array.from({ length: 30 }, (_, index) => ({
      id: `rand-${index}`,
      tokens: Array.from({ length: 6 }, () => vocab[Math.floor(rng() * vocab.length)]!),
    }));

    const grammarReport = evaluateGeneralization(model, inGrammar);
    const randomReport = evaluateGeneralization(model, randomSeqs);
    expect(grammarReport.perplexity).toBeLessThan(randomReport.perplexity);
  });
});
