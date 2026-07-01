import { describe, expect, it } from "vitest";
import { tokenizeTrajectory } from "./movement-model.js";
import {
  createSeededRng,
  datasetVocabulary,
  defaultSyntheticTemplates,
  generateSyntheticMovementDataset,
} from "./synthetic.js";

describe("createSeededRng", () => {
  it("is deterministic for a given seed and varies across seeds", () => {
    const a = createSeededRng(42);
    const b = createSeededRng(42);
    const c = createSeededRng(43);
    const seqA = [a(), a(), a()];
    const seqB = [b(), b(), b()];
    expect(seqA).toEqual(seqB);
    expect(seqA).not.toEqual([c(), c(), c()]);
    for (const value of seqA) {
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThan(1);
    }
  });
});

describe("generateSyntheticMovementDataset", () => {
  it("produces the requested number of aligned spans and sequences", () => {
    const dataset = generateSyntheticMovementDataset({ seed: 1, count: 8 });
    expect(dataset.spans).toHaveLength(8);
    expect(dataset.sequences).toHaveLength(8);
    dataset.spans.forEach((span, index) => {
      // Each recorded span tokenizes to exactly its paired sequence (round-trip).
      expect(tokenizeTrajectory(span)).toEqual(dataset.sequences[index]);
      expect(span.actions.length).toBeGreaterThan(0);
    });
  });

  it("is fully deterministic for a fixed seed", () => {
    const first = generateSyntheticMovementDataset({ seed: 7, count: 5 });
    const second = generateSyntheticMovementDataset({ seed: 7, count: 5 });
    expect(second.sequences).toEqual(first.sequences);
    expect(second.spans).toEqual(first.spans);
  });

  it("differs across seeds", () => {
    const a = generateSyntheticMovementDataset({ seed: 1, count: 12 });
    const b = generateSyntheticMovementDataset({ seed: 2, count: 12 });
    expect(b.sequences).not.toEqual(a.sequences);
  });

  it("still emits at least one action when every step may drop", () => {
    const dataset = generateSyntheticMovementDataset({ seed: 3, count: 6, dropProbability: 1 });
    for (const sequence of dataset.sequences) {
      expect(sequence.length).toBeGreaterThan(0);
    }
  });

  it("derives a sorted vocabulary from the dataset", () => {
    const dataset = generateSyntheticMovementDataset({ seed: 9, count: 20 });
    const vocab = datasetVocabulary(dataset);
    expect(vocab.length).toBeGreaterThan(0);
    expect([...vocab].sort()).toEqual(vocab);
  });

  it("rejects an empty template set", () => {
    expect(() => generateSyntheticMovementDataset({ seed: 1, count: 1, templates: [] })).toThrow();
  });

  it("ships a non-empty default template set", () => {
    expect(defaultSyntheticTemplates().length).toBeGreaterThan(0);
  });
});
