import { describe, expect, it } from "vitest";
import {
  DEFAULT_MOVEMENT_MOTIFS,
  generateSyntheticMovementDataset,
} from "./movement-synthetic.js";

describe("generateSyntheticMovementDataset", () => {
  it("is deterministic for a given seed", () => {
    const a = generateSyntheticMovementDataset({ seed: 42, sequenceCount: 5 });
    const b = generateSyntheticMovementDataset({ seed: 42, sequenceCount: 5 });
    expect(a).toEqual(b);
  });

  it("varies with the seed", () => {
    const a = generateSyntheticMovementDataset({ seed: 1, sequenceCount: 5 });
    const b = generateSyntheticMovementDataset({ seed: 2, sequenceCount: 5 });
    expect(a).not.toEqual(b);
  });

  it("produces the requested number of non-empty sequences from known motifs", () => {
    const dataset = generateSyntheticMovementDataset({
      seed: 9,
      sequenceCount: 6,
      minMotifs: 2,
      maxMotifs: 4,
    });
    expect(dataset.sequences).toHaveLength(6);
    const motifTokens = new Set(Object.values(DEFAULT_MOVEMENT_MOTIFS).flat());
    for (const sequence of dataset.sequences) {
      expect(sequence.tokens.length).toBeGreaterThan(0);
      for (const token of sequence.tokens) {
        expect(motifTokens.has(token)).toBe(true);
      }
    }
  });

  it("returns no sequences when the motif library is empty", () => {
    const dataset = generateSyntheticMovementDataset({ seed: 3, sequenceCount: 4, motifs: {} });
    expect(dataset.sequences).toEqual([]);
  });
});
