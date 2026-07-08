import { describe, expect, it } from "vitest";
import {
  generateSyntheticMovementDataset,
  splitMovementDataset,
} from "./synthetic.js";

describe("synthetic movement generator", () => {
  it("is deterministic for a given seed", () => {
    const a = generateSyntheticMovementDataset({ seed: 3, sequenceCount: 10 });
    const b = generateSyntheticMovementDataset({ seed: 3, sequenceCount: 10 });
    expect(a).toEqual(b);
  });

  it("varies with the seed", () => {
    const a = generateSyntheticMovementDataset({ seed: 1, sequenceCount: 10 });
    const b = generateSyntheticMovementDataset({ seed: 2, sequenceCount: 10 });
    expect(a).not.toEqual(b);
  });

  it("produces non-empty sequences within the requested bounds", () => {
    const dataset = generateSyntheticMovementDataset({ seed: 5, sequenceCount: 12, minSteps: 3, maxSteps: 6 });
    expect(dataset.sequences).toHaveLength(12);
    for (const sequence of dataset.sequences) {
      expect(sequence.steps.length).toBeGreaterThanOrEqual(1);
      expect(sequence.steps.length).toBeLessThanOrEqual(6);
      expect(sequence.steps[0]!.token).toBe("os.focus-changed");
    }
  });

  it("splits into disjoint train/holdout partitions", () => {
    const dataset = generateSyntheticMovementDataset({ seed: 9, sequenceCount: 20 });
    const { train, holdout } = splitMovementDataset(dataset, 0.25);
    expect(train.sequences.length + holdout.sequences.length).toBe(20);
    const trainIds = new Set(train.sequences.map((s) => s.id));
    expect(holdout.sequences.some((s) => trainIds.has(s.id))).toBe(false);
  });
});
