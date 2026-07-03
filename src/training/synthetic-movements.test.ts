import { describe, expect, it } from "vitest";
import {
  generateSyntheticMovementSequences,
  generateSyntheticMovementSplit,
  SYNTHETIC_MOVEMENT_PATTERN_NAMES,
} from "./synthetic-movements.js";
import { movementFeatureKey } from "./movement-model.js";

describe("generateSyntheticMovementSequences", () => {
  it("is deterministic for a given seed", () => {
    const a = generateSyntheticMovementSequences({ seed: 42, count: 6 });
    const b = generateSyntheticMovementSequences({ seed: 42, count: 6 });
    expect(JSON.stringify(a)).toBe(JSON.stringify(b));
  });

  it("varies with the seed", () => {
    const a = generateSyntheticMovementSequences({ seed: 1, count: 6 });
    const b = generateSyntheticMovementSequences({ seed: 2, count: 6 });
    expect(JSON.stringify(a)).not.toBe(JSON.stringify(b));
  });

  it("produces non-empty sequences across all known patterns", () => {
    const sequences = generateSyntheticMovementSequences({ seed: 7, count: SYNTHETIC_MOVEMENT_PATTERN_NAMES.length });
    expect(sequences).toHaveLength(SYNTHETIC_MOVEMENT_PATTERN_NAMES.length);
    for (const sequence of sequences) {
      expect(sequence.features.length).toBeGreaterThan(0);
    }
  });

  it("restricts to requested patterns", () => {
    const sequences = generateSyntheticMovementSequences({
      seed: 3,
      count: 4,
      patterns: ["scroll-and-select"],
    });
    for (const sequence of sequences) {
      expect(sequence.id).toContain("scroll-and-select");
    }
  });
});

describe("generateSyntheticMovementSplit", () => {
  it("uses a disjoint target vocabulary for the held-out split", () => {
    const { train, heldOut } = generateSyntheticMovementSplit({
      seed: 10,
      trainCount: 6,
      heldOutCount: 6,
    });
    const trainTargets = new Set(train.flatMap((s) => s.features.map((f) => f.target).filter(Boolean)));
    const heldOutTargets = new Set(heldOut.flatMap((s) => s.features.map((f) => f.target).filter(Boolean)));
    // "send"/"save" are shared fixed steps, but the sampled app/button/field
    // vocabularies are disjoint — so most held-out targets are unseen.
    const overlap = [...heldOutTargets].filter((t) => trainTargets.has(t as string));
    expect(overlap.length).toBeLessThan(heldOutTargets.size);
  });

  it("preserves movement shapes across the split", () => {
    const { train, heldOut } = generateSyntheticMovementSplit({
      seed: 11,
      trainCount: 3,
      heldOutCount: 3,
      patterns: ["compose-and-send"],
    });
    const shape = (features: { gesture: string }[]) => features.map((f) => f.gesture).join(",");
    expect(shape(heldOut[0]!.features)).toBe(shape(train[0]!.features));
    // Exact features differ because the targets differ.
    expect(train[0]!.features.map(movementFeatureKey)).not.toEqual(heldOut[0]!.features.map(movementFeatureKey));
  });
});
