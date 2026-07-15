import { describe, expect, it } from "vitest";
import {
  createSeededRng,
  desktopMovementFamily,
  generateSyntheticTrajectories,
} from "./synthetic-trajectories.js";

describe("createSeededRng", () => {
  it("is deterministic for a given seed", () => {
    const a = createSeededRng(42);
    const b = createSeededRng(42);
    const seqA = [a(), a(), a()];
    const seqB = [b(), b(), b()];
    expect(seqA).toEqual(seqB);
    expect(seqA.every((value) => value >= 0 && value < 1)).toBe(true);
  });

  it("diverges for different seeds", () => {
    expect(createSeededRng(1)()).not.toBe(createSeededRng(2)());
  });
});

describe("generateSyntheticTrajectories", () => {
  const family = desktopMovementFamily();

  it("produces the requested number of spans with monotonic timestamps", () => {
    const spans = generateSyntheticTrajectories({
      scenarios: family.train,
      spansPerScenario: 3,
      seed: 7,
    });

    expect(spans).toHaveLength(family.train.length * 3);
    // Every event gets a distinct, strictly increasing timestamp (each step
    // advances ts by at least 1), so the merged replay timeline is unambiguous.
    const timestamps = spans
      .flatMap((span) => [...span.observations, ...span.actions].map((e) => e.ts))
      .sort((a, b) => a - b);
    for (let i = 1; i < timestamps.length; i += 1) {
      expect(timestamps[i]).toBeGreaterThan(timestamps[i - 1]);
    }
    for (const span of spans) {
      expect(span.captureTier).toBe("full");
      expect(span.actions.length).toBeGreaterThan(0);
    }
  });

  it("is fully reproducible for the same seed", () => {
    const first = generateSyntheticTrajectories({ scenarios: family.train, seed: 11 });
    const second = generateSyntheticTrajectories({ scenarios: family.train, seed: 11 });
    expect(first).toEqual(second);
  });
});
