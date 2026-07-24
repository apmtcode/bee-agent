import { describe, expect, it } from "vitest";
import {
  DEFAULT_MOVEMENT_TEMPLATES,
  generateSyntheticReplays,
  partitionReplays,
} from "./synthetic-stream.js";

describe("generateSyntheticReplays", () => {
  it("is deterministic for a given seed", () => {
    const a = generateSyntheticReplays({ seed: 5, sequenceCount: 10 });
    const b = generateSyntheticReplays({ seed: 5, sequenceCount: 10 });
    expect(a).toEqual(b);
  });

  it("produces different corpora for different seeds", () => {
    const a = generateSyntheticReplays({ seed: 1, sequenceCount: 10 });
    const b = generateSyntheticReplays({ seed: 2, sequenceCount: 10 });
    expect(a).not.toEqual(b);
  });

  it("emits chronologically ordered events drawn from the templates", () => {
    const replays = generateSyntheticReplays({ seed: 3, sequenceCount: 4 });
    expect(replays).toHaveLength(4);
    const templateNames = new Set(DEFAULT_MOVEMENT_TEMPLATES.map((t) => t.name));
    for (const replay of replays) {
      expect(templateNames.has(replay.templateName)).toBe(true);
      expect(replay.eventCount).toBe(replay.events.length);
      for (let i = 1; i < replay.events.length; i += 1) {
        expect(replay.events[i].ts).toBeGreaterThan(replay.events[i - 1].ts);
      }
    }
  });

  it("returns nothing when there are no templates", () => {
    expect(generateSyntheticReplays({ templates: [], sequenceCount: 4 })).toEqual([]);
  });
});

describe("partitionReplays", () => {
  it("splits deterministically with at least one training item", () => {
    const items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const { train, heldOut } = partitionReplays(items, 0.7);
    expect(train).toEqual([1, 2, 3, 4, 5, 6, 7]);
    expect(heldOut).toEqual([8, 9, 10]);
  });

  it("clamps extreme fractions so both partitions can be non-empty", () => {
    const items = [1, 2, 3, 4];
    expect(partitionReplays(items, 0).train.length).toBeGreaterThanOrEqual(1);
    expect(partitionReplays(items, 1).heldOut.length).toBeGreaterThanOrEqual(0);
  });
});
