import { describe, expect, it } from "vitest";
import { createSeededRandom, generateSyntheticMovementDataset } from "./synthetic.js";

describe("createSeededRandom", () => {
  it("is deterministic for a given seed", () => {
    const a = createSeededRandom(42);
    const b = createSeededRandom(42);
    const seqA = [a(), a(), a()];
    const seqB = [b(), b(), b()];
    expect(seqA).toEqual(seqB);
    expect(seqA.every((value) => value >= 0 && value < 1)).toBe(true);
  });

  it("diverges across seeds", () => {
    expect(createSeededRandom(1)()).not.toBe(createSeededRandom(2)());
  });
});

describe("generateSyntheticMovementDataset", () => {
  it("produces the requested number of non-empty sequences", () => {
    const dataset = generateSyntheticMovementDataset({ seed: 7, count: 12 });
    expect(dataset.sequences).toHaveLength(12);
    expect(dataset.sequences.every((sequence) => sequence.events.length > 0)).toBe(true);
  });

  it("is byte-identical for identical options", () => {
    const a = generateSyntheticMovementDataset({ seed: 7, count: 5 });
    const b = generateSyntheticMovementDataset({ seed: 7, count: 5 });
    expect(JSON.stringify(a)).toBe(JSON.stringify(b));
  });

  it("emits monotonically increasing timestamps within a sequence", () => {
    const dataset = generateSyntheticMovementDataset({ seed: 3, count: 4 });
    for (const sequence of dataset.sequences) {
      for (let i = 1; i < sequence.events.length; i += 1) {
        expect(sequence.events[i].ts).toBeGreaterThan(sequence.events[i - 1].ts);
      }
    }
  });

  it("restricts to the provided task families", () => {
    const dataset = generateSyntheticMovementDataset({
      seed: 9,
      count: 6,
      tasks: [
        {
          name: "only",
          build: () => [{ action: "tap", descriptor: "x" }],
        },
      ],
    });
    expect(dataset.sequences.every((sequence) => sequence.id.includes("only"))).toBe(true);
    expect(dataset.sequences.every((sequence) => sequence.events[0].token === "tap:x")).toBe(true);
  });
});
