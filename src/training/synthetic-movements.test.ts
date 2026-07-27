import { describe, expect, it } from "vitest";
import {
  DEFAULT_SYNTHETIC_TASKS,
  generateSyntheticMovements,
} from "./synthetic-movements.js";

describe("generateSyntheticMovements", () => {
  it("is deterministic for a given seed", () => {
    const a = generateSyntheticMovements({ seed: 42, sequenceCount: 10 });
    const b = generateSyntheticMovements({ seed: 42, sequenceCount: 10 });
    expect(a).toEqual(b);
    expect(a).toHaveLength(10);
  });

  it("produces different data for different seeds", () => {
    const a = generateSyntheticMovements({ seed: 1, sequenceCount: 10 });
    const b = generateSyntheticMovements({ seed: 2, sequenceCount: 10 });
    expect(a).not.toEqual(b);
  });

  it("emits only actions from the task grammar", () => {
    const allowed = new Set(DEFAULT_SYNTHETIC_TASKS.flatMap((task) => task.actions));
    const sequences = generateSyntheticMovements({ seed: 5, sequenceCount: 30 });
    for (const sequence of sequences) {
      expect(sequence.steps.length).toBeGreaterThan(0);
      for (const step of sequence.steps) {
        expect(allowed.has(step.action)).toBe(true);
      }
    }
  });

  it("respects a custom task grammar", () => {
    const sequences = generateSyntheticMovements({
      seed: 9,
      sequenceCount: 6,
      tasks: [{ name: "only", actions: ["a", "b"] }],
    });
    for (const sequence of sequences) {
      expect(sequence.steps.map((step) => step.action)).toEqual(["a", "b"]);
    }
  });

  it("returns an empty dataset when there are no tasks", () => {
    expect(generateSyntheticMovements({ seed: 1, sequenceCount: 5, tasks: [] })).toEqual([]);
  });

  it("applies bounded jitter to positional params", () => {
    const sequences = generateSyntheticMovements({ seed: 11, sequenceCount: 50, jitter: 4 });
    const moves = sequences.flatMap((sequence) =>
      sequence.steps.filter((step) => step.action === "mouse.move" && step.params),
    );
    expect(moves.length).toBeGreaterThan(0);
    for (const move of moves) {
      expect(typeof move.params?.x).toBe("number");
      expect(typeof move.params?.y).toBe("number");
    }
  });
});
