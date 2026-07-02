import { describe, expect, it } from "vitest";
import {
  BUILTIN_MOVEMENT_TEMPLATES,
  generateSyntheticDataset,
  generateSyntheticSequence,
} from "./synthetic.js";

describe("synthetic movement generator", () => {
  it("is deterministic for a given template + seed", () => {
    const template = BUILTIN_MOVEMENT_TEMPLATES[0];
    const a = generateSyntheticSequence(template, 42);
    const b = generateSyntheticSequence(template, 42);
    expect(a).toEqual(b);
  });

  it("produces different content for different seeds", () => {
    const template = BUILTIN_MOVEMENT_TEMPLATES[0];
    const a = generateSyntheticSequence(template, 1);
    const b = generateSyntheticSequence(template, 2);
    expect(a.trajectoryId).not.toBe(b.trajectoryId);
    const summariesA = a.events.map((event) => event.summary).join("|");
    const summariesB = b.events.map((event) => event.summary).join("|");
    expect(summariesA).not.toBe(summariesB);
  });

  it("fills slot placeholders (no braces remain)", () => {
    const sequence = generateSyntheticSequence(BUILTIN_MOVEMENT_TEMPLATES[0], 7);
    for (const event of sequence.events) {
      expect(event.summary).not.toMatch(/\{\w+\}/);
    }
  });

  it("emits strictly increasing timestamps", () => {
    const sequence = generateSyntheticSequence(BUILTIN_MOVEMENT_TEMPLATES[1], 9);
    for (let i = 1; i < sequence.events.length; i += 1) {
      expect(sequence.events[i].ts).toBeGreaterThan(sequence.events[i - 1].ts);
    }
  });

  it("generates a dataset with count-per-template sequences", () => {
    const sequences = generateSyntheticDataset({ countPerTemplate: 3 });
    expect(sequences).toHaveLength(BUILTIN_MOVEMENT_TEMPLATES.length * 3);
  });
});
