import { describe, expect, it } from "vitest";
import {
  buildMovementDataset,
  parseDatasetJsonl,
  serializeDatasetJsonl,
  trajectoryToExample,
} from "./dataset.js";
import { generateClickGesture, generateTypingGesture } from "./synthetic-stream.js";

describe("dataset", () => {
  it("builds a normalized profile from a click gesture", () => {
    const gesture = generateClickGesture({ id: "g", start: { x: 10, y: 10 }, target: { x: 110, y: 60 }, steps: 8 });
    const example = trajectoryToExample(gesture);
    expect(example).toBeDefined();
    expect(example!.start).toEqual({ x: 10, y: 10 });
    expect(example!.target).toEqual({ x: 110, y: 60 });
    expect(example!.stepCount).toBe(8);
    // Profile starts at fraction 0 and ends at fraction 1.
    expect(example!.profile[0]).toMatchObject({ t: 0, fx: 0, fy: 0 });
    expect(example!.profile.at(-1)).toMatchObject({ t: 1, fx: 1, fy: 1 });
    expect(example!.stepMs).toBeGreaterThan(0);
  });

  it("skips gestures with no pointer motion", () => {
    const typing = generateTypingGesture({ id: "t", text: "abc" });
    expect(trajectoryToExample(typing)).toBeUndefined();
    const dataset = buildMovementDataset([typing]);
    expect(dataset.examples).toHaveLength(0);
  });

  it("round-trips through JSONL serialization", () => {
    const dataset = buildMovementDataset([
      generateClickGesture({ id: "a", start: { x: 0, y: 0 }, target: { x: 50, y: 50 }, steps: 6 }),
      generateClickGesture({ id: "b", start: { x: 0, y: 0 }, target: { x: 90, y: 10 }, steps: 6 }),
    ]);
    const text = serializeDatasetJsonl(dataset);
    expect(text.split("\n")).toHaveLength(2);
    const parsed = parseDatasetJsonl(text);
    expect(parsed).toEqual(dataset);
  });

  it("handles a purely horizontal target (zero dy) without NaNs", () => {
    const gesture = generateClickGesture({ id: "h", start: { x: 0, y: 5 }, target: { x: 100, y: 5 }, steps: 5 });
    const example = trajectoryToExample(gesture)!;
    expect(example.profile.every((sample) => Number.isFinite(sample.fx) && Number.isFinite(sample.fy))).toBe(true);
    expect(example.profile.at(-1)!.fy).toBe(1);
  });
});
