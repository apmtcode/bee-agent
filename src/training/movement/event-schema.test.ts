import { describe, expect, it } from "vitest";
import {
  normalizeMovementTrajectory,
  pointerPath,
  validateMovementTrajectory,
  type MovementTrajectory,
} from "./event-schema.js";

const sample: MovementTrajectory = {
  id: "t1",
  label: "click:save",
  target: { x: 10, y: 20 },
  events: [
    { kind: "pointer-move", ts: 0, x: 0, y: 0 },
    { kind: "pointer-move", ts: 16, x: 5, y: 10 },
    { kind: "pointer-down", ts: 32, x: 10, y: 20, button: "left" },
    { kind: "pointer-up", ts: 48, x: 10, y: 20, button: "left" },
    { kind: "key-down", ts: 64, key: "Enter" },
    { kind: "key-up", ts: 80, key: "Enter" },
  ],
};

describe("event-schema", () => {
  it("extracts the pointer path and skips key events", () => {
    const path = pointerPath(sample);
    expect(path).toHaveLength(4);
    expect(path[0]).toEqual({ x: 0, y: 0 });
    expect(path.at(-1)).toEqual({ x: 10, y: 20 });
  });

  it("validates a well-formed trajectory with no issues", () => {
    expect(validateMovementTrajectory(sample)).toEqual([]);
  });

  it("flags non-monotonic timestamps and bad fields", () => {
    const broken: MovementTrajectory = {
      id: "",
      label: "bad",
      events: [
        { kind: "pointer-move", ts: 10, x: 0, y: 0 },
        { kind: "pointer-move", ts: 5, x: Number.NaN, y: 0 },
        { kind: "key-down", ts: 12, key: "" },
      ],
    };
    const issues = validateMovementTrajectory(broken);
    expect(issues.some((issue) => issue.message.includes("missing an id"))).toBe(true);
    expect(issues.some((issue) => issue.message.includes("precedes previous"))).toBe(true);
    expect(issues.some((issue) => issue.message.includes("non-finite coordinates"))).toBe(true);
    expect(issues.some((issue) => issue.message.includes("empty key"))).toBe(true);
  });

  it("normalizes event order by timestamp", () => {
    const shuffled: MovementTrajectory = {
      id: "t2",
      label: "shuffled",
      events: [
        { kind: "pointer-up", ts: 48, x: 10, y: 20, button: "left" },
        { kind: "pointer-move", ts: 0, x: 0, y: 0 },
        { kind: "pointer-down", ts: 32, x: 10, y: 20, button: "left" },
      ],
    };
    const normalized = normalizeMovementTrajectory(shuffled);
    expect(normalized.events.map((event) => event.ts)).toEqual([0, 32, 48]);
    // Input is not mutated.
    expect(shuffled.events[0]!.ts).toBe(48);
  });
});
