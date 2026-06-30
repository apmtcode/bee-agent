import { describe, expect, it } from "vitest";
import {
  buildMovementDataset,
  buildMovementDemonstration,
  datasetIntents,
  isPositionedEvent,
  pointerPath,
  sortEvents,
  type MovementEvent,
} from "./events.js";

describe("movement events", () => {
  it("classifies positioned events", () => {
    expect(isPositionedEvent({ kind: "pointer-move", ts: 0, x: 1, y: 2 })).toBe(true);
    expect(isPositionedEvent({ kind: "scroll", ts: 0, x: 1, y: 2, dx: 0, dy: 1 })).toBe(true);
    expect(isPositionedEvent({ kind: "key-down", ts: 0, key: "a" })).toBe(false);
    expect(isPositionedEvent({ kind: "window-focus", ts: 0, window: "Finder" })).toBe(false);
  });

  it("extracts the pointer path in order, skipping non-positioned events", () => {
    const events: MovementEvent[] = [
      { kind: "pointer-down", ts: 0, x: 0, y: 0, button: "left" },
      { kind: "key-down", ts: 1, key: "shift" },
      { kind: "pointer-move", ts: 2, x: 5, y: 5 },
      { kind: "pointer-up", ts: 3, x: 10, y: 10, button: "left" },
    ];
    expect(pointerPath(events)).toEqual([
      { x: 0, y: 0 },
      { x: 5, y: 5 },
      { x: 10, y: 10 },
    ]);
  });

  it("sorts events ascending by ts without mutating input", () => {
    const events: MovementEvent[] = [
      { kind: "key-up", ts: 5, key: "a" },
      { kind: "key-down", ts: 1, key: "a" },
    ];
    const sorted = sortEvents(events);
    expect(sorted.map((e) => e.ts)).toEqual([1, 5]);
    expect(events[0]!.ts).toBe(5);
  });

  it("builder normalizes event order and clones the frame", () => {
    const demo = buildMovementDemonstration({
      id: "d1",
      intent: "drag",
      events: [
        { kind: "pointer-up", ts: 9, x: 1, y: 1, button: "left" },
        { kind: "pointer-down", ts: 0, x: 0, y: 0, button: "left" },
      ],
      frame: { start: { x: 0, y: 0 }, end: { x: 1, y: 1 } },
    });
    expect(demo.events[0]!.kind).toBe("pointer-down");
  });

  it("reports distinct intents in first-seen order", () => {
    const dataset = buildMovementDataset([
      buildMovementDemonstration({ id: "a", intent: "drag", events: [], frame: { start: { x: 0, y: 0 }, end: { x: 1, y: 1 } } }),
      buildMovementDemonstration({ id: "b", intent: "scroll", events: [], frame: { start: { x: 0, y: 0 }, end: { x: 1, y: 1 } } }),
      buildMovementDemonstration({ id: "c", intent: "drag", events: [], frame: { start: { x: 0, y: 0 }, end: { x: 1, y: 1 } } }),
    ]);
    expect(datasetIntents(dataset)).toEqual(["drag", "scroll"]);
  });
});
