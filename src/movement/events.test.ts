import { describe, expect, it } from "vitest";
import {
  MOVEMENT_BOS,
  MOVEMENT_EOS,
  cellCenter,
  createCoordinateQuantizer,
  labelToken,
  quantizeCell,
  tokenizeEvent,
  tokenizeTrajectory,
  type MovementEvent,
} from "./events.js";

const quantizer = createCoordinateQuantizer({ width: 100, height: 100, cols: 10, rows: 10 });

describe("coordinate quantizer", () => {
  it("maps coordinates to grid cells and clamps out-of-range values", () => {
    expect(quantizeCell(quantizer, 5, 5)).toEqual({ col: 0, row: 0 });
    expect(quantizeCell(quantizer, 55, 25)).toEqual({ col: 5, row: 2 });
    expect(quantizeCell(quantizer, 999, -20)).toEqual({ col: 9, row: 0 });
  });

  it("cellCenter is the inverse of quantizeCell up to cell resolution", () => {
    const center = cellCenter(quantizer, 5, 2);
    expect(quantizeCell(quantizer, center.x, center.y)).toEqual({ col: 5, row: 2 });
  });

  it("rejects degenerate quantizers", () => {
    expect(() => createCoordinateQuantizer({ width: 0, height: 100, cols: 4, rows: 4 })).toThrow();
    expect(() => createCoordinateQuantizer({ width: 100, height: 100, cols: 0, rows: 4 })).toThrow();
  });
});

describe("tokenizeEvent", () => {
  it("quantizes pointer coordinates and stays robust to sub-cell jitter", () => {
    const a: MovementEvent = { kind: "mouse-down", ts: 0, x: 52, y: 24, button: "left" };
    const b: MovementEvent = { kind: "mouse-down", ts: 1, x: 55, y: 21, button: "left" };
    expect(tokenizeEvent(a, quantizer)).toBe("down:left:5,2");
    expect(tokenizeEvent(a, quantizer)).toBe(tokenizeEvent(b, quantizer));
  });

  it("buckets scroll deltas by sign and normalizes key classes", () => {
    expect(tokenizeEvent({ kind: "scroll", ts: 0, x: 5, y: 5, dx: 0, dy: 3 }, quantizer)).toBe("scroll:0+:0,0");
    expect(tokenizeEvent({ kind: "key-down", ts: 0, key: "a", modifiers: [] }, quantizer)).toBe("kd:char");
    expect(tokenizeEvent({ kind: "key-down", ts: 0, key: "7", modifiers: [] }, quantizer)).toBe("kd:digit");
    expect(tokenizeEvent({ kind: "key-down", ts: 0, key: "Enter", modifiers: [] }, quantizer)).toBe("kd:enter");
  });

  it("sorts modifiers into a canonical order", () => {
    const shuffled = tokenizeEvent({ kind: "key-down", ts: 0, key: "a", modifiers: ["shift", "ctrl"] }, quantizer);
    const canonical = tokenizeEvent({ kind: "key-down", ts: 0, key: "a", modifiers: ["ctrl", "shift"] }, quantizer);
    expect(shuffled).toBe("kd:ctrl+shift-char");
    expect(shuffled).toBe(canonical);
  });
});

describe("tokenizeTrajectory", () => {
  it("wraps events with intent-conditioning sentinels", () => {
    const tokens = tokenizeTrajectory(
      {
        id: "t1",
        label: "open-and-type",
        events: [{ kind: "window-focus", ts: 0, windowId: "editor" }],
      },
      quantizer,
    );
    expect(tokens[0]).toBe(MOVEMENT_BOS);
    expect(tokens[1]).toBe(labelToken("open-and-type"));
    expect(tokens.at(-1)).toBe(MOVEMENT_EOS);
    expect(tokens).toContain("focus:editor");
  });
});
