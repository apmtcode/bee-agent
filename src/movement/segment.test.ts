import { describe, expect, it } from "vitest";
import { validateMovementEpisode, type MovementEpisode, type MovementEvent } from "./events.js";
import { TemplateMovementModel } from "./model.js";
import { segmentEpisode } from "./segment.js";
import { synthesizeDrag } from "./source.js";

function session(events: MovementEvent[], label = "session"): MovementEpisode {
  return {
    version: 1,
    id: "s1",
    label,
    createdAt: "2026-10-27T00:00:00.000Z",
    source: "simulated",
    screen: { width: 1920, height: 1080 },
    events,
  };
}

describe("segmentEpisode", () => {
  it("splits a mixed session into move-to, click, drag, scroll and typing segments", () => {
    const events: MovementEvent[] = [
      { type: "window.focus", t: 0, appId: "com.finder", title: "Finder" },
      { type: "pointer.move", t: 10, x: 100, y: 100 },
      { type: "pointer.move", t: 30, x: 150, y: 120 },
      { type: "pointer.move", t: 50, x: 200, y: 140 },
      { type: "pointer.down", t: 70, x: 200, y: 140, button: "left" },
      { type: "pointer.up", t: 120, x: 201, y: 141, button: "left" },
      ...synthesizeDrag({ from: { x: 300, y: 300 }, to: { x: 700, y: 500 }, startT: 2000, durationMs: 400 }),
      { type: "scroll", t: 3000, x: 500, y: 500, dx: 0, dy: -3 },
      { type: "scroll", t: 3050, x: 500, y: 500, dx: 0, dy: -3 },
      { type: "window.focus", t: 3500, appId: "com.editor" },
      { type: "key.down", t: 4000, key: "h" },
      { type: "key.up", t: 4050, key: "h" },
      { type: "key.down", t: 4150, key: "i" },
      { type: "key.up", t: 4200, key: "i" },
    ];
    const { segments, counts, droppedEvents } = segmentEpisode(session(events));

    expect(segments.map((s) => s.label)).toEqual([
      "session/move-to",
      "session/click",
      "session/drag",
      "session/scroll",
      "session/typing",
    ]);
    expect(counts).toEqual({ "move-to": 1, click: 1, drag: 1, scroll: 1, typing: 1 });
    expect(droppedEvents).toBe(0);
    for (const segment of segments) {
      expect(validateMovementEpisode(segment)).toEqual([]);
      expect(segment.events[0]!.t).toBe(0);
    }
    expect(segments.map((s) => s.id)).toEqual(["s1#0", "s1#1", "s1#2", "s1#3", "s1#4"]);
    // Segments carry the app context they happened in.
    expect(segments[0]!.events[0]).toMatchObject({ type: "window.focus", appId: "com.finder" });
    expect(segments[4]!.events[0]).toMatchObject({ type: "window.focus", appId: "com.editor", t: 0 });
    expect(segments[4]!.events.filter((e) => e.type === "key.down")).toHaveLength(2);
  });

  it("splits typing bursts and move runs on idle gaps, and drops jitter", () => {
    const events: MovementEvent[] = [
      { type: "pointer.move", t: 0, x: 10, y: 10 },
      { type: "pointer.move", t: 20, x: 11, y: 11 }, // jitter: < minMoveDistance
      { type: "key.down", t: 100, key: "a" },
      { type: "key.up", t: 150, key: "a" },
      { type: "key.down", t: 2000, key: "b" }, // idle gap → new burst
      { type: "key.up", t: 2050, key: "b" },
      { type: "pointer.move", t: 3000, x: 0, y: 0 },
      { type: "pointer.move", t: 3020, x: 50, y: 0 },
      { type: "pointer.move", t: 5000, x: 400, y: 0 }, // idle gap → new move run (single point = jitter)
    ];
    const { segments, counts, droppedEvents } = segmentEpisode(session(events), { includeFocus: false });
    expect(counts).toEqual({ typing: 2, move: 1 });
    expect(droppedEvents).toBe(3);
    expect(segments.map((s) => s.label)).toEqual(["session/typing", "session/typing", "session/move"]);
  });

  it("keeps modifier+pointer chords balanced and drops unreleased presses", () => {
    const events: MovementEvent[] = [
      { type: "key.down", t: 0, key: "Shift", modifiers: ["shift"] },
      { type: "pointer.down", t: 50, x: 10, y: 10, button: "left" },
      { type: "pointer.up", t: 90, x: 10, y: 10, button: "left" },
      { type: "key.up", t: 2000, key: "Shift" }, // long hold still one segment
      { type: "pointer.down", t: 3000, x: 5, y: 5, button: "right" },
      { type: "pointer.move", t: 3100, x: 50, y: 50 },
    ];
    const { segments, counts, droppedEvents } = segmentEpisode(session(events));
    expect(counts).toEqual({ chord: 1 });
    expect(segments[0]!.events.map((e) => e.type)).toEqual(["key.down", "pointer.down", "pointer.up", "key.up"]);
    expect(droppedEvents).toBe(2);
  });

  it("filters kinds and supports custom labels", () => {
    const events: MovementEvent[] = [
      { type: "pointer.down", t: 0, x: 10, y: 10, button: "left" },
      { type: "pointer.up", t: 50, x: 10, y: 10, button: "left" },
      ...synthesizeDrag({ from: { x: 0, y: 0 }, to: { x: 300, y: 0 }, startT: 1000 }),
    ];
    const { segments, droppedEvents } = segmentEpisode(session(events), {
      kinds: ["drag"],
      label: (kind) => `custom-${kind}`,
    });
    expect(segments.map((s) => s.label)).toEqual(["custom-drag"]);
    expect(droppedEvents).toBe(2);
  });

  it("turns one long session into enough demonstrations to train and retarget a drag", () => {
    const events: MovementEvent[] = [];
    const drags = [
      [{ x: 100, y: 100 }, { x: 500, y: 300 }],
      [{ x: 200, y: 600 }, { x: 800, y: 400 }],
      [{ x: 900, y: 100 }, { x: 1200, y: 700 }],
    ] as const;
    drags.forEach(([from, to], k) => {
      events.push({ type: "pointer.move", t: k * 5000, x: from.x - 80, y: from.y - 40 });
      events.push(...synthesizeDrag({ from, to, startT: k * 5000 + 1000, durationMs: 500, jitter: 1, seed: k + 1 }));
    });
    const { segments, counts } = segmentEpisode(session(events, "desk"), { kinds: ["drag"] });
    expect(counts.drag).toBe(3);

    const model = new TemplateMovementModel();
    const report = model.train(segments);
    expect(report.labels).toEqual([expect.objectContaining({ label: "desk/drag", demonstrations: 3 })]);
    const out = model.generate({ label: "desk/drag", start: { x: 1000, y: 900 }, end: { x: 400, y: 200 } });
    const pointer = out.events.filter((e) => e.type !== "window.focus" && e.type !== "key.down" && e.type !== "key.up");
    expect(pointer[0]).toMatchObject({ type: "pointer.down", x: 1000, y: 900 });
    expect(pointer.at(-1)).toMatchObject({ type: "pointer.up" });
  });
});
