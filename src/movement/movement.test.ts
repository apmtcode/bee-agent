import { describe, expect, it } from "vitest";
import {
  decodeMovementDataset,
  encodeMovementDataset,
  pointerAnchors,
  validateMovementEpisode,
  type MovementEpisode,
} from "./events.js";
import { TemplateMovementModel, evaluateMovementModel } from "./model.js";
import { RecordingActuator, replayEpisode } from "./replay.js";
import { SimulatedInputEventSource, recordEpisode, synthesizeDrag } from "./source.js";

const fixedNow = () => new Date("2026-10-22T00:00:00.000Z");

async function dragEpisode(id: string, from: { x: number; y: number }, to: { x: number; y: number }, seed = 1) {
  const source = new SimulatedInputEventSource(
    synthesizeDrag({ from, to, startT: 1_000_000, jitter: 1.5, seed, durationMs: 500 }),
  );
  return await recordEpisode(source, { id, label: "drag-file", now: fixedNow });
}

describe("movement events + dataset", () => {
  it("records a simulated source into a valid, time-rebased episode", async () => {
    const episode = await dragEpisode("ep-1", { x: 100, y: 100 }, { x: 500, y: 300 });
    expect(validateMovementEpisode(episode)).toEqual([]);
    expect(episode.events[0]).toMatchObject({ type: "pointer.down", t: 0, x: 100, y: 100 });
    expect(episode.events.at(-1)).toMatchObject({ type: "pointer.up", t: 500, x: 500, y: 300 });
    expect(episode.source).toBe("simulated");
  });

  it("round-trips through JSONL and rejects invalid lines without failing the file", async () => {
    const episode = await dragEpisode("ep-1", { x: 10, y: 10 }, { x: 200, y: 50 });
    const jsonl = encodeMovementDataset([episode]) + '{"version":1,"id":"bad"}\nnot json\n';
    const decoded = decodeMovementDataset(jsonl);
    expect(decoded.episodes).toEqual([episode]);
    expect(decoded.rejected.map((r) => r.line)).toEqual([3, 4]);
    expect(() => decodeMovementDataset('{"kind":"other"}\n')).toThrow(/not a movement dataset/);
  });

  it("flags unknown event types and decreasing timestamps", () => {
    const errors = validateMovementEpisode({
      version: 1,
      id: "x",
      label: "y",
      screen: { width: 10, height: 10 },
      events: [
        { type: "pointer.move", t: 5, x: 1, y: 1 },
        { type: "pointer.move", t: 2, x: 1, y: 1 },
        { type: "teleport", t: 6 },
      ],
    });
    expect(errors).toContain("events[1]: timestamps must be non-decreasing");
    expect(errors.some((e) => e.includes("unknown event type"))).toBe(true);
  });
});

describe("recordEpisode privacy", () => {
  it("drops keystrokes while a sensitive app is focused and redacts others on request", async () => {
    const source = new SimulatedInputEventSource([
      { type: "window.focus", t: 0, appId: "com.editor" },
      { type: "key.down", t: 10, key: "a" },
      { type: "key.up", t: 20, key: "a" },
      { type: "window.focus", t: 30, appId: "com.1password.app" },
      { type: "key.down", t: 40, key: "s" },
      { type: "key.up", t: 50, key: "s" },
      { type: "window.focus", t: 60, appId: "com.editor" },
      { type: "key.down", t: 70, key: "Enter" },
    ]);
    const episode = await recordEpisode(source, { id: "k", label: "typing", redactKeys: true, now: fixedNow });
    const keys = episode.events.flatMap((e) => (e.type === "key.down" || e.type === "key.up" ? [e.key] : []));
    expect(keys).toEqual(["•", "•", "Enter"]);
  });

  it("honours maxEvents and maxDurationMs", async () => {
    const script = synthesizeDrag({ from: { x: 0, y: 0 }, to: { x: 100, y: 0 }, durationMs: 1000, steps: 10 });
    const source = new SimulatedInputEventSource(script);
    expect((await recordEpisode(source, { id: "a", label: "l", maxEvents: 3 })).events).toHaveLength(3);
    const capped = await recordEpisode(source, { id: "b", label: "l", maxDurationMs: 250 });
    expect(capped.events.every((e) => e.t <= 250)).toBe(true);
  });
});

describe("TemplateMovementModel", () => {
  it("repeats a learned movement at its original endpoints", async () => {
    const demo = await dragEpisode("ep-1", { x: 100, y: 100 }, { x: 500, y: 300 });
    const model = new TemplateMovementModel();
    const report = model.train([demo]);
    expect(report.labels).toEqual([{ label: "drag-file", demonstrations: 1, meanDurationMs: 500 }]);
    const out = model.generate({ label: "drag-file" });
    expect(pointerAnchors(out)).toEqual(pointerAnchors(demo));
    expect(out.events.map((e) => e.type)).toEqual(demo.events.map((e) => e.type));
  });

  it("generalizes to new, rotated/scaled endpoints with low held-out error", async () => {
    const train = await Promise.all([
      dragEpisode("t1", { x: 100, y: 100 }, { x: 500, y: 100 }, 1),
      dragEpisode("t2", { x: 200, y: 600 }, { x: 600, y: 400 }, 2),
      dragEpisode("t3", { x: 900, y: 200 }, { x: 900, y: 700 }, 3),
    ]);
    const heldOut = await Promise.all([
      dragEpisode("h1", { x: 1500, y: 900 }, { x: 1100, y: 300 }, 4),
      dragEpisode("h2", { x: 50, y: 1000 }, { x: 1800, y: 900 }, 5),
    ]);
    const model = new TemplateMovementModel();
    model.train(train);
    const result = evaluateMovementModel(model, heldOut);
    expect(result.episodes).toBe(2);
    // Same movement "shape" transferred to unseen geometry: error is a small
    // fraction of the movement length (jitter alone contributes ~1-2px).
    expect(result.meanNormalizedError).toBeLessThan(0.02);

    const out = model.generate({ label: "drag-file", start: { x: 1500, y: 900 }, end: { x: 1100, y: 300 } });
    expect(out.events[0]).toMatchObject({ type: "pointer.down", x: 1500, y: 900 });
    expect(out.events.at(-1)).toMatchObject({ type: "pointer.up" });
    expect(validateMovementEpisode(out)).toEqual([]);
  });

  it("rejects outlier demonstrations via the medoid and persists through JSON", async () => {
    const good = await Promise.all([
      dragEpisode("g1", { x: 0, y: 0 }, { x: 400, y: 0 }, 1),
      dragEpisode("g2", { x: 0, y: 100 }, { x: 400, y: 100 }, 2),
    ]);
    const wild: MovementEpisode = await recordEpisode(
      new SimulatedInputEventSource(synthesizeDrag({ from: { x: 0, y: 0 }, to: { x: 400, y: 0 }, curvature: 1.2 })),
      { id: "wild", label: "drag-file", now: fixedNow },
    );
    const model = new TemplateMovementModel();
    model.train([...good, wild]);
    const restored = TemplateMovementModel.fromJSON(JSON.parse(JSON.stringify(model.toJSON())));
    const result = evaluateMovementModel(restored, good);
    expect(result.meanNormalizedError).toBeLessThan(0.02);
    expect(restored.labels()).toEqual(["drag-file"]);
  });

  it("scales duration sub-linearly with distance and clamps to the screen", async () => {
    const demo = await dragEpisode("d", { x: 100, y: 100 }, { x: 200, y: 100 });
    const model = new TemplateMovementModel();
    model.train([demo]);
    const far = model.generate({ label: "drag-file", start: { x: 0, y: 0 }, end: { x: 400, y: 0 } });
    expect(far.events.at(-1)!.t).toBe(1000); // 500ms * sqrt(4)
    const off = model.generate({ label: "drag-file", start: { x: 1900, y: 10 }, end: { x: 2500, y: 10 } });
    for (const e of off.events) if ("x" in e) expect(e.x).toBeLessThanOrEqual(1919);
    expect(() => model.generate({ label: "nope" })).toThrow(/no learned movement/);
  });

  it("handles in-place movements (clicks) by translation", async () => {
    const click = await recordEpisode(
      new SimulatedInputEventSource([
        { type: "pointer.down", t: 0, x: 300, y: 300, button: "left" },
        { type: "pointer.up", t: 80, x: 301, y: 300, button: "left" },
      ]),
      { id: "c", label: "click", now: fixedNow },
    );
    const model = new TemplateMovementModel();
    model.train([click]);
    const out = model.generate({ label: "click", start: { x: 50, y: 60 } });
    expect(out.events).toEqual([
      { type: "pointer.down", t: 0, x: 50, y: 60, button: "left" },
      { type: "pointer.up", t: 80, x: 51, y: 60, button: "left" },
    ]);
  });
});

describe("replayEpisode", () => {
  it("replays with scaled timing through a mock actuator", async () => {
    const episode = await dragEpisode("r", { x: 10, y: 10 }, { x: 300, y: 10 });
    const actuator = new RecordingActuator();
    const sleeps: number[] = [];
    const result = await replayEpisode(episode, actuator, { speed: 2, sleep: async (ms) => void sleeps.push(ms) });
    expect(result).toMatchObject({ performed: episode.events.length, aborted: false, sleptMs: 250 });
    expect(actuator.performed).toEqual(episode.events);
  });

  it("releases held buttons and keys when aborted or capped", async () => {
    const episode = await dragEpisode("r", { x: 10, y: 10 }, { x: 300, y: 10 });
    episode.events.unshift({ type: "key.down", t: 0, key: "Shift" });
    const actuator = new RecordingActuator();
    const result = await replayEpisode(episode, actuator, { maxEvents: 4, sleep: async () => {} });
    expect(result.aborted).toBe(true);
    const tail = actuator.performed.slice(-2).map((e) => e.type);
    expect(tail).toEqual(["pointer.up", "key.up"]);
  });

  it("filters by allowTypes and rejects non-positive speed", async () => {
    const episode = await dragEpisode("r", { x: 10, y: 10 }, { x: 300, y: 10 });
    const actuator = new RecordingActuator();
    const result = await replayEpisode(episode, actuator, { allowTypes: ["pointer.move"], sleep: async () => {} });
    expect(actuator.performed.every((e) => e.type === "pointer.move")).toBe(true);
    expect(result.skipped).toBe(2);
    await expect(replayEpisode(episode, actuator, { speed: 0 })).rejects.toThrow(/speed/);
  });
});
