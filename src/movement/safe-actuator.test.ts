import { EventEmitter } from "node:events";
import { describe, expect, it } from "vitest";
import type { MovementEvent } from "./events.js";
import { RecordingActuator, replayEpisode, type InputActuator } from "./replay.js";
import { SafeActuator, type SafeActuatorReleaseReport } from "./safe-actuator.js";

function manualTimers() {
  const timers = new Map<number, () => void>();
  let next = 1;
  return {
    setTimer: (fn: () => void) => {
      const id = next++;
      timers.set(id, fn);
      return id;
    },
    clearTimer: (id: unknown) => void timers.delete(id as number),
    fireAll: () => {
      const fns = [...timers.values()];
      timers.clear();
      for (const fn of fns) fn();
    },
    get pending() {
      return timers.size;
    },
  };
}

const flush = () => new Promise((resolve) => setImmediate(resolve));

describe("SafeActuator", () => {
  it("tracks held inputs across calls and releases at the last pointer position", async () => {
    const inner = new RecordingActuator();
    const safe = new SafeActuator(inner);
    await safe.perform({ type: "pointer.down", t: 10, x: 5, y: 5, button: "left" });
    await safe.perform({ type: "key.down", t: 12, key: "Shift", modifiers: ["shift"] });
    await safe.perform({ type: "pointer.move", t: 30, x: 80, y: 40 });
    expect(safe.held()).toEqual({ buttons: ["left"], keys: ["Shift"] });

    const report = await safe.releaseAll();
    expect(report.reason).toBe("manual");
    expect(report.released).toEqual([
      { type: "pointer.up", t: 30, x: 80, y: 40, button: "left" },
      { type: "key.up", t: 30, key: "Shift", modifiers: ["shift"] },
    ]);
    expect(safe.held()).toEqual({ buttons: [], keys: [] });
    expect((await safe.releaseAll()).released).toEqual([]);
  });

  it("idle watchdog releases a stalled drag and disarms when nothing is held", async () => {
    const timers = manualTimers();
    const reports: SafeActuatorReleaseReport[] = [];
    const inner = new RecordingActuator();
    const safe = new SafeActuator(inner, { idleTimeoutMs: 500, ...timers, onRelease: (r) => reports.push(r) });

    await safe.perform({ type: "pointer.move", t: 0, x: 1, y: 1 });
    expect(timers.pending).toBe(0);
    await safe.perform({ type: "pointer.down", t: 1, x: 1, y: 1, button: "left" });
    expect(timers.pending).toBe(1);
    await safe.perform({ type: "pointer.up", t: 2, x: 1, y: 1, button: "left" });
    expect(timers.pending).toBe(0);

    await safe.perform({ type: "pointer.down", t: 3, x: 2, y: 2, button: "right" });
    timers.fireAll();
    await flush();
    expect(reports).toHaveLength(1);
    expect(reports[0]!.reason).toBe("idle-timeout");
    expect(inner.performed.at(-1)).toMatchObject({ type: "pointer.up", button: "right" });
  });

  it("releases on abort and refuses new presses afterwards", async () => {
    const controller = new AbortController();
    const inner = new RecordingActuator();
    const safe = new SafeActuator(inner, { signal: controller.signal });
    await safe.perform({ type: "key.down", t: 0, key: "a" });
    controller.abort();
    await flush();
    expect(inner.performed.at(-1)).toEqual({ type: "key.up", t: 0, key: "a" });
    await expect(safe.perform({ type: "key.down", t: 1, key: "b" })).rejects.toThrow(/aborted/);
    await safe.perform({ type: "pointer.move", t: 1, x: 0, y: 0 });
  });

  it("guard() releases and rethrows on error; still releases when inner throws mid-press", async () => {
    let failNext = false;
    const performed: MovementEvent[] = [];
    const flaky: InputActuator = {
      perform(event) {
        performed.push(event);
        if (failNext) {
          failNext = false;
          throw new Error("os rejected event");
        }
      },
    };
    const safe = new SafeActuator(flaky);
    await expect(
      safe.guard(async (a) => {
        await a.perform({ type: "pointer.down", t: 0, x: 3, y: 4, button: "left" });
        failNext = true;
        await a.perform({ type: "key.down", t: 1, key: "x" });
      }),
    ).rejects.toThrow(/os rejected/);
    expect(performed.slice(2).map((e) => e.type).sort()).toEqual(["key.up", "pointer.up"]);
    expect(safe.held()).toEqual({ buttons: [], keys: [] });
  });

  it("records release failures without throwing", async () => {
    const safe = new SafeActuator({
      perform(event) {
        if (event.type === "pointer.up") throw new Error("boom");
      },
    });
    await safe.perform({ type: "pointer.down", t: 0, x: 0, y: 0, button: "middle" });
    const report = await safe.releaseAll();
    expect(report.released).toEqual([]);
    expect(report.failed).toHaveLength(1);
  });

  it("signal guard releases on SIGINT and detaches cleanly", async () => {
    const emitter = new EventEmitter();
    const inner = new RecordingActuator();
    const safe = new SafeActuator(inner);
    const detach = safe.installSignalGuard(emitter);
    await safe.perform({ type: "pointer.down", t: 0, x: 0, y: 0, button: "left" });
    emitter.emit("SIGINT");
    await flush();
    expect(safe.held().buttons).toEqual([]);
    detach();
    expect(emitter.listenerCount("SIGINT")).toBe(0);
    expect(emitter.listenerCount("SIGTERM")).toBe(0);
  });

  it("dispose releases, cancels the watchdog, and blocks late input", async () => {
    const timers = manualTimers();
    const safe = new SafeActuator(new RecordingActuator(), { idleTimeoutMs: 100, ...timers });
    await safe.perform({ type: "key.down", t: 0, key: "Meta" });
    const report = await safe.dispose();
    expect(report.reason).toBe("dispose");
    expect(report.released).toHaveLength(1);
    expect(timers.pending).toBe(0);
    expect(safe.isDisposed).toBe(true);
    await expect(safe.perform({ type: "pointer.move", t: 1, x: 0, y: 0 })).rejects.toThrow(/disposed/);
  });

  it("composes with replayEpisode as a drop-in actuator", async () => {
    const inner = new RecordingActuator();
    const safe = new SafeActuator(inner);
    const result = await replayEpisode(
      {
        version: 1,
        id: "e",
        label: "drag",
        createdAt: "2026-10-28T00:00:00.000Z",
        source: "simulated",
        screen: { width: 100, height: 100 },
        events: [
          { type: "pointer.down", t: 0, x: 10, y: 10, button: "left" },
          { type: "pointer.move", t: 5, x: 50, y: 50 },
          { type: "pointer.up", t: 10, x: 50, y: 50, button: "left" },
        ],
      },
      safe,
      { sleep: async () => {} },
    );
    expect(result.performed).toBe(3);
    expect(safe.held()).toEqual({ buttons: [], keys: [] });
    expect(inner.performed).toHaveLength(3);
  });
});
