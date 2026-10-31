import { describe, expect, it } from "vitest";
import type { MovementEpisode, MovementEvent } from "./events.js";
import { GovernedActuator, GovernorViolationError, type GovernorViolation } from "./governor.js";
import { RecordingActuator, replayEpisode } from "./replay.js";
import { SafeActuator } from "./safe-actuator.js";

function fakeClock(start = 0) {
  let now = start;
  return {
    now: () => now,
    advance: (ms: number) => {
      now += ms;
    },
    sleep: async (ms: number) => {
      now += ms;
    },
  };
}

const move = (t: number, x: number, y: number): MovementEvent => ({ type: "pointer.move", t, x, y });

describe("GovernedActuator", () => {
  it("rejects events over the rate limit in reject mode, but never rejects releases", async () => {
    const clock = fakeClock();
    const inner = new RecordingActuator();
    const gov = new GovernedActuator(inner, { maxEventsPerSecond: 2, now: clock.now, sleep: clock.sleep });
    await gov.perform({ type: "key.down", t: 0, key: "a" });
    await gov.perform({ type: "key.down", t: 1, key: "b" });
    await expect(gov.perform({ type: "key.down", t: 2, key: "c" })).rejects.toBeInstanceOf(GovernorViolationError);
    await gov.perform({ type: "key.up", t: 3, key: "a" });
    await gov.perform({ type: "key.up", t: 4, key: "b" });
    expect(inner.performed.map((e) => e.type)).toEqual(["key.down", "key.down", "key.up", "key.up"]);
    expect(gov.stats()).toMatchObject({ forwarded: 4, rejected: 1 });
    clock.advance(1001);
    await gov.perform({ type: "key.down", t: 5, key: "c" });
    expect(gov.stats().forwarded).toBe(5);
  });

  it("delays rate-limited events in clamp mode instead of dropping them", async () => {
    const clock = fakeClock();
    const inner = new RecordingActuator();
    const violations: GovernorViolation[] = [];
    const gov = new GovernedActuator(inner, {
      maxEventsPerSecond: 3,
      mode: "clamp",
      now: clock.now,
      sleep: clock.sleep,
      onViolation: (v) => violations.push(v),
    });
    for (let i = 0; i < 7; i += 1) {
      await gov.perform(move(i, i, i));
    }
    expect(inner.performed).toHaveLength(7);
    expect(gov.stats().delayed).toBe(2);
    expect(violations.every((v) => v.kind === "rate" && v.action === "delayed")).toBe(true);
    // 7 events at 3/s need at least two full windows.
    expect(clock.now()).toBeGreaterThanOrEqual(2000);
  });

  it("rejects pointer teleports above the speed cap, clamps them in clamp mode", async () => {
    const clock = fakeClock();
    const rejecting = new GovernedActuator(new RecordingActuator(), { maxPointerSpeedPxPerMs: 2, now: clock.now });
    await rejecting.perform(move(0, 0, 0));
    clock.advance(10);
    await rejecting.perform(move(10, 20, 0)); // exactly 2px/ms: allowed
    clock.advance(10);
    await expect(rejecting.perform(move(20, 1000, 0))).rejects.toThrow(/velocity/);

    const clock2 = fakeClock();
    const inner = new RecordingActuator();
    const clamping = new GovernedActuator(inner, { maxPointerSpeedPxPerMs: 2, mode: "clamp", now: clock2.now });
    await clamping.perform(move(0, 0, 0));
    clock2.advance(10);
    await clamping.perform(move(10, 300, 400)); // 500px in 10ms → clamp to 20px along the same direction
    const last = inner.performed.at(-1) as Extract<MovementEvent, { x: number }>;
    expect(last.x).toBeCloseTo(12);
    expect(last.y).toBeCloseTo(16);
    expect(clamping.stats().clamped).toBe(1);
  });

  it("confines pointer events to allowed regions and clamps releases rather than rejecting them", async () => {
    const inner = new RecordingActuator();
    const gov = new GovernedActuator(inner, { regions: [{ x: 100, y: 100, width: 200, height: 100 }] });
    await gov.perform({ type: "pointer.down", t: 0, x: 150, y: 150, button: "left" });
    await expect(gov.perform(move(1, 50, 50))).rejects.toThrow(/region/);
    await expect(
      gov.perform({ type: "pointer.down", t: 2, x: 400, y: 150, button: "right" }),
    ).rejects.toBeInstanceOf(GovernorViolationError);
    await gov.perform({ type: "pointer.up", t: 3, x: 999, y: 999, button: "left" });
    expect(inner.performed.at(-1)).toMatchObject({ type: "pointer.up", x: 299, y: 199 });
  });

  it("clamps to the nearest of several regions", async () => {
    const inner = new RecordingActuator();
    const gov = new GovernedActuator(inner, {
      mode: "clamp",
      regions: [
        { x: 0, y: 0, width: 10, height: 10 },
        { x: 100, y: 0, width: 10, height: 10 },
      ],
    });
    await gov.perform(move(0, 95, 5));
    expect(inner.performed[0]).toMatchObject({ x: 100, y: 5 });
  });

  it("followFocus retargets the allowed region to the focused window's bounds", async () => {
    const inner = new RecordingActuator();
    const gov = new GovernedActuator(inner, { followFocus: true });
    await gov.perform(move(0, 5, 5)); // no regions yet → whole screen
    await gov.perform({
      type: "window.focus",
      t: 1,
      appId: "com.example.editor",
      bounds: { x: 500, y: 500, width: 100, height: 100 },
    });
    expect(gov.allowedRegions()).toEqual([{ x: 500, y: 500, width: 100, height: 100 }]);
    await expect(gov.perform(move(2, 5, 5))).rejects.toThrow(/region/);
    await gov.perform(move(3, 550, 550));
    expect(inner.performed).toHaveLength(3);
  });

  it("validates its configuration", () => {
    const inner = new RecordingActuator();
    expect(() => new GovernedActuator(inner, { maxEventsPerSecond: 0 })).toThrow(/maxEventsPerSecond/);
    expect(() => new GovernedActuator(inner, { maxPointerSpeedPxPerMs: -1 })).toThrow(/maxPointerSpeed/);
    expect(() => new GovernedActuator(inner, { regions: [{ x: 0, y: 0, width: 0, height: 5 }] })).toThrow(/region/);
  });

  it("stacks over SafeActuator: a rejected mid-drag move still leaves no button held", async () => {
    const os = new RecordingActuator();
    const safe = new SafeActuator(os);
    const gov = new GovernedActuator(safe, { regions: [{ x: 0, y: 0, width: 100, height: 100 }] });
    const episode: MovementEpisode = {
      version: 1,
      id: "runaway",
      label: "drag",
      createdAt: "2026-10-31T00:00:00.000Z",
      source: "simulated",
      screen: { width: 1920, height: 1080 },
      events: [
        { type: "pointer.down", t: 0, x: 10, y: 10, button: "left" },
        move(5, 50, 50),
        move(10, 900, 900), // escapes the window → rejected
        { type: "pointer.up", t: 15, x: 60, y: 60, button: "left" },
      ],
    };
    await expect(safe.guard(() => replayEpisode(episode, gov, { sleep: async () => {} }))).rejects.toBeInstanceOf(
      GovernorViolationError,
    );
    expect(safe.held()).toEqual({ buttons: [], keys: [] });
    expect(os.performed.at(-1)).toMatchObject({ type: "pointer.up", button: "left" });
    expect(os.performed.some((e) => "x" in e && e.x === 900)).toBe(false);
  });
});
