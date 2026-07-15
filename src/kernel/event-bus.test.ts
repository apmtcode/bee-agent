import { describe, expect, it } from "vitest";
import { OperatorEventBus } from "./event-bus.js";

describe("OperatorEventBus", () => {
  it("replays recent events to new streams", async () => {
    const bus = new OperatorEventBus({ replayLimit: 2 });
    bus.publish({ type: "one", ts: 1 });
    bus.publish({ type: "two", ts: 2 });
    bus.publish({ type: "three", ts: 3 });

    const stream = bus.stream(undefined, { replay: true })[Symbol.asyncIterator]();
    await expect(stream.next()).resolves.toMatchObject({ done: false, value: { type: "two" } });
    await expect(stream.next()).resolves.toMatchObject({ done: false, value: { type: "three" } });
    await stream.return?.();
  });

  it("assigns strictly monotonic timestamps so no two events collide", () => {
    const bus = new OperatorEventBus({ replayLimit: 10 });
    // Simulate three events published within the same wall-clock millisecond.
    bus.publish({ type: "a", ts: 100 });
    bus.publish({ type: "b", ts: 100 });
    bus.publish({ type: "c", ts: 100 });
    // A later event whose clock went backwards must still advance.
    bus.publish({ type: "d", ts: 99 });

    const stamps = bus.snapshot().map((event) => event.ts);
    expect(stamps).toEqual([100, 101, 102, 103]);
    // Uniqueness + strict monotonicity: a `ts > cursor` replay cursor can never
    // drop a later event by colliding on its millisecond.
    expect(new Set(stamps).size).toBe(stamps.length);
    for (let i = 1; i < stamps.length; i += 1) {
      expect(stamps[i]!).toBeGreaterThan(stamps[i - 1]!);
    }
  });

  it("leaves events without a timestamp untouched", () => {
    const bus = new OperatorEventBus({ replayLimit: 5 });
    bus.publish({ type: "no-ts" });
    bus.publish({ type: "with-ts", ts: 5 });
    expect(bus.snapshot().map((event) => event.ts)).toEqual([undefined, 5]);
  });
});
