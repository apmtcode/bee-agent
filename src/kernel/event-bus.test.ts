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

  it("assigns strictly monotonic timestamps so a reconnect cursor cannot drop colliding events", () => {
    const bus = new OperatorEventBus({ replayLimit: 10 });
    // Simulate a burst where publishers stamp the same coarse millisecond.
    bus.publish({ type: "a", ts: 100 });
    bus.publish({ type: "b", ts: 100 });
    bus.publish({ type: "c", ts: 100 });
    // A later event whose wall-clock ts went backwards must still advance.
    bus.publish({ type: "d", ts: 50 });

    const events = bus.snapshot();
    const timestamps = events.map((event) => event.ts);
    expect(timestamps).toEqual([100, 101, 102, 103]);
    // Strictly increasing → every adjacent pair is distinguishable by `ts >`.
    for (let i = 1; i < timestamps.length; i += 1) {
      expect(timestamps[i]! > timestamps[i - 1]!).toBe(true);
    }

    // A cursor at the first event's ts replays exactly the events after it.
    const cursor = timestamps[0]!;
    const missed = events.filter((event) => event.ts !== undefined && event.ts > cursor);
    expect(missed.map((event) => event.type)).toEqual(["b", "c", "d"]);
  });

  it("leaves events without a timestamp untouched", () => {
    const bus = new OperatorEventBus({ replayLimit: 4 });
    bus.publish({ type: "no-ts" });
    bus.publish({ type: "with-ts", ts: 5 });
    bus.publish({ type: "also-no-ts" });
    expect(bus.snapshot().map((event) => event.ts)).toEqual([undefined, 5, undefined]);
  });
});
