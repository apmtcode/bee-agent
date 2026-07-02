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

  it("assigns strictly increasing timestamps so reconnect cursors never drop tied events", () => {
    const bus = new OperatorEventBus({ replayLimit: 10 });
    // Simulate a burst published within the same millisecond (Date.now() tie),
    // plus a clock that briefly goes backwards.
    bus.publish({ type: "a", ts: 100 });
    bus.publish({ type: "b", ts: 100 });
    bus.publish({ type: "c", ts: 100 });
    bus.publish({ type: "d", ts: 99 });
    bus.publish({ type: "e", ts: 105 });

    const events = bus.snapshot();
    const timestamps = events.map((event) => event.ts as number);
    expect(timestamps).toEqual([100, 101, 102, 103, 105]);
    // Strictly increasing — the invariant the `ts > afterTs` reconnect cursor relies on.
    for (let index = 1; index < timestamps.length; index += 1) {
      expect(timestamps[index]).toBeGreaterThan(timestamps[index - 1] as number);
    }

    // A consumer resuming after "b" (ts 101) must still receive c, d, e.
    const resumed = events.filter((event) => (event.ts as number) > 101);
    expect(resumed.map((event) => event.type)).toEqual(["c", "d", "e"]);
  });

  it("leaves events without a timestamp untouched", () => {
    const bus = new OperatorEventBus({ replayLimit: 5 });
    bus.publish({ type: "no-ts" });
    bus.publish({ type: "with-ts", ts: 5 });
    expect(bus.snapshot().map((event) => event.ts)).toEqual([undefined, 5]);
  });
});
