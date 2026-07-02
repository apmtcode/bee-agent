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

  it("assigns strictly-monotonic timestamps even within the same millisecond", () => {
    const bus = new OperatorEventBus<{ type: string; ts?: number }>({ replayLimit: 10 });
    // Three events stamped with the same wall-clock millisecond, as happens in
    // a synchronous burst of Date.now() calls.
    bus.publish({ type: "a", ts: 1000 });
    bus.publish({ type: "b", ts: 1000 });
    bus.publish({ type: "c", ts: 1000 });
    // A later event whose wall-clock time has advanced is preserved as-is.
    bus.publish({ type: "d", ts: 1005 });
    // An event with no ts still gets a strictly-greater timestamp.
    bus.publish({ type: "e" });

    const stamps = bus.snapshot().map((event) => event.ts);
    expect(stamps).toEqual([1000, 1001, 1002, 1005, 1006]);
    for (let i = 1; i < stamps.length; i += 1) {
      expect(stamps[i] as number).toBeGreaterThan(stamps[i - 1] as number);
    }
  });
});
