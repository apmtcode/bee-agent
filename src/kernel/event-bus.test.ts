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

  it("bumps colliding timestamps so replay ts is strictly monotonic", () => {
    const bus = new OperatorEventBus({ replayLimit: 10 });
    // Three events published within the same millisecond (as Date.now() would).
    bus.publish({ type: "a", ts: 100 });
    bus.publish({ type: "b", ts: 100 });
    bus.publish({ type: "c", ts: 100 });
    // A later real timestamp must still win.
    bus.publish({ type: "d", ts: 250 });

    const stamped = bus.snapshot().map((event) => event.ts);
    expect(stamped).toEqual([100, 101, 102, 250]);
    for (let i = 1; i < stamped.length; i += 1) {
      expect(stamped[i]!).toBeGreaterThan(stamped[i - 1]!);
    }
  });

  it("does not stamp events that omit ts", () => {
    const bus = new OperatorEventBus({ replayLimit: 4 });
    bus.publish({ type: "no-ts" });
    bus.publish({ type: "with-ts", ts: 5 });
    expect(bus.snapshot().map((event) => event.ts)).toEqual([undefined, 5]);
  });
});
