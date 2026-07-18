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

  it("assigns strictly-monotonic timestamps so same-ms events never collide", () => {
    const bus = new OperatorEventBus({ replayLimit: 10 });
    // Three events published with the same wall-clock millisecond.
    bus.publish({ type: "a", ts: 100 });
    bus.publish({ type: "b", ts: 100 });
    bus.publish({ type: "c", ts: 100 });
    // A later event that happens to carry a smaller clock reading must still advance.
    bus.publish({ type: "d", ts: 90 });

    const tsValues = bus.snapshot().map((event) => event.ts);
    expect(tsValues).toEqual([100, 101, 102, 103]);
    // afterTs filtering (used on reconnect) must not drop any distinct event.
    const afterTs = 100;
    expect(bus.snapshot().filter((event) => (event.ts ?? 0) > afterTs).map((e) => e.type)).toEqual([
      "b",
      "c",
      "d",
    ]);
  });

  it("leaves events without a timestamp untouched", () => {
    const bus = new OperatorEventBus({ replayLimit: 10 });
    bus.publish({ type: "a" });
    bus.publish({ type: "b", ts: 5 });
    bus.publish({ type: "c" });
    expect(bus.snapshot().map((event) => event.ts)).toEqual([undefined, 5, undefined]);
  });
});
