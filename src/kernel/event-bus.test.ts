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

  it("forces strictly-increasing timestamps so same-ms events stay addressable by a reconnect cursor", () => {
    const bus = new OperatorEventBus({ replayLimit: 10 });
    // Three events stamped with an identical (colliding) millisecond timestamp,
    // as Date.now() would produce under load.
    bus.publish({ type: "a", ts: 100 });
    bus.publish({ type: "b", ts: 100 });
    bus.publish({ type: "c", ts: 100 });

    const events = bus.snapshot();
    expect(events.map((event) => event.ts)).toEqual([100, 101, 102]);

    // An exclusive reconnect cursor at the first event's ts still replays b and c.
    const afterTs = events[0]!.ts as number;
    const missed = events.filter((event) => (event.ts as number) > afterTs);
    expect(missed.map((event) => event.type)).toEqual(["b", "c"]);
  });

  it("preserves already-increasing timestamps unchanged", () => {
    const bus = new OperatorEventBus({ replayLimit: 10 });
    bus.publish({ type: "a", ts: 5 });
    bus.publish({ type: "b", ts: 9 });
    bus.publish({ type: "c", ts: 20 });
    expect(bus.snapshot().map((event) => event.ts)).toEqual([5, 9, 20]);
  });
});
