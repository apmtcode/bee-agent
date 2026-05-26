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
});
