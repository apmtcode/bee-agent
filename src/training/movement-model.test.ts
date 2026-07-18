import { describe, expect, it } from "vitest";
import {
  buildMovementDataset,
  countDatasetEvents,
  movementEventFromAction,
  movementFromToken,
  movementToken,
  MOVEMENT_END_TOKEN,
  MOVEMENT_START_TOKEN,
} from "./movement-model.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

function span(id: string, actions: TrajectorySpan["actions"]): TrajectorySpan {
  return {
    id,
    sessionId: "s1",
    createdAt: "2026-07-18T00:00:00.000Z",
    captureTier: "full",
    observations: [],
    actions,
  };
}

describe("movement token round-trip", () => {
  it("collapses value into a stable symbol and reconstructs kind/target/direction", () => {
    const token = movementToken({ ts: 1, kind: "scroll", target: "page", direction: "down", value: "x" });
    expect(token).toBe("scroll|page|down");
    const event = movementFromToken(token, 42);
    expect(event).toEqual({ ts: 42, kind: "scroll", target: "page", direction: "down" });
  });

  it("treats identical movements with different typed values as one token", () => {
    const a = movementToken({ ts: 1, kind: "type", target: "field", value: "alice" });
    const b = movementToken({ ts: 2, kind: "type", target: "field", value: "bob" });
    expect(a).toBe(b);
  });

  it("returns undefined for sentinel and malformed tokens", () => {
    expect(movementFromToken(MOVEMENT_START_TOKEN, 0)).toBeUndefined();
    expect(movementFromToken(MOVEMENT_END_TOKEN, 0)).toBeUndefined();
    expect(movementFromToken("not-a-kind|x|*", 0)).toBeUndefined();
  });
});

describe("movementEventFromAction", () => {
  it("reads gesture metadata from a device action", () => {
    const event = movementEventFromAction({
      kind: "action",
      tool: "device",
      summary: "swiped up",
      ts: 5,
      metadata: { gesture: "swipe", target: "list", direction: "up", valueSummary: "fast" },
    });
    expect(event).toEqual({ ts: 5, kind: "swipe", target: "list", direction: "up", value: "fast" });
  });

  it("maps synonyms and falls back to the tool name", () => {
    expect(movementEventFromAction({ kind: "action", tool: "key", summary: "", ts: 1 })?.kind).toBe("keypress");
    expect(movementEventFromAction({ kind: "action", tool: "click", summary: "", ts: 1 })?.kind).toBe("click");
  });

  it("drops actions with no movement mapping", () => {
    expect(movementEventFromAction({ kind: "action", tool: "Bash", summary: "ran ls", ts: 1 })).toBeUndefined();
  });
});

describe("buildMovementDataset", () => {
  it("produces one sorted sequence per trajectory with movements", () => {
    const dataset = buildMovementDataset([
      span("t1", [
        { kind: "action", tool: "device", summary: "b", ts: 20, metadata: { gesture: "click", target: "b" } },
        { kind: "action", tool: "device", summary: "a", ts: 10, metadata: { gesture: "focus", target: "a" } },
      ]),
      span("empty", [{ kind: "action", tool: "Bash", summary: "noop", ts: 1 }]),
    ]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]!.id).toBe("t1");
    expect(dataset.sequences[0]!.events.map((e) => e.ts)).toEqual([10, 20]);
    expect(countDatasetEvents(dataset)).toBe(2);
  });
});
