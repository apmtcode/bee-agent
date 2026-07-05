import { describe, expect, it } from "vitest";
import {
  MOVEMENT_EVENT_KINDS,
  buildMovementDataset,
  movementSequenceFromTrajectory,
  tokenizeMovementEvent,
  tokenizeSequence,
  type MovementEvent,
} from "./movement-event.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";

describe("movement-event schema", () => {
  it("tokenizes every event kind into a stable coarse token", () => {
    for (const kind of MOVEMENT_EVENT_KINDS) {
      const event: MovementEvent = { kind, ts: 0, target: "t", key: "cmd+s", direction: "down" };
      const token = tokenizeMovementEvent(event);
      expect(token.startsWith(kind === "pointer-move" ? "pointer-move" : kind.split(":")[0]!)).toBe(true);
      expect(token.length).toBeGreaterThan(0);
    }
  });

  it("produces coordinate-free tokens so the model generalizes", () => {
    const a: MovementEvent = { kind: "click", ts: 0, target: "save-button" };
    const b: MovementEvent = { kind: "click", ts: 999, target: "save-button" };
    expect(tokenizeMovementEvent(a)).toBe(tokenizeMovementEvent(b));
    expect(tokenizeMovementEvent(a)).toBe("click:save-button");
  });

  it("bridges recorded trajectory gestures into movement events", () => {
    const span = buildTrajectorySpan({
      id: "traj-1",
      sessionId: "session-1",
      actions: [
        {
          kind: "action",
          tool: "device",
          summary: "tapped save",
          ts: 1_000,
          metadata: { gesture: "tap", target: "save-button" },
        },
        {
          kind: "action",
          tool: "device",
          summary: "typed body",
          ts: 1_200,
          metadata: { gesture: "type", target: "body", valueSummary: "hello" },
        },
        {
          kind: "action",
          tool: "device",
          summary: "cmd+s",
          ts: 1_400,
          metadata: { gesture: "shortcut", target: "cmd+s" },
        },
      ],
      outcome: { status: "success", summary: "save document" },
    });

    const sequence = movementSequenceFromTrajectory(span);
    expect(sequence.id).toBe("traj-1");
    expect(sequence.intent).toBe("save document");
    expect(sequence.events.map((event) => event.kind)).toEqual(["click", "key-type", "shortcut"]);
    // ts is normalized relative to the first action.
    expect(sequence.events.map((event) => event.ts)).toEqual([0, 200, 400]);
    expect(sequence.events[1]?.value).toBe("hello");
    expect(sequence.events[2]?.key).toBe("cmd+s");
    expect(tokenizeSequence(sequence)).toEqual(["click:save-button", "key-type:body", "shortcut:cmd+s"]);
  });

  it("builds a versioned dataset wrapper", () => {
    const dataset = buildMovementDataset([{ id: "s1", events: [] }]);
    expect(dataset.version).toBe(1);
    expect(dataset.sequences).toHaveLength(1);
  });
});
