import { describe, expect, it } from "vitest";
import { buildTrajectorySpan, type TrajectoryAction } from "../../capture/trajectory.js";
import {
  buildMovementDataset,
  movementSequenceFromReplayEvents,
  movementStepFromAction,
  movementTokenFromAction,
  movementVocabulary,
} from "./tokenizer.js";

function action(overrides: Partial<TrajectoryAction>): TrajectoryAction {
  return {
    kind: "action",
    tool: "device",
    summary: "did a thing",
    ts: 1,
    ...overrides,
  };
}

describe("movement tokenizer", () => {
  it("derives structural tokens from gesture + direction, ignoring free text", () => {
    expect(
      movementTokenFromAction(action({ metadata: { gesture: "tap", target: "Send button" } })),
    ).toBe("device.tap");
    expect(
      movementTokenFromAction(action({ metadata: { gesture: "scroll", direction: "down" } })),
    ).toBe("device.scroll.down");
  });

  it("collapses distinct-but-related movements onto the same token", () => {
    const a = movementTokenFromAction(action({ metadata: { gesture: "tap", target: "OK" } }));
    const b = movementTokenFromAction(action({ metadata: { gesture: "tap", target: "Cancel" } }));
    expect(a).toBe(b);
  });

  it("falls back to os event kind when no gesture is present", () => {
    expect(
      movementTokenFromAction(action({ tool: "os", metadata: { event: "focus-changed" } })),
    ).toBe("os.focus-changed");
  });

  it("preserves replayable structure in the step", () => {
    const step = movementStepFromAction(
      action({ metadata: { gesture: "swipe", direction: "left", target: "card" }, ts: 42 }),
    );
    expect(step).toMatchObject({ token: "device.swipe.left", tool: "device", ts: 42, direction: "left", target: "card" });
  });

  it("builds a dataset ordered by timestamp and drops empty trajectories", () => {
    const withActions = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [
        action({ ts: 20, metadata: { gesture: "type" } }),
        action({ ts: 10, metadata: { gesture: "tap" } }),
      ],
    });
    const empty = buildTrajectorySpan({ id: "t2", sessionId: "s1" });
    const dataset = buildMovementDataset([withActions, empty]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]!.steps.map((s) => s.token)).toEqual(["device.tap", "device.type"]);
    expect(movementVocabulary(dataset)).toEqual(["device.tap", "device.type"]);
  });

  it("extracts only action events from a replay timeline", () => {
    const sequence = movementSequenceFromReplayEvents("r1", [
      { kind: "transcript", ts: 1, messageId: "m", role: "user", content: "hi" },
      { kind: "observation", ts: 2, trajectoryId: "t", source: "os", summary: "focused" },
      { kind: "action", ts: 4, trajectoryId: "t", tool: "device", summary: "tapped" },
      { kind: "action", ts: 3, trajectoryId: "t", tool: "browser", summary: "clicked" },
    ]);
    expect(sequence.steps.map((s) => s.token)).toEqual(["browser", "device"]);
  });
});
