import { describe, expect, it } from "vitest";
import { buildTrajectorySpan, type TrajectorySpan } from "../../capture/trajectory.js";
import type { ReplayManifest } from "../../capture/replay.js";
import {
  buildMovementDataset,
  buildMovementDatasetFromReplays,
  trajectoryToMovementSequence,
} from "./dataset.js";

describe("buildMovementDataset", () => {
  it("merges observations and actions into a time-ordered sequence", () => {
    const trajectory = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      observations: [{ kind: "observation", source: "window", summary: "focus", ts: 100 }],
      actions: [
        { kind: "action", tool: "mouse.click", summary: "click", ts: 300 },
        { kind: "action", tool: "keyboard.type", summary: "type", ts: 200 },
      ],
      outcome: { status: "success", summary: "done", reward: 1 },
    });
    const sequence = trajectoryToMovementSequence(trajectory);
    expect(sequence.events.map((event) => event.ts)).toEqual([100, 200, 300]);
    expect(sequence.outcome).toBe("success");
    expect(sequence.reward).toBe(1);
  });

  it("prefers reviewed/redacted views when present", () => {
    const trajectory: TrajectorySpan = {
      ...buildTrajectorySpan({
        id: "t2",
        sessionId: "s1",
        observations: [{ kind: "observation", source: "raw", summary: "secret", ts: 10 }],
        actions: [{ kind: "action", tool: "raw.tool", summary: "secret", ts: 20 }],
      }),
      review: {
        status: "approved",
        reviewedAt: "2026-01-01T00:00:00.000Z",
        reviewedBy: "operator",
        redactedObservations: [{ ts: 10, source: "window", summary: "focus" }],
        redactedActions: [{ ts: 20, tool: "mouse.click", summary: "click" }],
      },
    };
    const sequence = trajectoryToMovementSequence(trajectory);
    expect(sequence.events.map((event) => event.summary)).toEqual(["focus", "click"]);
  });

  it("skips empty trajectories", () => {
    const empty = buildTrajectorySpan({ id: "t3", sessionId: "s1" });
    const dataset = buildMovementDataset([empty]);
    expect(dataset.sequences).toHaveLength(0);
  });
});

describe("buildMovementDatasetFromReplays", () => {
  it("groups replay events by trajectory and drops transcript events", () => {
    const replay: ReplayManifest = {
      version: 1,
      sessionId: "s1",
      trajectoryIds: ["t1"],
      eventCount: 4,
      events: [
        { kind: "transcript", ts: 50, messageId: "m1", role: "user", content: "hi" },
        { kind: "observation", ts: 100, trajectoryId: "t1", source: "window", summary: "focus" },
        { kind: "action", ts: 200, trajectoryId: "t1", tool: "mouse.click", summary: "click" },
        { kind: "action", ts: 150, trajectoryId: "t1", tool: "keyboard.type", summary: "type" },
      ],
    };
    const dataset = buildMovementDatasetFromReplays([replay]);
    expect(dataset.sequences).toHaveLength(1);
    const events = dataset.sequences[0].events;
    expect(events).toHaveLength(3);
    expect(events.map((event) => event.ts)).toEqual([100, 150, 200]);
    expect(events.every((event) => event.kind !== ("transcript" as unknown))).toBe(true);
  });
});
