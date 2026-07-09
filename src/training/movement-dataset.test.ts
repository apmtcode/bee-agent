import { describe, expect, it } from "vitest";
import type { TrajectorySpan } from "../capture/trajectory.js";
import { buildMovementDataset } from "./movement-dataset.js";

function span(overrides: Partial<TrajectorySpan> & Pick<TrajectorySpan, "id">): TrajectorySpan {
  return {
    sessionId: "session-1",
    createdAt: "2026-01-01T00:00:00.000Z",
    captureTier: "app",
    observations: [],
    actions: [],
    ...overrides,
  };
}

describe("buildMovementDataset", () => {
  it("emits one example per action, conditioned on preceding context", () => {
    const dataset = buildMovementDataset([
      span({
        id: "traj-1",
        observations: [
          {
            kind: "observation",
            source: "device",
            summary: "Editor on Untitled",
            ts: 10,
            metadata: { appName: "Editor" },
          },
        ],
        actions: [
          {
            kind: "action",
            tool: "device",
            summary: "tapped Save",
            ts: 20,
            metadata: { gesture: "tap", target: "Save" },
          },
          {
            kind: "action",
            tool: "device",
            summary: "typed into Body",
            ts: 30,
            metadata: { gesture: "type", target: "Body" },
          },
        ],
      }),
    ]);

    expect(dataset.exampleCount).toBe(2);
    const [first, second] = dataset.examples;

    // First action sees the observation but no prior action.
    expect(first.context).toEqual({
      appName: "Editor",
      observationSource: "device",
      observationSummary: "Editor on Untitled",
      stepIndex: 0,
    });
    expect(first.action).toEqual({ tool: "device", summary: "tapped Save", gesture: "tap", target: "Save" });

    // Second action carries the first action forward as context — no label leak.
    expect(second.context.lastActionTool).toBe("device");
    expect(second.context.lastActionSummary).toBe("tapped Save");
    expect(second.context.stepIndex).toBe(1);
    expect(second.action).toEqual({ tool: "device", summary: "typed into Body", gesture: "type", target: "Body" });
  });

  it("orders interleaved events by timestamp, observation before action on ties", () => {
    const dataset = buildMovementDataset([
      span({
        id: "traj-2",
        observations: [
          { kind: "observation", source: "os", summary: "window A", ts: 5, metadata: { appName: "A" } },
          { kind: "observation", source: "os", summary: "window B", ts: 15, metadata: { appName: "B" } },
        ],
        actions: [
          { kind: "action", tool: "os", summary: "click A", ts: 5 },
          { kind: "action", tool: "os", summary: "click B", ts: 15 },
        ],
      }),
    ]);

    expect(dataset.examples[0].context.appName).toBe("A");
    expect(dataset.examples[0].action.summary).toBe("click A");
    expect(dataset.examples[1].context.appName).toBe("B");
    expect(dataset.examples[1].action.summary).toBe("click B");
  });

  it("filters to approved trajectories when requested", () => {
    const trajectories = [
      span({
        id: "approved",
        review: { status: "approved", reviewedAt: "x", reviewedBy: "y" },
        actions: [{ kind: "action", tool: "device", summary: "ok", ts: 1 }],
      }),
      span({
        id: "pending",
        actions: [{ kind: "action", tool: "device", summary: "nope", ts: 1 }],
      }),
    ];

    const all = buildMovementDataset(trajectories);
    const approvedOnly = buildMovementDataset(trajectories, { requireApprovedReview: true });

    expect(all.exampleCount).toBe(2);
    expect(approvedOnly.exampleCount).toBe(1);
    expect(approvedOnly.examples[0].trajectoryId).toBe("approved");
  });
});
