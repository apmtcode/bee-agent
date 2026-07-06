import { describe, expect, it } from "vitest";
import type { TrajectorySpan } from "../capture/trajectory.js";
import { buildMovementDataset } from "./movement-dataset.js";
import { syntheticDeviceTrajectory } from "./movement-test-utils.js";

describe("buildMovementDataset", () => {
  it("derives context and ordered steps from a recorded device trajectory", () => {
    const trajectory = syntheticDeviceTrajectory({
      id: "traj-1",
      sessionId: "session-1",
      appId: "mail",
      platform: "macos",
      screenTitle: "Inbox",
      goal: "compose a message",
      gestures: [
        { kind: "tap", target: "Compose" },
        { kind: "type", target: "To", valueSummary: "alice@example.com" },
      ],
      approved: true,
    });

    const dataset = buildMovementDataset([trajectory]);

    expect(dataset.version).toBe(1);
    expect(dataset.examples).toHaveLength(1);
    const [example] = dataset.examples;
    expect(example.context.appId).toBe("mail");
    expect(example.context.platform).toBe("macos");
    expect(example.context.screenTitle).toBe("Inbox");
    expect(example.steps).toHaveLength(2);
    expect(example.steps[0]).toMatchObject({ tool: "device", gesture: "tap", target: "Compose" });
    expect(example.steps[1]).toMatchObject({ gesture: "type", target: "To", valueSummary: "alice@example.com" });
  });

  it("excludes unreviewed trajectories by default and includes them when reviewedOnly=false", () => {
    const unreviewed = syntheticDeviceTrajectory({
      id: "traj-2",
      sessionId: "session-2",
      appId: "mail",
      platform: "macos",
      gestures: [{ kind: "tap", target: "Compose" }],
      approved: false,
    });

    expect(buildMovementDataset([unreviewed]).examples).toHaveLength(0);
    expect(buildMovementDataset([unreviewed], { reviewedOnly: false }).examples).toHaveLength(1);
  });

  it("skips trajectories with no actions", () => {
    const observationOnly: TrajectorySpan = {
      id: "traj-3",
      sessionId: "session-3",
      createdAt: "2026-01-01T00:00:00.000Z",
      captureTier: "app",
      observations: [{ kind: "observation", source: "device", summary: "idle", ts: 1 }],
      actions: [],
      review: { status: "approved", reviewedAt: "2026-01-01T00:00:00.000Z", reviewedBy: "op" },
    };

    expect(buildMovementDataset([observationOnly]).examples).toHaveLength(0);
  });
});
