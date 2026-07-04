import { describe, expect, it } from "vitest";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import type { ReplayManifest } from "../capture/replay.js";
import {
  defaultMovementTokenizer,
  movementDatasetFromReplays,
  movementDatasetFromTrajectories,
  movementSequenceFromTrajectory,
} from "./movement-dataset.js";

describe("defaultMovementTokenizer", () => {
  it("prefers structured gesture metadata over the summary", () => {
    expect(
      defaultMovementTokenizer({
        tool: "device",
        summary: "swiped left",
        metadata: { gesture: "swipe", direction: "left" },
      }),
    ).toBe("device:swipe:left");
  });

  it("falls back to the leading verb of the summary", () => {
    expect(defaultMovementTokenizer({ tool: "browser", summary: "Clicked the login button" })).toBe(
      "browser:clicked",
    );
  });
});

describe("movementSequenceFromTrajectory", () => {
  it("orders actions by timestamp and tokenizes them", () => {
    const span = buildTrajectorySpan({
      id: "traj-1",
      sessionId: "sess-1",
      actions: [
        { kind: "action", tool: "device", summary: "tapped save", ts: 30, metadata: { gesture: "tap", target: "save" } },
        { kind: "action", tool: "device", summary: "tapped open", ts: 10, metadata: { gesture: "tap", target: "open" } },
      ],
    });

    expect(movementSequenceFromTrajectory(span).tokens).toEqual(["device:tap:open", "device:tap:save"]);
  });
});

describe("movementDatasetFromTrajectories", () => {
  it("drops trajectories with no actions", () => {
    const withActions = buildTrajectorySpan({
      id: "traj-a",
      sessionId: "sess-1",
      actions: [{ kind: "action", tool: "device", summary: "tap", ts: 1, metadata: { gesture: "tap" } }],
    });
    const empty = buildTrajectorySpan({ id: "traj-b", sessionId: "sess-1" });

    const dataset = movementDatasetFromTrajectories([withActions, empty]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]!.id).toBe("traj-a");
  });
});

describe("movementDatasetFromReplays", () => {
  it("builds one sequence per trajectory ordered by event time", () => {
    const replay: ReplayManifest = {
      version: 1,
      sessionId: "sess-1",
      trajectoryIds: ["traj-1"],
      eventCount: 3,
      events: [
        { kind: "observation", ts: 5, trajectoryId: "traj-1", source: "os", summary: "focused editor" },
        { kind: "action", ts: 20, trajectoryId: "traj-1", tool: "device", summary: "tapped save" },
        { kind: "action", ts: 10, trajectoryId: "traj-1", tool: "device", summary: "tapped open" },
      ],
    };

    const dataset = movementDatasetFromReplays([replay]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]!.id).toBe("sess-1:traj-1");
    expect(dataset.sequences[0]!.tokens).toEqual(["device:tapped", "device:tapped"]);
  });
});
