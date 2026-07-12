import { describe, expect, it } from "vitest";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import { buildReplayManifest } from "../capture/replay.js";
import {
  buildMovementDataset,
  buildMovementDatasetFromReplays,
  movementVerb,
  tokenFromReplayEvent,
} from "./movement-dataset.js";

describe("movementVerb", () => {
  it("normalizes the first word of a summary", () => {
    expect(movementVerb("Clicked the deploy button")).toBe("clicked");
    expect(movementVerb("  press-Enter now")).toBe("pressenter");
    expect(movementVerb("")).toBe("act");
    expect(movementVerb("!!!")).toBe("act");
  });
});

describe("buildMovementDataset", () => {
  it("orders tokens by timestamp and derives canonical symbols", () => {
    const trajectory = buildTrajectorySpan({
      id: "traj-1",
      sessionId: "sess-1",
      observations: [{ kind: "observation", source: "Screen", summary: "page loaded", ts: 5 }],
      actions: [
        { kind: "action", tool: "Mouse", summary: "click deploy", ts: 30 },
        { kind: "action", tool: "Keyboard", summary: "type name", ts: 10 },
      ],
    });

    const dataset = buildMovementDataset([trajectory]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0].tokens.map((token) => token.symbol)).toEqual([
      "observation:screen:page",
      "action:keyboard:type",
      "action:mouse:click",
    ]);
    expect(dataset.vocabulary).toEqual([
      "action:keyboard:type",
      "action:mouse:click",
      "observation:screen:page",
    ]);
    expect(dataset.representatives["action:mouse:click"].summary).toBe("click deploy");
  });

  it("drops trajectories with no movements", () => {
    const empty = buildTrajectorySpan({ id: "empty", sessionId: "s" });
    expect(buildMovementDataset([empty]).sequences).toHaveLength(0);
  });
});

describe("buildMovementDatasetFromReplays", () => {
  it("reconstructs sequences from replay manifests and skips transcript events", () => {
    const trajectory = buildTrajectorySpan({
      id: "traj-9",
      sessionId: "sess-9",
      observations: [{ kind: "observation", source: "browser", summary: "opened page", ts: 20 }],
      actions: [{ kind: "action", tool: "browser", summary: "clicked link", ts: 40 }],
    });
    const replay = buildReplayManifest({
      sessionId: "sess-9",
      transcript: [{ id: "m1", message: { role: "user", content: "go", timestamp: 10 } }],
      trajectories: [trajectory],
    });

    const dataset = buildMovementDatasetFromReplays([replay]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0].trajectoryId).toBe("traj-9");
    expect(dataset.sequences[0].tokens.map((token) => token.symbol)).toEqual([
      "observation:browser:opened",
      "action:browser:clicked",
    ]);
  });

  it("ignores transcript events at the token level", () => {
    expect(
      tokenFromReplayEvent({ kind: "transcript", ts: 1, messageId: "m", role: "user", content: "hi" }),
    ).toBeUndefined();
  });
});
