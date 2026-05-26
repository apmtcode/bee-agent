import { describe, expect, it } from "vitest";
import { buildReplayManifest } from "./replay.js";
import { buildTrajectorySpan } from "./trajectory.js";
import type { TranscriptRecord } from "../harness/transcript-store.js";

describe("buildReplayManifest", () => {
  it("combines transcript and trajectory events into a deterministic timeline", () => {
    const transcript: TranscriptRecord[] = [
      { type: "session", version: 1, id: "sess-1", timestamp: "2026-01-01T00:00:00.000Z" },
      {
        id: "msg-1",
        message: { role: "user", content: "check deploy", timestamp: 10 },
      },
      {
        id: "msg-2",
        message: { role: "assistant", content: "deploy checked", timestamp: 30 },
      },
    ];
    const trajectory = buildTrajectorySpan({
      id: "traj-1",
      sessionId: "sess-1",
      observations: [{ kind: "observation", source: "browser", summary: "opened deploy page", ts: 20 }],
      actions: [{ kind: "action", tool: "browser", summary: "clicked deploy", ts: 40 }],
      outcome: { status: "success", summary: "deploy queued", reward: 1 },
    });

    const replay = buildReplayManifest({
      sessionId: "sess-1",
      transcript,
      trajectories: [trajectory],
    });

    expect(replay.trajectoryIds).toEqual(["traj-1"]);
    expect(replay.eventCount).toBe(4);
    expect(replay.events).toEqual([
      { kind: "transcript", ts: 10, messageId: "msg-1", role: "user", content: "check deploy" },
      { kind: "observation", ts: 20, trajectoryId: "traj-1", source: "browser", summary: "opened deploy page" },
      { kind: "transcript", ts: 30, messageId: "msg-2", role: "assistant", content: "deploy checked" },
      { kind: "action", ts: 40, trajectoryId: "traj-1", tool: "browser", summary: "clicked deploy" },
    ]);
  });
});
