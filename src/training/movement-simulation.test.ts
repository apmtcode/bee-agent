import { describe, expect, it } from "vitest";
import { buildReplayManifest } from "../capture/replay.js";
import { buildMovementDataset } from "./movement-model.js";
import { generateSyntheticTrajectories } from "./movement-simulation.js";

describe("generateSyntheticTrajectories", () => {
  it("is deterministic for the same inputs", () => {
    const a = generateSyntheticTrajectories({ count: 8, startTs: 1000 });
    const b = generateSyntheticTrajectories({ count: 8, startTs: 1000 });
    expect(JSON.stringify(a)).toEqual(JSON.stringify(b));
  });

  it("produces time-ordered device actions with gesture metadata", () => {
    const [span] = generateSyntheticTrajectories({ count: 1 });
    expect(span?.actions.length).toBeGreaterThan(0);
    const timestamps = span?.actions.map((a) => a.ts) ?? [];
    expect(timestamps).toEqual([...timestamps].sort((x, y) => x - y));
    for (const item of span?.actions ?? []) {
      expect(item.tool).toBe("device");
      expect(item.metadata?.gesture).toBeTruthy();
    }
  });

  it("emits related-but-new variants for odd indices", () => {
    const spans = generateSyntheticTrajectories({ count: 2, templateCount: 1 });
    const verbatim = spans[0]?.actions.length ?? 0;
    const variant = spans[1]?.actions.length ?? 0;
    // The odd-indexed variant inserts one extra related step.
    expect(variant).toBe(verbatim + 1);
  });

  it("round-trips through the capture dataset + replay pipeline", () => {
    const trajectories = generateSyntheticTrajectories({ count: 5 });
    const dataset = buildMovementDataset(trajectories);
    expect(dataset.sequences).toHaveLength(5);

    const replay = buildReplayManifest({
      sessionId: "sim-session",
      transcript: [],
      trajectories,
    });
    const totalActions = trajectories.reduce((sum, t) => sum + t.actions.length, 0);
    expect(replay.eventCount).toBe(totalActions);
    expect(replay.events.every((event) => event.kind === "action")).toBe(true);
  });
});
