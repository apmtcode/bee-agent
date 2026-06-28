import { describe, expect, it } from "vitest";
import { generateSyntheticTrajectories } from "./synthetic.js";
import { buildMovementDataset } from "./movement-dataset.js";

describe("generateSyntheticTrajectories", () => {
  it("is deterministic for a given seed", () => {
    const a = generateSyntheticTrajectories({ seed: 42, trajectoryCount: 5 });
    const b = generateSyntheticTrajectories({ seed: 42, trajectoryCount: 5 });
    expect(JSON.stringify(a)).toEqual(JSON.stringify(b));
  });

  it("produces different streams for different seeds", () => {
    const a = generateSyntheticTrajectories({ seed: 1, trajectoryCount: 8 });
    const b = generateSyntheticTrajectories({ seed: 2, trajectoryCount: 8 });
    expect(JSON.stringify(a)).not.toEqual(JSON.stringify(b));
  });

  it("emits approved spans whose actions follow observations in time", () => {
    const [span] = generateSyntheticTrajectories({ seed: 7, trajectoryCount: 1, actionsPerTrajectory: 2 });
    expect(span.review?.status).toBe("approved");
    expect(span.observations).toHaveLength(2);
    expect(span.actions).toHaveLength(2);
    expect(span.actions[0].ts).toBeGreaterThan(span.observations[0].ts);
  });

  it("round-trips into a non-empty movement dataset", () => {
    const trajectories = generateSyntheticTrajectories({ seed: 99, trajectoryCount: 6, actionsPerTrajectory: 3 });
    const dataset = buildMovementDataset(trajectories);
    expect(dataset.examples.length).toBe(18);
    expect(dataset.apps.length).toBeGreaterThan(0);
    expect(dataset.targets.length).toBeGreaterThan(0);
  });
});
