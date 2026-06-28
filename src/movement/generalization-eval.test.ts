import { describe, expect, it } from "vitest";
import { evaluateMovementGeneralization, splitMovementDataset } from "./generalization-eval.js";
import { buildMovementDataset } from "./movement-dataset.js";
import { MockMovementModelBackend } from "./movement-backend.js";
import { generateSyntheticTrajectories } from "./synthetic.js";

describe("splitMovementDataset", () => {
  it("holds out every Nth example deterministically", () => {
    const trajectories = generateSyntheticTrajectories({ seed: 3, trajectoryCount: 4, actionsPerTrajectory: 3 });
    const dataset = buildMovementDataset(trajectories);
    const { train, holdout } = splitMovementDataset(dataset, 3);
    expect(train.examples.length + holdout.length).toBe(dataset.examples.length);
    expect(holdout.length).toBe(Math.floor(dataset.examples.length / 3));
  });
});

describe("evaluateMovementGeneralization", () => {
  it("reproduces a meaningful fraction of held-out movements", async () => {
    const trajectories = generateSyntheticTrajectories({ seed: 21, trajectoryCount: 30, actionsPerTrajectory: 4 });
    const report = await evaluateMovementGeneralization({
      backend: new MockMovementModelBackend(),
      trajectories,
      holdoutEvery: 3,
      trainedAt: "2026-01-01T00:00:00.000Z",
    });

    expect(report.holdoutCount).toBeGreaterThan(0);
    expect(report.trainCount).toBeGreaterThan(report.holdoutCount);
    // The synthetic grammar is learnable: the model should at least recover the
    // tool/gesture for most held-out movements.
    expect(report.toolAccuracy).toBeGreaterThanOrEqual(0.9);
    expect(report.exactAccuracy).toBeGreaterThan(0);
    expect(report.cases).toHaveLength(report.holdoutCount);
  });

  it("is reproducible across runs with a fixed trainedAt", async () => {
    const trajectories = generateSyntheticTrajectories({ seed: 5, trajectoryCount: 12 });
    const run = () =>
      evaluateMovementGeneralization({
        backend: new MockMovementModelBackend(),
        trajectories,
        trainedAt: "2026-01-01T00:00:00.000Z",
      });
    const a = await run();
    const b = await run();
    expect(a.exactAccuracy).toBe(b.exactAccuracy);
    expect(a.toolAccuracy).toBe(b.toolAccuracy);
    expect(a.generalizationRate).toBe(b.generalizationRate);
  });
});
