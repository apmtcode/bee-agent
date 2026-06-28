import { describe, expect, it } from "vitest";
import { MarkovMovementBackend, tokenizeTrajectory } from "./movement-model.js";
import {
  evaluateMovementModel,
  evaluateMovementModelOnTrajectories,
} from "./movement-eval.js";
import { generateSyntheticTrajectories } from "./synthetic-movements.js";

describe("evaluateMovementModel", () => {
  it("scores perfect replay on an unambiguous memorized dataset", async () => {
    // A single sequence of distinct tokens: every context (including the empty
    // start context) maps to exactly one continuation, so replay is exact.
    const dataset = [
      [
        { tool: "device", action: "tap", target: "a" },
        { tool: "device", action: "type", target: "b" },
        { tool: "device", action: "tap", target: "c" },
        { tool: "device", action: "shortcut", target: "save" },
      ],
    ];
    const model = await new MarkovMovementBackend().train(dataset, { order: 3 });

    const report = evaluateMovementModel(model, dataset, { topK: 3 });
    expect(report.sequenceCount).toBe(dataset.length);
    expect(report.accuracy).toBe(1);
    expect(report.meanReplayFidelity).toBe(1);
  });

  it("achieves high next-token accuracy on memorized branching data", async () => {
    const dataset = generateSyntheticTrajectories({ seed: 11, count: 12 }).map(tokenizeTrajectory);
    const model = await new MarkovMovementBackend().train(dataset, { order: 3 });

    const report = evaluateMovementModel(model, dataset, { topK: 3 });
    expect(report.predictionCount).toBeGreaterThan(0);
    // Inherent first-token branching (multiple templates/targets) caps top-1,
    // but conditioned on true history most steps are correct, and top-3 nearly all.
    expect(report.accuracy).toBeGreaterThan(0.7);
    expect(report.topKAccuracy).toBeGreaterThan(0.9);
    expect(report.topKAccuracy).toBeGreaterThanOrEqual(report.accuracy);
  });

  it("generalizes to held-out but related trajectories better than chance", async () => {
    // Train and test on the same skill templates but different seeds, so the
    // held-out trajectories are related-but-novel instances.
    const train = generateSyntheticTrajectories({ seed: 100, count: 40 });
    const heldOut = generateSyntheticTrajectories({ seed: 999, count: 20, sessionPrefix: "holdout" });

    const model = await new MarkovMovementBackend().train(train.map(tokenizeTrajectory), { order: 3 });
    const report = evaluateMovementModelOnTrajectories(model, heldOut, { topK: 3 });

    // The templates have ~2-way target branching, so blind chance is well under
    // 0.5; a model that has learned the skill structure should clear that bar.
    expect(report.topKAccuracy).toBeGreaterThan(0.6);
    expect(report.accuracy).toBeGreaterThan(0.4);
  });

  it("returns zeroed metrics for an empty held-out set", async () => {
    const model = await new MarkovMovementBackend().train([], {});
    const report = evaluateMovementModel(model, []);
    expect(report).toMatchObject({
      sequenceCount: 0,
      predictionCount: 0,
      accuracy: 0,
      meanReplayFidelity: 0,
      generalizationRate: 0,
    });
  });
});
