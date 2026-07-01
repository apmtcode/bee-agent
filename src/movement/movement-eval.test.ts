import { describe, expect, it } from "vitest";
import { evaluateMovementModel } from "./movement-eval.js";
import { NGramMovementModel } from "./movement-model.js";
import { generateSyntheticMovementDataset, type SyntheticTaskTemplate } from "./synthetic.js";

describe("evaluateMovementModel", () => {
  it("scores a memorized deterministic flow perfectly", () => {
    const templates: SyntheticTaskTemplate[] = [
      {
        app: "terminal",
        steps: [
          { tool: "os", summary: "focused terminal" },
          { tool: "device", summary: "typed command" },
          { tool: "device", summary: "triggered enter" },
        ],
      },
    ];
    const dataset = generateSyntheticMovementDataset({ seed: 5, count: 6, templates });
    const model = NGramMovementModel.train(dataset.sequences);
    const metrics = evaluateMovementModel(model, dataset.sequences);

    expect(metrics.sequenceCount).toBe(6);
    expect(metrics.nextTokenAccuracy).toBe(1);
    expect(metrics.exactReplayRate).toBe(1);
    expect(metrics.meanReplayOverlap).toBe(1);
  });

  it("keeps all metrics within [0,1] on a branching dataset", () => {
    const dataset = generateSyntheticMovementDataset({ seed: 11, count: 30 });
    const model = NGramMovementModel.train(dataset.sequences);
    const metrics = evaluateMovementModel(model, dataset.sequences, { topK: 2 });

    for (const value of [
      metrics.nextTokenAccuracy,
      metrics.topKAccuracy,
      metrics.exactReplayRate,
      metrics.meanReplayOverlap,
    ]) {
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThanOrEqual(1);
    }
    // top-K accuracy can never be worse than top-1.
    expect(metrics.topKAccuracy).toBeGreaterThanOrEqual(metrics.nextTokenAccuracy);
    // The learned canonical flows should be largely predictable.
    expect(metrics.nextTokenAccuracy).toBeGreaterThan(0.6);
    expect(metrics.predictionCount).toBeGreaterThan(0);
  });

  it("generalizes to held-out trajectories drawn from the same grammars", () => {
    const train = generateSyntheticMovementDataset({ seed: 100, count: 40 });
    const heldOut = generateSyntheticMovementDataset({ seed: 999, count: 20 });
    const model = NGramMovementModel.train(train.sequences);
    const metrics = evaluateMovementModel(model, heldOut.sequences);

    // The model never saw these exact trajectories but shares the grammars,
    // so next-action prediction should still be strong.
    expect(metrics.nextTokenAccuracy).toBeGreaterThan(0.6);
    expect(metrics.meanReplayOverlap).toBeGreaterThan(0.6);
  });

  it("returns zeroed metrics for an empty evaluation set", () => {
    const model = NGramMovementModel.train([["a", "b"]]);
    const metrics = evaluateMovementModel(model, []);
    expect(metrics.sequenceCount).toBe(0);
    expect(metrics.predictionCount).toBe(0);
    expect(metrics.nextTokenAccuracy).toBe(0);
    expect(metrics.exactReplayRate).toBe(0);
  });
});
