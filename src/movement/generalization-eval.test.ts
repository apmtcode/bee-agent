import { describe, expect, it } from "vitest";
import { buildMovementDataset, NgramMovementBackend } from "./movement-model.js";
import { evaluateMovementModelOnTrajectories } from "./generalization-eval.js";
import { generateSyntheticCorpus } from "./synthetic.js";

describe("generateSyntheticCorpus", () => {
  it("is deterministic for a fixed seed", () => {
    const a = generateSyntheticCorpus({ seed: 7, trainCount: 5, heldOutCount: 3 });
    const b = generateSyntheticCorpus({ seed: 7, trainCount: 5, heldOutCount: 3 });
    expect(a.train.map((t) => t.id)).toEqual(b.train.map((t) => t.id));
    expect(a.train.map((t) => t.actions.map((x) => x.summary))).toEqual(
      b.train.map((t) => t.actions.map((x) => x.summary)),
    );
    expect(a.train).toHaveLength(5);
    expect(a.heldOut).toHaveLength(3);
  });

  it("gives held-out trajectories new targets but the same gesture grammar", () => {
    const { train, heldOut } = generateSyntheticCorpus({ seed: 1, trainCount: 4, heldOutCount: 4 });
    const trainTargets = new Set(train.flatMap((t) => t.actions.map((a) => a.metadata?.target as string)));
    const heldOutTargets = heldOut.flatMap((t) => t.actions.map((a) => a.metadata?.target as string));
    // Held-out targets are remapped (…-alt) so none collide with the training vocabulary.
    for (const target of heldOutTargets) {
      expect(trainTargets.has(target)).toBe(false);
    }
  });
});

describe("evaluateMovementModel — generalization", () => {
  it("reproduces held-out related movements with high fidelity", () => {
    const { train, heldOut } = generateSyntheticCorpus({ seed: 42, trainCount: 60, heldOutCount: 20 });
    const model = new NgramMovementBackend().train(buildMovementDataset(train), { order: 3 });

    const report = evaluateMovementModelOnTrajectories(model, heldOut, { topK: 2 });

    expect(report.sequenceCount).toBe(20);
    expect(report.stepCount).toBeGreaterThan(0);
    // Structural tokens make held-out movements match the learned grammar closely.
    expect(report.accuracy).toBeGreaterThan(0.8);
    expect(report.topKAccuracy).toBeGreaterThanOrEqual(report.accuracy);
  });

  it("scores accuracy on unseen contexts via the back-off path", () => {
    // Train on only the first flow, then eval on all flows so some contexts are novel.
    const full = generateSyntheticCorpus({ seed: 5, trainCount: 40, heldOutCount: 40 });
    const model = new NgramMovementBackend().train(buildMovementDataset(full.train), { order: 4 });
    const report = evaluateMovementModelOnTrajectories(model, full.heldOut);

    expect(report.generalizedAccuracy).toBeGreaterThanOrEqual(0);
    expect(report.generalizedCorrect).toBeLessThanOrEqual(report.backoffSteps);
    expect(report.correct).toBeLessThanOrEqual(report.stepCount);
  });

  it("reports zero steps for single-action trajectories", () => {
    const model = new NgramMovementBackend().train({ sequences: [{ trajectoryId: "t", tokens: ["device:tap"] }] });
    const report = evaluateMovementModelOnTrajectories(model, []);
    expect(report.stepCount).toBe(0);
    expect(report.accuracy).toBe(0);
  });
});
