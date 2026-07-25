import { describe, expect, it } from "vitest";
import { DeterministicNgramBackend } from "./movement-model.js";
import {
  DEFAULT_MOVEMENT_TEMPLATES,
  evaluateMovementModel,
  generateSyntheticCorpus,
  splitCorpus,
} from "./movement-eval.js";

describe("generateSyntheticCorpus", () => {
  it("is deterministic for a fixed seed", () => {
    const a = generateSyntheticCorpus({ count: 12, seed: 7 });
    const b = generateSyntheticCorpus({ count: 12, seed: 7 });
    expect(a).toEqual(b);
  });

  it("varies with the seed", () => {
    const a = generateSyntheticCorpus({ count: 12, seed: 1 });
    const b = generateSyntheticCorpus({ count: 12, seed: 2 });
    expect(a).not.toEqual(b);
  });

  it("cycles through the template library and produces non-empty flows", () => {
    const corpus = generateSyntheticCorpus({ count: DEFAULT_MOVEMENT_TEMPLATES.length, seed: 3 });
    expect(corpus).toHaveLength(DEFAULT_MOVEMENT_TEMPLATES.length);
    for (const sequence of corpus) {
      expect(sequence.steps.length).toBeGreaterThan(0);
      expect(sequence.appId).toBeTruthy();
    }
  });
});

describe("splitCorpus", () => {
  it("partitions deterministically by ratio without overlap", () => {
    const corpus = generateSyntheticCorpus({ count: 10, seed: 5 });
    const { train, heldOut } = splitCorpus(corpus, 0.7);
    expect(train).toHaveLength(7);
    expect(heldOut).toHaveLength(3);
    const ids = new Set(train.map((s) => s.id));
    expect(heldOut.every((s) => !ids.has(s.id))).toBe(true);
  });
});

describe("evaluateMovementModel", () => {
  it("scores near-perfectly when trained and evaluated on the same flows", () => {
    const corpus = generateSyntheticCorpus({ count: 30, seed: 11 });
    const model = new DeterministicNgramBackend().train(corpus, { order: 3 });
    const report = evaluateMovementModel(model, corpus);
    expect(report.sequences).toBe(30);
    expect(report.predictions).toBeGreaterThan(0);
    expect(report.top1Accuracy).toBeGreaterThan(0.8);
    expect(report.recall).toBeGreaterThanOrEqual(report.top1Accuracy);
  });

  it("generalizes to held-out related flows better than chance", () => {
    const corpus = generateSyntheticCorpus({ count: 60, seed: 21 });
    const { train, heldOut } = splitCorpus(corpus, 0.7);
    const model = new DeterministicNgramBackend().train(train, { order: 3 });
    const report = evaluateMovementModel(model, heldOut);
    // Held-out flows are unseen instances of known templates: the model should
    // predict most next-steps correctly, and its recall must dominate accuracy.
    expect(report.top1Accuracy).toBeGreaterThan(0.5);
    expect(report.recall).toBeGreaterThanOrEqual(report.top1Accuracy);
    expect(report.meanConfidence).toBeGreaterThan(0);
    expect(report.meanConfidence).toBeLessThanOrEqual(1);
  });

  it("reports zeros for an empty held-out set", () => {
    const model = new DeterministicNgramBackend().train(generateSyntheticCorpus({ count: 5, seed: 1 }));
    const report = evaluateMovementModel(model, []);
    expect(report).toEqual({
      sequences: 0,
      predictions: 0,
      top1Accuracy: 0,
      recall: 0,
      meanConfidence: 0,
      backoffRate: 0,
    });
  });
});
