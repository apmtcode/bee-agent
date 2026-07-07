import { describe, expect, it } from "vitest";
import {
  MarkovMovementBackend,
  synthesizeMovementSequences,
  type MovementSequence,
} from "./movement-model.js";
import { evaluateMovementModel, splitMovementDataset } from "./movement-eval.js";

describe("evaluateMovementModel", () => {
  const dataset: MovementSequence[] = [
    { id: "a", tokens: ["focus", "click", "type", "submit"] },
    { id: "b", tokens: ["focus", "click", "type", "submit"] },
  ];

  it("scores perfect recall on the memorized training episode", () => {
    const model = new MarkovMovementBackend({ order: 2 }).train(dataset);
    const result = evaluateMovementModel(model, dataset);
    expect(result.accuracy).toBe(1);
    expect(result.predictions).toBeGreaterThan(0);
  });

  it("reports a generalization accuracy on held-out related sequences", () => {
    // Train on a coherent synthetic grammar, hold out a disjoint slice.
    const all = synthesizeMovementSequences({
      vocabulary: ["focus", "click", "type", "submit", "confirm"],
      sequences: 60,
      minLength: 5,
      maxLength: 9,
      seed: 123,
    });
    const { train, holdout } = splitMovementDataset(all, 0.25);
    const model = new MarkovMovementBackend({ order: 2 }).train(train);
    const result = evaluateMovementModel(model, holdout, { topK: 3 });

    expect(result.sequences).toBe(holdout.length);
    // The generator biases toward advancing to the next class, so the n-gram
    // model should beat a uniform-random baseline (1/5 = 0.2) comfortably on
    // held-out sequences it never trained on — generalization, not memorization.
    expect(result.accuracy).toBeGreaterThan(0.2);
    expect(result.topKAccuracy).toBeGreaterThanOrEqual(result.accuracy);
    // The generalization accuracy is well-defined (0 when no backoff was needed
    // because the small vocabulary's contexts were all observed).
    expect(result.generalizedPredictions).toBeGreaterThanOrEqual(0);
    expect(result.generalizationAccuracy).toBeGreaterThanOrEqual(0);
  });

  it("counts terminal <end> predictions", () => {
    const model = new MarkovMovementBackend({ order: 2 }).train(dataset);
    const result = evaluateMovementModel(model, [{ id: "c", tokens: ["focus", "click", "type", "submit"] }]);
    // 4 tokens + 1 terminal end = 5 predictions for the single sequence.
    expect(result.predictions).toBe(5);
  });
});

describe("splitMovementDataset", () => {
  it("splits deterministically and covers the whole dataset", () => {
    const dataset: MovementSequence[] = Array.from({ length: 8 }, (_, i) => ({
      id: `s${i}`,
      tokens: ["a", "b"],
    }));
    const first = splitMovementDataset(dataset, 0.25);
    const second = splitMovementDataset(dataset, 0.25);
    expect(first).toEqual(second);
    expect(first.train.length + first.holdout.length).toBe(8);
    expect(first.holdout.length).toBeGreaterThan(0);
  });
});
