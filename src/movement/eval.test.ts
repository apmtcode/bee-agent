import { describe, expect, it } from "vitest";
import { MarkovMovementBackend } from "./markov-backend.js";
import { evaluateMovementModel, splitMovementDataset } from "./eval.js";
import { generateSyntheticMovementDataset } from "./synthetic.js";
import type { MovementDataset } from "./movement-event.js";

function dataset(sequences: string[][]): MovementDataset {
  return {
    sequences: sequences.map((tokens, index) => ({
      id: `seq-${index}`,
      sessionId: `session-${index}`,
      events: tokens.map((token, i) => ({ ts: i, token, action: token.split(":")[0] })),
    })),
  };
}

describe("splitMovementDataset", () => {
  it("holds out every Nth sequence deterministically", () => {
    const data = dataset([["a"], ["b"], ["c"], ["d"], ["e"], ["f"]]);
    const { train, holdout } = splitMovementDataset(data, 3);
    expect(holdout.sequences.map((s) => s.events[0].token)).toEqual(["c", "f"]);
    expect(train.sequences.map((s) => s.events[0].token)).toEqual(["a", "b", "d", "e"]);
  });
});

describe("evaluateMovementModel", () => {
  it("scores perfect accuracy when held-out equals training data", () => {
    const backend = new MarkovMovementBackend(2);
    const data = dataset([["tap:a", "type:a", "tap:submit"], ["tap:a", "type:a", "tap:submit"]]);
    const model = backend.train(data);
    const report = evaluateMovementModel(model, data);
    expect(report.top1Accuracy).toBeCloseTo(1);
    expect(report.predictionCount).toBe(6);
  });

  it("generalizes to held-out but related synthetic movements above chance", () => {
    const backend = new MarkovMovementBackend(3);
    const data = generateSyntheticMovementDataset({ seed: 123, count: 60 });
    const { train, holdout } = splitMovementDataset(data, 4);
    const model = backend.train(train);
    const report = evaluateMovementModel(model, holdout);

    expect(report.sequenceCount).toBeGreaterThan(0);
    // The task grammars carry irreducible randomness (which field / submit
    // target is chosen), so perfect accuracy is impossible. But the vocabulary
    // is small and overlapping, so a backoff Markov model predicts the next
    // held-out movement well above chance (~0.15 for the random-choice steps).
    expect(report.top1Accuracy).toBeGreaterThan(0.35);
    // The true next movement is almost always among the ranked alternatives.
    expect(report.recall).toBeGreaterThan(0.7);
    expect(report.recall).toBeGreaterThanOrEqual(report.top1Accuracy);
    // Generalization signal: many held-out steps are answered below full order.
    expect(report.meanOrder).toBeLessThan(3);
  });

  it("reports per-sequence breakdown and a backoff histogram", () => {
    const backend = new MarkovMovementBackend(2);
    const data = dataset([["a", "b"], ["a", "b"]]);
    const model = backend.train(data);
    const report = evaluateMovementModel(model, data);
    expect(report.perSequence).toHaveLength(2);
    expect(Object.values(report.backoffOrderHistogram).reduce((a, b) => a + b, 0)).toBe(
      report.predictionCount,
    );
  });
});
