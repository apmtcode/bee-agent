import { describe, expect, it } from "vitest";
import { MarkovMovementBackend } from "./movement-model.js";
import { evaluateMovementGeneralization } from "./generalization-eval.js";
import { generateSyntheticMovementSplit } from "./synthetic-movements.js";

const backend = new MarkovMovementBackend();

describe("evaluateMovementGeneralization", () => {
  it("reports high gesture-match accuracy on related held-out sequences", () => {
    const { train, heldOut } = generateSyntheticMovementSplit({
      seed: 99,
      trainCount: 30,
      heldOutCount: 15,
    });
    const model = backend.train(train, { maxOrder: 3 });
    const report = evaluateMovementGeneralization(model, heldOut);

    expect(report.sequenceCount).toBe(heldOut.length);
    expect(report.predictionCount).toBeGreaterThan(0);
    // The model has never seen the held-out targets, but the movement *shapes*
    // are shared — so gesture-match generalization should be strong.
    expect(report.gestureMatchAccuracy).toBeGreaterThan(0.7);
    // And it should still get a meaningful share of exact targets from the
    // fixed steps (e.g. the terminal "send"/"save").
    expect(report.exactMatchAccuracy).toBeGreaterThan(0);
    expect(report.gestureMatchAccuracy).toBeGreaterThanOrEqual(report.exactMatchAccuracy);
  });

  it("uses shape backoff for the generalized predictions", () => {
    const { train, heldOut } = generateSyntheticMovementSplit({
      seed: 5,
      trainCount: 12,
      heldOutCount: 6,
    });
    const model = backend.train(train);
    const report = evaluateMovementGeneralization(model, heldOut);
    const total = Object.values(report.byStrategy).reduce((a, b) => a + b, 0);
    expect(total).toBe(report.predictionCount);
    // At least some predictions require generalizing past exact contexts.
    expect(report.byStrategy.shape + report.byStrategy.unigram).toBeGreaterThan(0);
  });

  it("returns zeroed accuracies for empty held-out input", () => {
    const model = backend.train([]);
    const report = evaluateMovementGeneralization(model, []);
    expect(report.predictionCount).toBe(0);
    expect(report.exactMatchAccuracy).toBe(0);
    expect(report.gestureMatchAccuracy).toBe(0);
  });
});
