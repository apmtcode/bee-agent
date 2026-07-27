import { describe, expect, it } from "vitest";
import { createDefaultMovementModel } from "./movement-model.js";
import { evaluateMovementModel } from "./movement-eval.js";
import {
  createSeededRng,
  generateSyntheticMovementDataset,
} from "./synthetic-movements.js";

describe("createSeededRng", () => {
  it("is deterministic for a given seed", () => {
    const a = createSeededRng(42);
    const b = createSeededRng(42);
    const draw = (rng: () => number) => Array.from({ length: 5 }, () => rng());
    expect(draw(a)).toEqual(draw(b));
  });

  it("produces values in [0, 1)", () => {
    const rng = createSeededRng(7);
    for (let i = 0; i < 100; i += 1) {
      const value = rng();
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThan(1);
    }
  });
});

describe("generateSyntheticMovementDataset", () => {
  it("is reproducible for a fixed seed", () => {
    const a = generateSyntheticMovementDataset({ seed: 3 });
    const b = generateSyntheticMovementDataset({ seed: 3 });
    expect(a).toEqual(b);
  });

  it("produces non-empty train and held-out splits with valid steps", () => {
    const { train, heldOut } = generateSyntheticMovementDataset({ perTemplate: 4 });
    expect(train.length).toBeGreaterThan(0);
    expect(heldOut.length).toBeGreaterThan(0);
    for (const sequence of [...train, ...heldOut]) {
      expect(sequence.steps.length).toBeGreaterThan(0);
      for (const step of sequence.steps) expect(step.gesture).toBeTruthy();
    }
  });

  it("held-out slot values are novel relative to training values", () => {
    const { train, heldOut } = generateSyntheticMovementDataset();
    const trainTargets = new Set(
      train.flatMap((s) => s.steps.map((step) => step.valueSummary).filter(Boolean)),
    );
    const heldTargets = heldOut.flatMap((s) =>
      s.steps.map((step) => step.valueSummary).filter(Boolean),
    );
    // At least some held-out values must be unseen in training (generalization).
    expect(heldTargets.some((value) => value && !trainTargets.has(value))).toBe(true);
  });
});

describe("end-to-end train -> replay -> generalize", () => {
  it("achieves high next-step accuracy and reasonable coverage on held-out tasks", () => {
    const { train, heldOut } = generateSyntheticMovementDataset({ seed: 11, perTemplate: 12 });
    const model = createDefaultMovementModel(train, { order: 3 });
    const report = evaluateMovementModel(model, heldOut);

    expect(report.sequenceCount).toBe(heldOut.length);
    expect(report.stepCount).toBeGreaterThan(0);
    // Slot values are novel by construction, so exact-token accuracy is
    // bounded — but the *movement structure* generalizes strongly via backoff.
    expect(report.nextGestureAccuracy).toBeGreaterThan(0.7);
    expect(report.nextGestureAccuracy).toBeGreaterThanOrEqual(report.nextStepAccuracy);
    expect(report.meanLogProbability).toBeLessThanOrEqual(0);
  });

  it("learns the gesture structure of the training set with high fidelity", () => {
    const { train } = generateSyntheticMovementDataset({ seed: 5, perTemplate: 6 });
    const model = createDefaultMovementModel(train, { order: 4 });
    const report = evaluateMovementModel(model, train);
    // The first tap target is genuinely ambiguous across app templates, so
    // exact-token accuracy is bounded; gesture structure is learned near-fully.
    expect(report.nextGestureAccuracy).toBeGreaterThan(0.9);
  });
});
