import { describe, expect, it } from "vitest";
import { MarkovMovementBackend } from "./movement-policy.js";
import {
  defaultMovementWorkflows,
  evaluateNextActionAccuracy,
  evaluateReplayFidelity,
  generateSyntheticMovementDataset,
} from "./movement-eval.js";

describe("generateSyntheticMovementDataset", () => {
  it("is reproducible for a given seed", () => {
    const a = generateSyntheticMovementDataset({ seed: 42, sequenceCount: 20 });
    const b = generateSyntheticMovementDataset({ seed: 42, sequenceCount: 20 });
    expect(a).toEqual(b);
  });

  it("produces different streams for different seeds", () => {
    const a = generateSyntheticMovementDataset({ seed: 1, sequenceCount: 20 });
    const b = generateSyntheticMovementDataset({ seed: 2, sequenceCount: 20 });
    expect(a).not.toEqual(b);
  });

  it("only emits tokens drawn from the workflow grammar", () => {
    const workflows = defaultMovementWorkflows();
    const vocabulary = new Set(workflows.flatMap((w) => w.stages.flatMap((s) => s.options)));
    const data = generateSyntheticMovementDataset({ seed: 7, sequenceCount: 30, workflows });
    for (const sequence of data.sequences) {
      for (const token of sequence.tokens) {
        expect(vocabulary.has(token)).toBe(true);
      }
    }
  });
});

describe("movement-policy generalization", () => {
  it("generalizes to held-out sequences from the same grammar far above baseline", () => {
    const maxOrder = 3;
    const train = generateSyntheticMovementDataset({ seed: 1, sequenceCount: 200 });
    const heldOut = generateSyntheticMovementDataset({ seed: 999, sequenceCount: 60 });

    const model = new MarkovMovementBackend().train(train, { maxOrder });
    const trained = evaluateNextActionAccuracy(model, heldOut.sequences, maxOrder);

    // Unigram baseline: a max-order-0 model that ignores context.
    const baselineModel = new MarkovMovementBackend().train(train, { maxOrder: 0 });
    const baseline = evaluateNextActionAccuracy(baselineModel, heldOut.sequences, 0);

    expect(trained.steps).toBeGreaterThan(0);
    expect(trained.accuracy).toBeGreaterThan(baseline.accuracy + 0.1);
    // Learned structure should predict a clear majority of next actions.
    expect(trained.accuracy).toBeGreaterThan(0.6);
  });

  it("replays training sequences with high fidelity", () => {
    const train = generateSyntheticMovementDataset({ seed: 5, sequenceCount: 120 });
    const model = new MarkovMovementBackend().train(train, { maxOrder: 3 });
    const fidelity = evaluateReplayFidelity(model, train.sequences.slice(0, 30));
    expect(fidelity.sequences).toBe(30);
    expect(fidelity.meanPrefixFidelity).toBeGreaterThan(0.5);
  });

  it("handles an empty held-out set without dividing by zero", () => {
    const model = new MarkovMovementBackend().train(
      generateSyntheticMovementDataset({ seed: 3, sequenceCount: 10 }),
    );
    expect(evaluateNextActionAccuracy(model, []).accuracy).toBe(0);
    expect(evaluateReplayFidelity(model, []).meanPrefixFidelity).toBe(0);
  });
});
