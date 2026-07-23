import { describe, expect, it } from "vitest";
import { MarkovMovementBackend } from "./backend.js";
import { generateSyntheticDataset } from "./event.js";
import {
  evaluateNextEventAccuracy,
  evaluateReplayFidelity,
  splitSequences,
} from "./eval.js";

const backend = new MarkovMovementBackend();

describe("movement/eval", () => {
  it("splits sequences deterministically without leakage", () => {
    const dataset = generateSyntheticDataset({ seed: 5, count: 10 });
    const { train, test } = splitSequences(dataset.sequences, 0.7);
    expect(train).toHaveLength(7);
    expect(test).toHaveLength(3);
    const trainIds = new Set(train.map((s) => s.id));
    expect(test.every((s) => !trainIds.has(s.id))).toBe(true);
  });

  it("achieves perfect replay fidelity on a recorded sequence (repeat)", async () => {
    // A model post-trained on a single recording with distinct steps repeats it
    // exactly under argmax rollout.
    const recorded = {
      id: "recorded-form",
      events: [
        { t: 0, type: "move" as const, x: 400, y: 200, target: "name-field" },
        { t: 120, type: "click" as const, button: "left" as const, target: "name-field" },
        { t: 300, type: "type" as const, key: "text" },
        { t: 900, type: "move" as const, x: 500, y: 360, target: "submit-button" },
        { t: 1020, type: "click" as const, button: "left" as const, target: "submit-button" },
      ],
    };
    const model = await backend.train({ version: 1, sequences: [recorded] }, { order: 3 });
    const report = await evaluateReplayFidelity(backend, model, recorded, {
      seedLength: 1,
    });
    expect(report.fidelity).toBe(1);
    expect(report.matchedTokens).toBe(report.expectedTokens);
  });

  it("generalizes to held-out related sequences well above chance", async () => {
    // Many jittered variants of the same small template library: the held-out
    // sequences are new, but structurally related to training data.
    const dataset = generateSyntheticDataset({ seed: 21, count: 40, jitter: 6 });
    const { train, test } = splitSequences(dataset.sequences, 0.75);
    const model = await backend.train({ version: 1, sequences: train }, { order: 3 });

    const report = await evaluateNextEventAccuracy(backend, model, test, { k: 3 });
    expect(report.predictionCount).toBeGreaterThan(0);
    // The templates are highly regular, so top-1 on held-out data should be strong.
    expect(report.top1Accuracy).toBeGreaterThan(0.7);
    expect(report.topKAccuracy).toBeGreaterThanOrEqual(report.top1Accuracy);
    expect(report.meanBackoffOrder).toBeGreaterThan(0);
  });

  it("reports zero accuracy on an empty test set", async () => {
    const dataset = generateSyntheticDataset({ seed: 1, count: 4 });
    const model = await backend.train(dataset, { order: 2 });
    const report = await evaluateNextEventAccuracy(backend, model, []);
    expect(report.predictionCount).toBe(0);
    expect(report.top1Accuracy).toBe(0);
    expect(report.topKAccuracy).toBe(0);
  });

  it("higher order improves specificity (mean backoff order) on regular data", async () => {
    const dataset = generateSyntheticDataset({ seed: 33, count: 30 });
    const { train, test } = splitSequences(dataset.sequences, 0.7);
    const lowOrder = await backend.train({ version: 1, sequences: train }, { order: 1 });
    const highOrder = await backend.train({ version: 1, sequences: train }, { order: 3 });
    const low = await evaluateNextEventAccuracy(backend, lowOrder, test);
    const high = await evaluateNextEventAccuracy(backend, highOrder, test);
    expect(high.meanBackoffOrder).toBeGreaterThanOrEqual(low.meanBackoffOrder);
  });
});
