import { describe, expect, it } from "vitest";
import { MarkovMovementBackend } from "./movement-model.js";
import { evaluateGeneralization, splitForGeneralization } from "./generalization-eval.js";
import { generateSyntheticMovementDataset } from "./synthetic-movements.js";

describe("splitForGeneralization", () => {
  it("holds out every stride-th sequence deterministically", () => {
    const dataset = generateSyntheticMovementDataset({ repeats: 3 });
    const split = splitForGeneralization(dataset, 3);
    expect(split.train.length + split.test.length).toBe(dataset.sequences.length);
    // With 9 sequences and stride 3, indices 2/5/8 (every 3rd) are held out.
    expect(split.test).toHaveLength(3);
  });

  it("never starves the test set on a tiny dataset", () => {
    const dataset = generateSyntheticMovementDataset({ repeats: 1 });
    const split = splitForGeneralization({ version: 1, sequences: dataset.sequences.slice(0, 2) }, 10);
    expect(split.test.length).toBeGreaterThanOrEqual(1);
    expect(split.train.length).toBeGreaterThanOrEqual(1);
  });
});

describe("evaluateGeneralization", () => {
  it("reports high fidelity on learnable synthetic workflows", async () => {
    const dataset = generateSyntheticMovementDataset({ repeats: 4 });
    const report = await evaluateGeneralization(new MarkovMovementBackend(), dataset, {
      seedSteps: 1,
      train: { order: 3 },
    });

    expect(report.backend).toBe("markov");
    expect(report.trainSequences).toBeGreaterThan(0);
    expect(report.testSequences).toBeGreaterThan(0);
    expect(report.evaluatedSteps).toBeGreaterThan(0);
    // The workflows are fully learnable and repeated, so a held-out copy shares
    // the same structure as a trained copy — the model should predict it well.
    expect(report.reproductionRate).toBe(1);
    expect(report.heldOutStepAccuracy).toBeGreaterThanOrEqual(0.9);
    expect(report.heldOutSequenceAccuracy).toBeGreaterThanOrEqual(0.9);
  });

  it("is reproducible across runs", async () => {
    const dataset = generateSyntheticMovementDataset({ repeats: 4 });
    const a = await evaluateGeneralization(new MarkovMovementBackend(), dataset, { train: { order: 2 } });
    const b = await evaluateGeneralization(new MarkovMovementBackend(), dataset, { train: { order: 2 } });
    expect(a).toEqual(b);
  });

  it("produces a meaningful signal when the model cannot generalize", async () => {
    // Every sequence is unique noise sharing no structure; held-out accuracy
    // should be well below the learnable case, proving the metric discriminates.
    const sequences = Array.from({ length: 6 }, (_, i) => ({
      id: `noise-${i}`,
      context: `ctx-${i}`,
      steps: [
        { ts: 0, gesture: `g${i}a`, summary: "a" },
        { ts: 1, gesture: `g${i}b`, summary: "b" },
        { ts: 2, gesture: `g${i}c`, summary: "c" },
      ],
    }));
    const report = await evaluateGeneralization(
      new MarkovMovementBackend(),
      { version: 1, sequences },
      { seedSteps: 1, train: { order: 2 } },
    );
    expect(report.heldOutStepAccuracy).toBeLessThan(0.5);
  });
});
