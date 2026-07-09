import { describe, expect, it } from "vitest";
import { MarkovMovementBackend } from "./movement-model.js";
import {
  createSeededRng,
  defaultMovementGrammar,
  evaluateNextTokenAccuracy,
  generateSyntheticMovementDataset,
} from "./movement-eval.js";

describe("createSeededRng", () => {
  it("is deterministic for a given seed", () => {
    const a = createSeededRng(42);
    const b = createSeededRng(42);
    const seqA = [a(), a(), a()];
    const seqB = [b(), b(), b()];
    expect(seqA).toEqual(seqB);
    expect(seqA.every((value) => value >= 0 && value < 1)).toBe(true);
  });

  it("produces different streams for different seeds", () => {
    expect(createSeededRng(1)()).not.toBe(createSeededRng(2)());
  });
});

describe("generateSyntheticMovementDataset", () => {
  it("is deterministic and grammar-constrained", () => {
    const first = generateSyntheticMovementDataset({ seed: 7, sequenceCount: 5 });
    const second = generateSyntheticMovementDataset({ seed: 7, sequenceCount: 5 });
    expect(first).toEqual(second);
    expect(first.sequences).toHaveLength(5);

    const validTokens = new Set(Object.values(defaultMovementGrammar().steps).map((step) => step.token));
    for (const sequence of first.sequences) {
      expect(sequence.tokens.length).toBeGreaterThan(0);
      for (const token of sequence.tokens) {
        expect(validTokens.has(token)).toBe(true);
      }
    }
  });

  it("varies with the seed", () => {
    const a = generateSyntheticMovementDataset({ seed: 1, sequenceCount: 20 });
    const b = generateSyntheticMovementDataset({ seed: 2, sequenceCount: 20 });
    expect(a.sequences).not.toEqual(b.sequences);
  });
});

describe("evaluateNextTokenAccuracy", () => {
  it("generalizes to held-out sequences from the same grammar", async () => {
    // Train and test on DISJOINT synthetic draws (different seeds) so held-out
    // accuracy measures generalization, not memorization.
    const train = generateSyntheticMovementDataset({ seed: 100, sequenceCount: 120 });
    const heldOut = generateSyntheticMovementDataset({ seed: 999, sequenceCount: 40 });

    const backend = new MarkovMovementBackend({ order: 3 });
    const model = await backend.train(train);
    const result = evaluateNextTokenAccuracy(backend, model, heldOut.sequences);

    expect(result.predictions).toBeGreaterThan(0);
    // The grammar's most-likely path is highly predictable; a back-off n-gram
    // should recover the majority of next tokens on unseen sequences.
    expect(result.accuracy).toBeGreaterThan(0.6);
  });

  it("scores a perfectly-memorized single sequence at 100%", async () => {
    const backend = new MarkovMovementBackend({ order: 2 });
    const dataset = { version: 1 as const, sequences: [{ id: "s", tokens: ["a", "b", "c", "d"] }] };
    const model = await backend.train(dataset);
    const result = evaluateNextTokenAccuracy(backend, model, dataset.sequences);
    expect(result.accuracy).toBe(1);
    expect(result.correct).toBe(result.predictions);
  });
});
