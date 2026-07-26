import { describe, expect, it } from "vitest";
import { NGramMovementBackend } from "./movement-model.js";
import {
  evaluateMovementFidelity,
  generateSyntheticMovementCorpus,
  splitMovementDataset,
} from "./movement-eval.js";

describe("synthetic movement corpus", () => {
  it("is deterministic for a fixed seed", () => {
    const a = generateSyntheticMovementCorpus({ seed: 7, sequenceCount: 10 });
    const b = generateSyntheticMovementCorpus({ seed: 7, sequenceCount: 10 });
    expect(JSON.stringify(a)).toEqual(JSON.stringify(b));
  });

  it("varies with the seed", () => {
    const a = generateSyntheticMovementCorpus({ seed: 1, sequenceCount: 10 });
    const b = generateSyntheticMovementCorpus({ seed: 2, sequenceCount: 10 });
    expect(JSON.stringify(a)).not.toEqual(JSON.stringify(b));
  });

  it("produces app-focused workflows ending in a per-app primary action", () => {
    const corpus = generateSyntheticMovementCorpus({ seed: 3, sequenceCount: 6 });
    for (const sequence of corpus.sequences) {
      expect(sequence.events[0].channel).toBe("os");
      expect(sequence.events.at(-1)?.channel).toBe("tool");
    }
  });
});

describe("splitMovementDataset", () => {
  it("partitions deterministically without overlap", () => {
    const corpus = generateSyntheticMovementCorpus({ seed: 5, sequenceCount: 12 });
    const { train, test } = splitMovementDataset(corpus, 4);
    expect(train.sequences).toHaveLength(9);
    expect(test.sequences).toHaveLength(3);
    const trainIds = new Set(train.sequences.map((s) => s.id));
    for (const held of test.sequences) {
      expect(trainIds.has(held.id)).toBe(false);
    }
  });
});

describe("evaluateMovementFidelity", () => {
  it("generalizes to held-out related sequences well above the majority baseline", async () => {
    const corpus = generateSyntheticMovementCorpus({ seed: 11, sequenceCount: 60 });
    const { train, test } = splitMovementDataset(corpus, 4);

    const backend = new NGramMovementBackend();
    const model = await backend.train(train, { order: 3 });
    const report = evaluateMovementFidelity(backend, model, test.sequences);

    expect(report.predictions).toBeGreaterThan(0);
    // The model must genuinely learn structure, not just guess the mode.
    expect(report.accuracy).toBeGreaterThan(report.baselineAccuracy + 0.2);
    expect(report.accuracy).toBeGreaterThan(0.6);
    // At least some held-out predictions come from the exact-context ngram path.
    const ngramHits = Object.entries(report.levelCounts)
      .filter(([level]) => level.startsWith("ngram-"))
      .reduce((sum, [, count]) => sum + count, 0);
    expect(ngramHits).toBeGreaterThan(0);
  });

  it("reports zero predictions for an empty test set", async () => {
    const backend = new NGramMovementBackend();
    const model = await backend.train(generateSyntheticMovementCorpus({ seed: 1, sequenceCount: 4 }));
    const report = evaluateMovementFidelity(backend, model, []);
    expect(report).toMatchObject({ predictions: 0, correct: 0, accuracy: 0, baselineAccuracy: 0 });
  });
});
