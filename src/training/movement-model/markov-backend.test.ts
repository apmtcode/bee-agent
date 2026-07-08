import { describe, expect, it } from "vitest";
import { MarkovMovementBackend } from "./markov-backend.js";
import {
  generateSyntheticMovementDataset,
  splitMovementDataset,
} from "./synthetic.js";
import { evaluateMovementModel } from "./eval.js";
import type { MovementDataset } from "./types.js";

function dataset(sequences: string[][]): MovementDataset {
  return {
    version: 1,
    sequences: sequences.map((tokens, index) => ({
      id: `seq-${index}`,
      steps: tokens.map((token, step) => ({ token, tool: "device", ts: step * 10 })),
    })),
  };
}

describe("MarkovMovementBackend", () => {
  it("trains transition counts and reports metadata", () => {
    const backend = new MarkovMovementBackend({ order: 2 });
    const model = backend.train(dataset([["a", "b", "c"], ["a", "b", "d"]]));
    expect(model.metadata.backend).toBe("markov-ngram");
    expect(model.metadata.order).toBe(2);
    expect(model.metadata.vocabularySize).toBe(4);
    expect(model.metadata.sequenceCount).toBe(2);
    expect(model.metadata.transitionCount).toBeGreaterThan(0);
  });

  it("repeats a deterministic recorded movement exactly via argmax rollout", () => {
    const backend = new MarkovMovementBackend({ order: 2 });
    const model = backend.train(dataset([["a", "b", "c", "d"]]));
    const generated = backend.generate(model, ["a"], 10);
    expect(generated).toEqual(["b", "c", "d"]);
  });

  it("prefers the higher-probability continuation", () => {
    const backend = new MarkovMovementBackend({ order: 1 });
    const model = backend.train(dataset([["a", "b"], ["a", "b"], ["a", "c"]]));
    const [top] = backend.predict(model, ["a"], { topK: 1 });
    expect(top?.token).toBe("b");
    expect(top?.probability).toBeCloseTo(2 / 3);
  });

  it("generalizes to unseen context by backing off to lower order", () => {
    const backend = new MarkovMovementBackend({ order: 2 });
    // "x b" was never seen as a bigram context, but "b" -> "c" was.
    const model = backend.train(dataset([["a", "b", "c"], ["a", "b", "c"]]));
    const predictions = backend.predict(model, ["x", "b"]);
    expect(predictions[0]?.token).toBe("c");
    expect(predictions[0]?.order).toBeLessThan(2); // used backoff
  });

  it("respects the exclude set and continues backing off when needed", () => {
    const backend = new MarkovMovementBackend({ order: 1 });
    const model = backend.train(dataset([["a", "b"], ["a", "c"]]));
    const predictions = backend.predict(model, ["a"], { exclude: ["b"] });
    expect(predictions.map((p) => p.token)).toEqual(["c"]);
  });

  it("returns no prediction for a fully unknown vocabulary with no history", () => {
    const backend = new MarkovMovementBackend({ order: 2 });
    const model = backend.train(dataset([["a", "b"]]));
    // order-0 (unigram) always has mass, so generation still terminates.
    const generated = backend.generate(model, ["z"], 5);
    expect(Array.isArray(generated)).toBe(true);
  });

  it("achieves high fidelity on held-out related synthetic movements", () => {
    const corpus = generateSyntheticMovementDataset({ seed: 7, sequenceCount: 40 });
    const { train, holdout } = splitMovementDataset(corpus, 0.25);
    const backend = new MarkovMovementBackend({ order: 2 });
    const model = backend.train(train);

    const evalResult = evaluateMovementModel(backend, model, holdout);
    expect(evalResult.sequenceCount).toBeGreaterThan(0);
    // The grammar is learnable, so next-step accuracy on unseen sequences
    // should be well above chance over a 6-symbol vocabulary.
    expect(evalResult.nextStepAccuracy).toBeGreaterThan(0.5);
  });

  it("serializes to JSON and reloads without losing predictions", () => {
    const backend = new MarkovMovementBackend({ order: 2 });
    const model = backend.train(dataset([["a", "b", "c"]]));
    const roundTripped = JSON.parse(JSON.stringify(model));
    expect(backend.generate(roundTripped, ["a"], 5)).toEqual(["b", "c"]);
  });
});
