import { describe, expect, it } from "vitest";
import { movementTokenKey, type MovementDataset, type MovementToken } from "./movement-dataset.js";
import {
  MarkovMovementBackend,
  createMovementBackend,
  listMovementBackends,
  registerMovementBackend,
} from "./movement-model.js";

const A: MovementToken = { modality: "window", verb: "open", target: "app" };
const B: MovementToken = { modality: "pointer", verb: "tap", target: "one" };
const C: MovementToken = { modality: "keyboard", verb: "type", target: "two" };
const D: MovementToken = { modality: "pointer", verb: "tap", target: "send" };

const fixedDataset: MovementDataset = {
  version: 1,
  sequences: [
    { id: "s1", tokens: [A, B, C, D] },
    { id: "s2", tokens: [A, B, C, D] },
    { id: "s3", tokens: [A, B, C, D] },
  ],
};

describe("MarkovMovementBackend", () => {
  it("learns transition statistics from recorded movements", () => {
    const backend = new MarkovMovementBackend();
    const model = backend.train(fixedDataset, { order: 2 });
    expect(model.trainedSequences).toBe(3);
    expect(model.trainedTokens).toBe(12);
    expect(model.vocabulary).toContain(movementTokenKey(A));
  });

  it("predicts the next recorded movement deterministically", () => {
    const backend = new MarkovMovementBackend();
    const model = backend.train(fixedDataset, { order: 2 });
    expect(backend.predictNext(model, [A])?.key).toBe(movementTokenKey(B));
    expect(backend.predictNext(model, [A, B])?.key).toBe(movementTokenKey(C));
    expect(backend.predictNext(model, [A, B, C])?.key).toBe(movementTokenKey(D));
  });

  it("reproduces a recorded movement sequence via generate()", () => {
    const backend = new MarkovMovementBackend();
    const model = backend.train(fixedDataset, { order: 2 });
    expect(backend.generate(model, { seed: [A], maxLength: 4 })).toEqual([A, B, C, D]);
  });

  it("picks the start movement when generating with no seed", () => {
    const backend = new MarkovMovementBackend();
    const model = backend.train(fixedDataset, { order: 2 });
    expect(backend.generate(model, { maxLength: 4 })).toEqual([A, B, C, D]);
  });

  it("generalizes to an unseen prefix by backing off to a shorter context", () => {
    const backend = new MarkovMovementBackend();
    // Two contexts share the suffix B -> C, but the length-2 context [D, B] is unseen.
    const dataset: MovementDataset = {
      version: 1,
      sequences: [
        { id: "s1", tokens: [A, B, C] },
        { id: "s2", tokens: [A, B, C] },
      ],
    };
    const model = backend.train(dataset, { order: 2 });
    const prediction = backend.predictNext(model, [D, B]);
    expect(prediction?.key).toBe(movementTokenKey(C));
    expect(prediction?.backoff).toBe(true);
    expect(prediction?.contextOrder).toBe(1);
  });

  it("ranks the full next-movement distribution", () => {
    const backend = new MarkovMovementBackend();
    const dataset: MovementDataset = {
      version: 1,
      sequences: [
        { id: "s1", tokens: [A, B] },
        { id: "s2", tokens: [A, B] },
        { id: "s3", tokens: [A, C] },
      ],
    };
    const model = backend.train(dataset, { order: 1 });
    const ranked = backend.rank(model, [A]);
    expect(ranked.map((prediction) => prediction.key)).toEqual([movementTokenKey(B), movementTokenKey(C)]);
    expect(ranked[0]!.probability).toBeCloseTo(2 / 3, 5);
  });
});

describe("movement backend registry", () => {
  it("creates the built-in markov backend", () => {
    expect(createMovementBackend("markov").name).toBe("markov");
    expect(createMovementBackend().name).toBe("markov");
    expect(listMovementBackends()).toContain("markov");
  });

  it("throws for an unknown backend", () => {
    expect(() => createMovementBackend("nope")).toThrow(/unknown movement backend/);
  });

  it("supports registering a custom pluggable backend", () => {
    registerMovementBackend("test-passthrough", () => new MarkovMovementBackend(1));
    expect(createMovementBackend("test-passthrough").name).toBe("markov");
    expect(listMovementBackends()).toContain("test-passthrough");
  });
});
