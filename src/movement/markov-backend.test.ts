import { describe, expect, it } from "vitest";
import { MarkovMovementBackend } from "./markov-backend.js";
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

describe("MarkovMovementBackend", () => {
  it("predicts the next token after a memorized context", () => {
    const backend = new MarkovMovementBackend(2);
    const model = backend.train(
      dataset([
        ["tap:email", "type:email", "tap:submit"],
        ["tap:email", "type:email", "tap:submit"],
      ]),
    );
    const prediction = model.predictNext(["tap:email", "type:email"]);
    expect(prediction.token).toBe("tap:submit");
    expect(prediction.order).toBe(2);
    expect(prediction.probability).toBeCloseTo(1);
  });

  it("is deterministic with a lexicographic tie-break", () => {
    const backend = new MarkovMovementBackend(1);
    // From context "a": "b" and "c" each appear once → tie broken lexically → "b".
    const model = backend.train(dataset([["a", "b"], ["a", "c"]]));
    const first = model.predictNext(["a"]);
    const second = model.predictNext(["a"]);
    expect(first.token).toBe("b");
    expect(second.token).toBe(first.token);
    expect(first.alternatives.map((alt) => alt.token)).toEqual(["b", "c"]);
  });

  it("backs off to a shorter context for unseen full contexts (generalization)", () => {
    const backend = new MarkovMovementBackend(3);
    // "type:email" is always followed by "tap:submit" in training. A NEW, longer
    // context that has never been seen as a trigram should still resolve via
    // backoff to the known bigram continuation.
    const model = backend.train(
      dataset([
        ["tap:email", "type:email", "tap:submit"],
        ["tap:email", "type:email", "tap:submit"],
      ]),
    );
    const prediction = model.predictNext(["scroll:down", "tap:phone", "type:email"]);
    expect(prediction.token).toBe("tap:submit");
    // Answered below the trained order ⇒ the model generalized via backoff.
    expect(prediction.order).toBeLessThan(3);
  });

  it("generates a full continuation that terminates at the modeled boundary", () => {
    const backend = new MarkovMovementBackend(2);
    const model = backend.train(
      dataset([
        ["tap:email", "type:email", "tap:submit"],
        ["tap:email", "type:email", "tap:submit"],
      ]),
    );
    const rollout = model.generate([], { maxSteps: 10 });
    expect(rollout).toEqual(["tap:email", "type:email", "tap:submit"]);
    // Boundary tokens are never emitted.
    expect(rollout).not.toContain("END");
    expect(rollout).not.toContain("START");
  });

  it("returns a null prediction when it has no basis at all", () => {
    const backend = new MarkovMovementBackend(2);
    const model = backend.train({ sequences: [] });
    expect(model.predictNext(["anything"]).token).toBeNull();
  });

  it("round-trips through serialize/restore", () => {
    const backend = new MarkovMovementBackend(2);
    const model = backend.train(dataset([["a", "b", "c"], ["a", "b", "c"]]));
    const restored = backend.restore(model.serialize());
    expect(restored.predictNext(["a", "b"]).token).toBe("c");
    expect(restored.generate([], { maxSteps: 10 })).toEqual(["a", "b", "c"]);
  });

  it("rejects invalid restore state", () => {
    const backend = new MarkovMovementBackend(2);
    expect(() => backend.restore({ backend: "nope" })).toThrow();
  });
});
