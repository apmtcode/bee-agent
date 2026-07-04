import { describe, expect, it } from "vitest";
import {
  MOVEMENT_END_TOKEN,
  MarkovMovementBackend,
  type MovementDataset,
} from "./movement-model.js";

function dataset(sequences: string[][]): MovementDataset {
  return {
    version: 1,
    sequences: sequences.map((tokens, index) => ({ id: `seq-${index}`, tokens })),
  };
}

describe("MarkovMovementBackend", () => {
  it("repeats a recorded movement sequence from an empty seed", () => {
    const backend = new MarkovMovementBackend();
    const model = backend.train(dataset([["tap:submit", "type:field", "tap:save"]]), { order: 2 });

    expect(model.generate([])).toEqual(["tap:submit", "type:field", "tap:save"]);
  });

  it("predicts the most likely next movement deterministically", () => {
    const backend = new MarkovMovementBackend();
    const model = backend.train(
      dataset([
        ["open", "click", "submit"],
        ["open", "click", "submit"],
        ["open", "click", "cancel"],
      ]),
      { order: 1 },
    );

    const predictions = model.predict(["click"]);
    expect(predictions[0]).toMatchObject({ token: "submit" });
    expect(predictions[0]!.probability).toBeCloseTo(2 / 3, 10);
    expect(model.predictNext(["click"])).toBe("submit");
    // Deterministic ordering: same probabilities break ties by token asc.
    expect(model.predict(["click"]).map((p) => p.token)).toEqual(["submit", "cancel"]);
  });

  it("generalizes to an unseen context via back-off", () => {
    const backend = new MarkovMovementBackend();
    const model = backend.train(
      dataset([
        ["scroll:down", "tap:link", "tap:back"],
        ["swipe:left", "tap:link", "tap:back"],
      ]),
      { order: 2 },
    );

    // The bigram context ["type:new", "tap:link"] was never observed, but
    // backing off to the unigram ["tap:link"] still yields the learned follow-up.
    expect(model.predictNext(["type:new", "tap:link"])).toBe("tap:back");
  });

  it("stops generation at the end sentinel and honors stopToken", () => {
    const backend = new MarkovMovementBackend();
    const model = backend.train(dataset([["a", "b", "c"]]), { order: 2 });

    expect(model.generate([])).toEqual(["a", "b", "c"]);
    expect(model.generate([], { stopToken: "c" })).toEqual(["a", "b"]);
    expect(model.predictNext(["a", "b", "c"])).toBeUndefined();
  });

  it("never emits boundary sentinels as movements", () => {
    const backend = new MarkovMovementBackend();
    const model = backend.train(dataset([["only"]]), { order: 2 });

    const generated = model.generate([]);
    expect(generated).not.toContain(MOVEMENT_END_TOKEN);
    expect(generated).not.toContain("START");
  });

  it("round-trips through serialize / deserialize", () => {
    const backend = new MarkovMovementBackend();
    const model = backend.train(dataset([["x", "y", "z"], ["x", "y", "w"]]), { order: 2 });
    const restored = MarkovMovementBackend.deserialize(model.serialize());

    expect(restored.order).toBe(model.order);
    expect(restored.predict(["x", "y"])).toEqual(model.predict(["x", "y"]));
    expect(restored.generate([])).toEqual(model.generate([]));
  });

  it("returns no predictions for an empty model", () => {
    const backend = new MarkovMovementBackend();
    const model = backend.train(dataset([]), { order: 2 });

    expect(model.predict(["anything"])).toEqual([]);
    expect(model.predictNext([])).toBeUndefined();
    expect(model.generate([])).toEqual([]);
  });
});
