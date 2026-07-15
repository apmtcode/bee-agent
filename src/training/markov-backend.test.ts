import { describe, expect, it } from "vitest";
import { MarkovMovementBackend } from "./markov-backend.js";
import type { MovementDataset } from "./movement-model.js";

function dataset(sequences: string[][]): MovementDataset {
  return {
    version: 1,
    sequences: sequences.map((tokens, index) => ({ sourceId: `seq-${index}`, tokens })),
  };
}

describe("MarkovMovementBackend", () => {
  it("stamps its backend id and training metadata onto the artifact", () => {
    const backend = new MarkovMovementBackend();
    const model = backend.train(dataset([["a", "b", "c"], ["a", "b"]]), { maxOrder: 2 });

    expect(model.backend).toBe("markov-v1");
    expect(model.metadata.sequenceCount).toBe(2);
    expect(model.metadata.tokenCount).toBe(5);
    expect(model.metadata.vocabularySize).toBe(3);
    expect(model.metadata.maxOrder).toBe(2);
  });

  it("repeats a recorded movement exactly via high-order context", () => {
    const backend = new MarkovMovementBackend({ maxOrder: 3 });
    const recorded = ["focus.window", "mouse.move", "mouse.click", "key.type", "key.enter"];
    const model = backend.train(dataset([recorded]));

    // Seeding with the first token should regenerate the whole recording and
    // stop at the learned end-of-sequence, not run past it.
    const replayed = backend.generate(model, [recorded[0]!], 50);
    expect(replayed).toEqual(recorded);
  });

  it("predicts the unique continuation of a seen context with full confidence", () => {
    const backend = new MarkovMovementBackend({ maxOrder: 2 });
    const model = backend.train(dataset([["a", "b", "c"]]));

    const prediction = backend.predict(model, ["a", "b"]);
    expect(prediction.token).toBe("c");
    expect(prediction.confidence).toBe(1);
    expect(prediction.contextOrder).toBe(2);
  });

  it("generalizes to an unseen context by backing off to a shorter shared context", () => {
    const backend = new MarkovMovementBackend({ maxOrder: 3 });
    const model = backend.train(dataset([["open", "edit", "review", "save"]]));

    // The full trigram context ("zoom","edit","review") was never recorded, but
    // its order-2 suffix ("edit","review") was — so the model backs off and
    // still predicts the related continuation "save".
    const prediction = backend.predict(model, ["zoom", "edit", "review"]);
    expect(prediction.token).toBe("save");
    expect(prediction.contextOrder).toBe(2);
  });

  it("weights the distribution by observed frequency and sorts deterministically", () => {
    const backend = new MarkovMovementBackend({ maxOrder: 1 });
    const model = backend.train(dataset([["x", "a"], ["x", "a"], ["x", "b"]]));

    const prediction = backend.predict(model, ["x"]);
    expect(prediction.token).toBe("a");
    expect(prediction.distribution).toEqual([
      { token: "a", probability: 2 / 3 },
      { token: "b", probability: 1 / 3 },
    ]);
  });

  it("returns a null prediction when there is no basis to predict", () => {
    const backend = new MarkovMovementBackend();
    const model = backend.train(dataset([]));

    const prediction = backend.predict(model, ["anything"]);
    expect(prediction.token).toBeNull();
    expect(prediction.confidence).toBe(0);
    expect(prediction.contextOrder).toBe(-1);
  });

  it("does not confuse a delimited context with a single joined token", () => {
    const backend = new MarkovMovementBackend({ maxOrder: 2 });
    // Tokens whose concatenation ("a","b" -> "ab") could collide without a
    // real delimiter between context tokens.
    const model = backend.train(dataset([["a", "b", "left"], ["ab", "right"]]));

    expect(backend.predict(model, ["a", "b"]).token).toBe("left");
    expect(backend.predict(model, ["ab"]).token).toBe("right");
  });

  it("produces a JSON-round-trippable artifact with stable predictions", () => {
    const backend = new MarkovMovementBackend({ maxOrder: 2 });
    const model = backend.train(dataset([["a", "b", "c"], ["a", "b", "c"]]));

    const restored = JSON.parse(JSON.stringify(model));
    expect(backend.predict(restored, ["a", "b"])).toEqual(backend.predict(model, ["a", "b"]));
  });
});
