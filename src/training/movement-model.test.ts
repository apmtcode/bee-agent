import { describe, expect, it } from "vitest";
import {
  DETERMINISTIC_NGRAM_BACKEND,
  DeterministicNgramBackend,
  MOVEMENT_END_TOKEN,
  getMovementBackend,
  listMovementBackends,
  tokenizeSequence,
  tokenizeStep,
  type MovementSequence,
} from "./movement-model.js";

function seq(id: string, steps: MovementSequence["steps"]): MovementSequence {
  return { id, appId: "com.example.app", steps };
}

const LOGIN: MovementSequence[] = [
  seq("a", [
    { gesture: "tap", target: "username-field" },
    { gesture: "type", target: "username-field" },
    { gesture: "tap", target: "password-field" },
    { gesture: "type", target: "password-field" },
    { gesture: "tap", target: "submit-button" },
  ]),
  seq("b", [
    { gesture: "tap", target: "username-field" },
    { gesture: "type", target: "username-field" },
    { gesture: "tap", target: "password-field" },
    { gesture: "type", target: "password-field" },
    { gesture: "tap", target: "submit-button" },
  ]),
];

describe("tokenization", () => {
  it("encodes gesture, target, and direction into a stable token", () => {
    expect(tokenizeStep({ gesture: "tap", target: "submit-button" })).toBe("tap:submit-button");
    expect(tokenizeStep({ gesture: "swipe", direction: "down" })).toBe("swipe:down");
    expect(tokenizeStep({ gesture: "scroll", target: "list", direction: "up" })).toBe("scroll:list:up");
    expect(tokenizeSequence(LOGIN[0])).toEqual([
      "tap:username-field",
      "type:username-field",
      "tap:password-field",
      "type:password-field",
      "tap:submit-button",
    ]);
  });
});

describe("DeterministicNgramBackend", () => {
  it("registers itself and is retrievable from the registry", () => {
    expect(listMovementBackends()).toContain(DETERMINISTIC_NGRAM_BACKEND);
    expect(getMovementBackend(DETERMINISTIC_NGRAM_BACKEND)).toBeInstanceOf(DeterministicNgramBackend);
  });

  it("learns a memorized flow and predicts the next step with full confidence", () => {
    const model = new DeterministicNgramBackend().train(LOGIN, { order: 3 });
    const prediction = model.predictNext(["tap:password-field", "type:password-field"]);
    expect(prediction.token).toBe("tap:submit-button");
    expect(prediction.confidence).toBe(1);
    expect(prediction.backoffOrder).toBe(2);
  });

  it("builds a vocabulary excluding boundary sentinels", () => {
    const model = new DeterministicNgramBackend().train(LOGIN);
    expect(model.vocabulary).toContain("tap:submit-button");
    expect(model.vocabulary).not.toContain(MOVEMENT_END_TOKEN);
  });

  it("generates a complete new sequence that ends cleanly", () => {
    const model = new DeterministicNgramBackend().train(LOGIN, { order: 3 });
    const generated = model.generate();
    expect(generated).toEqual([
      "tap:username-field",
      "type:username-field",
      "tap:password-field",
      "type:password-field",
      "tap:submit-button",
    ]);
    expect(generated).not.toContain(MOVEMENT_END_TOKEN);
  });

  it("generalizes to an unseen context by backing off to a shorter one", () => {
    // Train on two related flows sharing a common suffix.
    const dataset: MovementSequence[] = [
      seq("x", [
        { gesture: "tap", target: "search-box" },
        { gesture: "type", target: "search-box" },
        { gesture: "tap", target: "search-submit" },
      ]),
      seq("y", [
        { gesture: "tap", target: "filter" },
        { gesture: "type", target: "search-box" },
        { gesture: "tap", target: "search-submit" },
      ]),
    ];
    const model = new DeterministicNgramBackend().train(dataset, { order: 3 });
    // This 2-token context was never seen verbatim; the model must back off.
    const prediction = model.predictNext(["tap:something-new", "type:search-box"]);
    expect(prediction.token).toBe("tap:search-submit");
    expect(prediction.backoffOrder).toBeLessThan(2);
    expect(prediction.backoffOrder).toBeGreaterThanOrEqual(0);
  });

  it("returns a null prediction for an empty model", () => {
    const model = new DeterministicNgramBackend().train([]);
    const prediction = model.predictNext(["tap:anything"]);
    expect(prediction.token).toBeNull();
    expect(prediction.confidence).toBe(0);
    expect(prediction.backoffOrder).toBe(-1);
  });

  it("round-trips through serialize/load with identical predictions", () => {
    const backend = new DeterministicNgramBackend();
    const model = backend.train(LOGIN, { order: 3 });
    const snapshot = model.serialize();
    expect(snapshot.version).toBe(1);
    expect(snapshot.backend).toBe(DETERMINISTIC_NGRAM_BACKEND);

    const reloaded = backend.load(snapshot);
    expect(reloaded.generate()).toEqual(model.generate());
    expect(reloaded.predictNext(["tap:username-field"]).token).toBe(
      model.predictNext(["tap:username-field"]).token,
    );
  });

  it("clamps invalid order up to at least 1", () => {
    const model = new DeterministicNgramBackend().train(LOGIN, { order: 0 });
    expect(model.config.order).toBe(1);
  });
});
