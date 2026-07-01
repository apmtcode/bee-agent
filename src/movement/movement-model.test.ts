import { describe, expect, it } from "vitest";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import {
  getMovementBackend,
  listMovementBackends,
  MOVEMENT_END,
  NGramMovementModel,
  ngramMovementBackend,
  registerMovementBackend,
  tokenizeAction,
  tokenizeTrajectory,
  type MovementModel,
  type MovementModelBackend,
} from "./movement-model.js";

describe("tokenizeAction", () => {
  it("normalizes tool and summary into a stable token", () => {
    expect(tokenizeAction("  Device ", "Tapped   Save  Button")).toBe("device::tapped save button");
  });
});

describe("tokenizeTrajectory", () => {
  it("orders actions by timestamp", () => {
    const span = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [
        { kind: "action", tool: "device", summary: "second", ts: 20 },
        { kind: "action", tool: "device", summary: "first", ts: 10 },
      ],
    });
    expect(tokenizeTrajectory(span)).toEqual(["device::second", "device::first"].sort());
    // Explicit ordering assertion (sorted-by-ts, not lexical).
    expect(tokenizeTrajectory(span)).toEqual(["device::first", "device::second"]);
  });
});

describe("NGramMovementModel", () => {
  it("repeats a memorized trajectory from a one-token seed", () => {
    const model = NGramMovementModel.train([["a", "b", "c"]]);
    expect(model.generate(["a"])).toEqual(["a", "b", "c"]);
  });

  it("predicts the END sentinel after a memorized sequence", () => {
    const model = NGramMovementModel.train([["a", "b", "c"]]);
    const [best] = model.predict(["a", "b", "c"], { topK: 1 });
    expect(best?.token).toBe(MOVEMENT_END);
  });

  it("generalizes a shared suffix onto a novel prefix via backoff", () => {
    const model = NGramMovementModel.train([
      ["a", "b", "c", "d"],
      ["a", "b", "c", "d"],
      ["x", "b", "c", "d"],
    ]);
    // "q" was never observed, but the suffix "b c" -> "d" was.
    const ranked = model.predict(["q", "b", "c"], { topK: 1 });
    expect(ranked[0]?.token).toBe("d");
    expect(ranked[0]?.order).toBe(2);
    // A full rollout from a novel prefix completes the learned flow.
    expect(model.generate(["q", "b", "c"])).toEqual(["q", "b", "c", "d"]);
  });

  it("ranks branch continuations by observed frequency", () => {
    const model = NGramMovementModel.train([
      ["m", "n", "p"],
      ["m", "n", "p"],
      ["m", "n", "q"],
    ]);
    const ranked = model.predict(["m", "n"]);
    expect(ranked[0]?.token).toBe("p");
    expect(ranked[0]?.probability).toBeCloseTo(2 / 3);
    expect(ranked[1]?.token).toBe("q");
  });

  it("returns no predictions for an empty model", () => {
    const model = NGramMovementModel.train([]);
    expect(model.predict(["anything"])).toEqual([]);
    expect(model.generate(["anything"])).toEqual(["anything"]);
    expect(model.vocabulary()).toEqual([]);
  });

  it("round-trips through serialize/load", () => {
    const model = NGramMovementModel.train([
      ["a", "b", "c", "d"],
      ["x", "b", "c", "e"],
    ]);
    const restored = NGramMovementModel.load(model.serialize());
    expect(restored.order).toBe(model.order);
    expect(restored.vocabulary()).toEqual(model.vocabulary());
    expect(restored.predict(["a", "b", "c"])).toEqual(model.predict(["a", "b", "c"]));
    // Serialization is deterministic (stable ordering).
    expect(restored.serialize()).toEqual(model.serialize());
  });

  it("respects the configured n-gram order", () => {
    // Order 1 keeps only unigram context, so it cannot distinguish branches.
    const model = NGramMovementModel.train([["a", "b"], ["c", "d"]], { order: 1 });
    expect(model.order).toBe(1);
    const ranked = model.predict(["a"]);
    // With no usable context, every observed successor is a candidate.
    expect(ranked.length).toBeGreaterThan(1);
  });
});

describe("movement backend registry", () => {
  it("exposes the default ngram backend", () => {
    expect(listMovementBackends()).toContain("ngram-backoff");
    expect(getMovementBackend("ngram-backoff")).toBe(ngramMovementBackend);
  });

  it("throws for an unknown backend", () => {
    expect(() => getMovementBackend("does-not-exist")).toThrow(/unknown movement-model backend/);
  });

  it("allows registering a custom pluggable backend", () => {
    const fakeModel = { backend: "fake" } as unknown as MovementModel;
    const backend: MovementModelBackend = {
      name: "test-fake-backend",
      train: () => fakeModel,
      load: () => fakeModel,
    };
    registerMovementBackend(backend);
    expect(getMovementBackend("test-fake-backend")).toBe(backend);
    expect(getMovementBackend("test-fake-backend").train([])).toBe(fakeModel);
  });
});
