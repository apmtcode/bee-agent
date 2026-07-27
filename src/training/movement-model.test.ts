import { describe, expect, it } from "vitest";
import {
  MarkovMovementBackend,
  MarkovMovementModel,
  MovementBackendRegistry,
  MOVEMENT_END_TOKEN,
  createDefaultMovementBackendRegistry,
  type MovementDataset,
} from "./movement-model.js";

function datasetFrom(sequences: string[][]): MovementDataset {
  const vocabulary = new Set<string>(["start", "end"]);
  let tokenCount = 0;
  for (const tokens of sequences) {
    for (const token of tokens) {
      vocabulary.add(token);
      tokenCount += 1;
    }
  }
  return {
    version: 1,
    createdAt: "2026-07-27T00:00:00.000Z",
    vocabulary: [...vocabulary].sort(),
    tokenCount,
    sequences: sequences.map((tokens, index) => ({
      id: `seq-${index}`,
      sourceTrajectoryIds: [`traj-${index}`],
      events: tokens.map((token, i) => ({ ts: i, kind: "action" as const, channel: "x", verb: token, token })),
      tokens,
    })),
  };
}

describe("MarkovMovementBackend", () => {
  it("trains deterministically and repeats a memorized sequence", async () => {
    const backend = new MarkovMovementBackend();
    const model = await backend.train(datasetFrom([["a", "b", "c", "d"]]), { order: 2 });

    // Greedy generation reproduces the single recorded movement sequence.
    expect(model.generate([], { stopToken: MOVEMENT_END_TOKEN })).toEqual(["a", "b", "c", "d"]);
    // Same input trained twice yields identical predictions.
    const again = await backend.train(datasetFrom([["a", "b", "c", "d"]]), { order: 2 });
    expect(again.predictNext(["a"])).toEqual(model.predictNext(["a"]));
  });

  it("predicts the most likely next token with a deterministic tie-break", async () => {
    const backend = new MarkovMovementBackend();
    // After "click": twice → "type", once → "scroll". Highest prob wins.
    const model = await backend.train(
      datasetFrom([
        ["click", "type"],
        ["click", "type"],
        ["click", "scroll"],
      ]),
      { order: 1 },
    );
    const prediction = model.predictNext(["click"]);
    expect(prediction?.token).toBe("type");
    expect(prediction?.probability).toBeCloseTo(2 / 3);
    expect(prediction?.backoffOrder).toBe(1);
    expect(prediction?.alternatives[0]?.token).toBe("scroll");
  });

  it("generalizes to an unseen context via backoff", async () => {
    const backend = new MarkovMovementBackend();
    // "submit" always follows "type" across sequences, but the specific
    // order-2 context ["hover","type"] is never seen — backoff to order-1
    // must still predict "submit".
    const model = await backend.train(
      datasetFrom([
        ["click", "type", "submit"],
        ["focus", "type", "submit"],
        ["scroll", "type", "submit"],
      ]),
      { order: 2 },
    );
    const prediction = model.predictNext(["hover", "type"]);
    expect(prediction?.token).toBe("submit");
    // Full order-2 context was unseen, so a shorter context did the work.
    expect(prediction?.backoffOrder).toBeLessThan(2);
  });

  it("returns undefined when nothing was ever trained", async () => {
    const model = await new MarkovMovementBackend().train(datasetFrom([]), { order: 2 });
    expect(model.predictNext(["anything"])).toBeUndefined();
    expect(model.generate()).toEqual([]);
  });

  it("survives a serialization round-trip", async () => {
    const model = await new MarkovMovementBackend().train(datasetFrom([["a", "b", "c"]]), { order: 2 });
    const restored = MarkovMovementModel.fromJSON(model.toJSON());
    expect(restored.predictNext(["a", "b"])).toEqual(model.predictNext(["a", "b"]));
    expect(restored.generate()).toEqual(model.generate());
  });
});

describe("MovementBackendRegistry", () => {
  it("resolves a registered backend and lists ids", () => {
    const registry = createDefaultMovementBackendRegistry();
    expect(registry.list()).toContain("markov-mock");
    expect(registry.resolve("markov-mock")).toBeInstanceOf(MarkovMovementBackend);
  });

  it("throws a helpful error for an unknown backend", () => {
    const registry = new MovementBackendRegistry();
    expect(() => registry.resolve("real-mlx")).toThrow(/unknown movement model backend: real-mlx/);
  });
});
