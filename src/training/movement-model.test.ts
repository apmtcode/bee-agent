import { describe, expect, it } from "vitest";
import {
  NgramMovementBackend,
  buildMovementDataset,
  evaluateMovementModel,
  generateSyntheticMovementSequences,
  toMovementDataset,
  type MovementSequence,
  type ReplayLike,
} from "./movement-model.js";

describe("buildMovementDataset", () => {
  it("derives one ordered action sequence per replay, sorted by ts", () => {
    const replays: ReplayLike[] = [
      {
        sessionId: "sess-1",
        trajectoryIds: ["traj-1"],
        events: [
          { kind: "action", ts: 3, tool: "browser", summary: "clicked deploy", trajectoryId: "traj-1" },
          { kind: "observation", ts: 2, summary: "opened", trajectoryId: "traj-1" },
          { kind: "action", ts: 1, tool: "browser", summary: "typed url", trajectoryId: "traj-1" },
        ],
      },
    ];

    const dataset = buildMovementDataset(replays);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]).toMatchObject({ id: "sess-1" });
    // observation dropped; actions reordered by ts.
    expect(dataset.sequences[0]!.tokens).toEqual(["browser›typed url", "browser›clicked deploy"]);
    expect(dataset.vocabulary).toEqual(["browser›clicked deploy", "browser›typed url"]);
  });

  it("supports tool-only granularity and trajectory grouping", () => {
    const replays: ReplayLike[] = [
      {
        sessionId: "sess-1",
        events: [
          { kind: "action", ts: 1, tool: "browser", summary: "a", trajectoryId: "t1" },
          { kind: "action", ts: 2, tool: "editor", summary: "b", trajectoryId: "t2" },
          { kind: "action", ts: 3, tool: "browser", summary: "c", trajectoryId: "t1" },
        ],
      },
    ];

    const dataset = buildMovementDataset(replays, { granularity: "tool", groupBy: "trajectory" });
    const byId = Object.fromEntries(dataset.sequences.map((s) => [s.id, s.tokens]));
    expect(byId.t1).toEqual(["browser", "browser"]);
    expect(byId.t2).toEqual(["editor"]);
  });
});

describe("NgramMovementBackend", () => {
  it("reproduces a memorized movement sequence exactly (repeat)", () => {
    const dataset = toMovementDataset([{ id: "seq", tokens: ["open", "click", "type", "submit"] }]);
    const model = new NgramMovementBackend().train(dataset, { order: 2 });

    expect(model.generate()).toEqual(["open", "click", "type", "submit"]);
  });

  it("predicts the next token deterministically", () => {
    const dataset = toMovementDataset([
      { id: "a", tokens: ["open", "click", "submit"] },
      { id: "b", tokens: ["open", "click", "submit"] },
    ]);
    const model = new NgramMovementBackend().train(dataset, { order: 2 });

    const ranked = model.predictNext(["open"]);
    expect(ranked[0]!.token).toBe("click");
    expect(ranked[0]!.probability).toBeGreaterThan(0.5);
    // Deterministic across calls.
    expect(model.predictNext(["open"])).toEqual(ranked);
  });

  it("serializes and restores to an identical model", () => {
    const dataset = toMovementDataset([{ id: "seq", tokens: ["open", "click", "type"] }]);
    const backend = new NgramMovementBackend();
    const model = backend.train(dataset, { order: 2 });
    const restored = backend.restore(model.serialize());

    expect(restored.serialize()).toEqual(model.serialize());
    expect(restored.generate()).toEqual(model.generate());
  });

  it("generalizes to unseen-but-related sequences via the shared chain", () => {
    // A 3-token chain (a→b→c→a) is fully covered by any sequence of length ≥ 4,
    // so every learnable bigram appears in training and held-out prediction is
    // deterministic — a crisp generalization signal.
    const alphabet = ["a", "b", "c"];
    const all = generateSyntheticMovementSequences({ count: 40, seed: 7, vocabulary: alphabet, minLength: 4, maxLength: 6 });
    const train = all.slice(0, 30);
    const heldOut = all.slice(30);

    const model = new NgramMovementBackend().train(toMovementDataset(train), { order: 2 });
    const result = evaluateMovementModel(model, heldOut, { topK: 2, scoreEnd: false });

    // Every synthetic sequence obeys token i ⇒ token i+1, so a bigram model
    // trained on the train split predicts held-out continuations exactly.
    expect(result.nextTokenAccuracy).toBe(1);
    expect(result.perplexity).toBeLessThan(2);
    expect(result.predictionCount).toBeGreaterThan(0);
  });
});

describe("generateSyntheticMovementSequences", () => {
  it("is deterministic for a fixed seed and follows the learnable chain", () => {
    const vocabulary = ["a", "b", "c"];
    const first = generateSyntheticMovementSequences({ count: 5, seed: 42, vocabulary });
    const second = generateSyntheticMovementSequences({ count: 5, seed: 42, vocabulary });
    expect(first).toEqual(second);

    for (const sequence of first) {
      for (let i = 1; i < sequence.tokens.length; i += 1) {
        const prevIndex = vocabulary.indexOf(sequence.tokens[i - 1]!);
        expect(sequence.tokens[i]).toBe(vocabulary[(prevIndex + 1) % vocabulary.length]);
      }
    }
  });

  it("returns nothing for an empty vocabulary", () => {
    expect(generateSyntheticMovementSequences({ count: 3, seed: 1, vocabulary: [] })).toEqual([]);
  });
});

describe("evaluateMovementModel", () => {
  it("reports a high exact-replay rate on memorized sequences", () => {
    const sequences: MovementSequence[] = [
      { id: "a", tokens: ["open", "click", "submit"] },
      { id: "b", tokens: ["launch", "wait", "confirm"] },
    ];
    const model = new NgramMovementBackend().train(toMovementDataset(sequences), { order: 2 });
    const result = evaluateMovementModel(model, sequences);
    expect(result.exactReplayRate).toBe(1);
    expect(result.sequenceCount).toBe(2);
  });
});
