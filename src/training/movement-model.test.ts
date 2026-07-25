import { describe, expect, it } from "vitest";
import { buildTrajectorySpan, type TrajectoryAction } from "../capture/trajectory.js";
import {
  MOVEMENT_END,
  NgramMovementBackend,
  deriveMovementDataset,
  deserializeMovementModel,
  evaluateNextTokenAccuracy,
  splitMovementDataset,
  synthesizeMovementSequences,
  tokenizeAction,
  type MovementDataset,
} from "./movement-model.js";

function action(tool: string, ts: number, movement?: string): TrajectoryAction {
  return {
    kind: "action",
    tool,
    summary: `${tool} @ ${ts}`,
    ts,
    ...(movement ? { metadata: { movement } } : {}),
  };
}

describe("tokenizeAction", () => {
  it("prefers an explicit movement metadata token over the tool name", () => {
    expect(tokenizeAction(action("mouse", 1, "click:submit"))).toBe("click:submit");
    expect(tokenizeAction(action("mouse", 1))).toBe("mouse");
  });
});

describe("deriveMovementDataset", () => {
  it("orders actions by timestamp and drops sub-minimum sequences", () => {
    const trajectories = [
      buildTrajectorySpan({
        id: "t1",
        sessionId: "s1",
        actions: [action("b", 20, "b"), action("a", 10, "a"), action("c", 30, "c")],
      }),
      buildTrajectorySpan({ id: "t2", sessionId: "s1", actions: [] }),
    ];
    const dataset = deriveMovementDataset(trajectories);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]?.tokens).toEqual(["a", "b", "c"]);
  });
});

describe("NgramMovementBackend", () => {
  const dataset: MovementDataset = {
    version: 1,
    sequences: [
      { id: "s1", tokens: ["open", "click", "type", "submit"] },
      { id: "s2", tokens: ["open", "click", "type", "submit"] },
      { id: "s3", tokens: ["open", "scroll", "click", "submit"] },
    ],
  };

  it("repeats a recorded movement sequence exactly (piece c: repeat)", async () => {
    const model = await new NgramMovementBackend().train(dataset, { order: 3 });
    const generated = model.generate(["open"], { maxLength: 16 });
    expect(["open", ...generated]).toEqual(["open", "click", "type", "submit"]);
  });

  it("predicts the next movement with backoff and stops at END", async () => {
    const model = await new NgramMovementBackend().train(dataset, { order: 2 });
    const ranked = model.predictNext(["open"]);
    expect(ranked[0]?.token).toBe("click");
    // After the full pattern the model should know to stop.
    const end = model.predictNext(["open", "click", "type", "submit"]);
    expect(end[0]?.token).toBe(MOVEMENT_END);
  });

  it("generalizes to a novel-but-related context via backoff (piece d: generalize)", async () => {
    const model = await new NgramMovementBackend().train(dataset, { order: 2 });
    // "type" was never preceded by "scroll" in training; backoff to the
    // unigram/bigram distribution still yields a known continuation.
    const ranked = model.predictNext(["scroll", "type"]);
    expect(ranked.length).toBeGreaterThan(0);
    // Every predicted token is drawn from the learned vocabulary (+END), never START.
    for (const prediction of ranked) {
      expect([...model.vocabulary, MOVEMENT_END]).toContain(prediction.token);
    }
  });

  it("is deterministic across training runs (same dataset -> same predictions)", async () => {
    const a = await new NgramMovementBackend().train(dataset, { order: 2 });
    const b = await new NgramMovementBackend().train(dataset, { order: 2 });
    expect(a.predictNext(["open"])).toEqual(b.predictNext(["open"]));
    expect(a.serialize()).toEqual(b.serialize());
  });

  it("round-trips through serialize/deserialize", async () => {
    const model = await new NgramMovementBackend().train(dataset, { order: 2 });
    const restored = deserializeMovementModel(model.serialize());
    expect(restored.predictNext(["open"])).toEqual(model.predictNext(["open"]));
    expect(restored.generate(["open"])).toEqual(model.generate(["open"]));
  });

  it("respects topK truncation", async () => {
    const model = await new NgramMovementBackend().train(dataset, { order: 2 });
    expect(model.predictNext(["open"], { topK: 1 })).toHaveLength(1);
  });
});

describe("evaluateNextTokenAccuracy + synthetic generalization", () => {
  it("generalizes to a novel sequence built from learned transitions", async () => {
    // Train on two overlapping workflows; every adjacent movement pair below is
    // observed in training, but the *combined* held-out sequence never is.
    const train: MovementDataset = {
      version: 1,
      sequences: [
        { id: "a", tokens: ["open", "click", "type", "submit"] },
        { id: "b", tokens: ["open", "click", "type", "submit"] },
        { id: "c", tokens: ["type", "submit", "close"] },
        { id: "d", tokens: ["type", "submit", "close"] },
      ],
    };
    const model = await new NgramMovementBackend().train(train, { order: 2 });

    // Novel recombination: open->click->type->submit->close (unseen as a whole).
    const holdout = [{ id: "novel", tokens: ["open", "click", "type", "submit", "close"] }];
    const result = evaluateNextTokenAccuracy(model, holdout, { k: 2 });
    expect(result.predictions).toBe(6); // 5 tokens + END
    // Backoff over learned bigrams recovers the novel continuation well above
    // chance (vocab is small but top-2 over ~6 tokens is far from guaranteed).
    expect(result.topKAccuracy).toBeGreaterThan(0.8);
    expect(result.top1Accuracy).toBeGreaterThanOrEqual(0.5);
  });

  it("splits synthetic variants into non-empty train/holdout partitions", () => {
    const sequences = synthesizeMovementSequences({
      patterns: [["open", "click", "type", "submit"], ["focus", "select", "copy", "paste"]],
      variants: 4,
    });
    const { train, holdout } = splitMovementDataset({ version: 1, sequences }, 4);
    expect(train.sequences.length).toBeGreaterThan(0);
    expect(holdout.sequences.length).toBeGreaterThan(0);
    expect(train.sequences.length + holdout.sequences.length).toBe(sequences.length);
  });

  it("reports zeroed accuracy when there is nothing to evaluate", async () => {
    const model = await new NgramMovementBackend().train({ version: 1, sequences: [] });
    const result = evaluateNextTokenAccuracy(model, []);
    expect(result).toMatchObject({ predictions: 0, top1Accuracy: 0, topKAccuracy: 0 });
  });
});

describe("synthesizeMovementSequences", () => {
  it("produces deterministic rotated variants", () => {
    const first = synthesizeMovementSequences({ patterns: [["a", "b", "c"]], variants: 3 });
    const second = synthesizeMovementSequences({ patterns: [["a", "b", "c"]], variants: 3 });
    expect(first).toEqual(second);
    expect(first.map((sequence) => sequence.tokens)).toEqual([
      ["a", "b", "c"],
      ["b", "c", "a"],
      ["c", "a", "b"],
    ]);
  });
});
