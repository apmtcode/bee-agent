import { describe, expect, it } from "vitest";
import {
  MOVEMENT_END_TOKEN,
  MarkovMovementBackend,
  TOKEN_FIELD_SEPARATOR,
  buildMovementDataset,
  defaultTokenize,
  evaluateNextTokenAccuracy,
  evaluateReplayFidelity,
  type MovementSequence,
  type ReplaySource,
} from "./movement-model.js";

const tok = (tool: string, summary = "") => defaultTokenize({ tool, summary });

function actionEvent(trajectoryId: string, ts: number, tool: string, summary: string) {
  return { kind: "action", ts, trajectoryId, tool, summary };
}

function replay(trajectoryId: string, actions: Array<[number, string, string]>): ReplaySource {
  return {
    sessionId: `session-${trajectoryId}`,
    trajectoryIds: [trajectoryId],
    events: actions.map(([ts, tool, summary]) => actionEvent(trajectoryId, ts, tool, summary)),
  };
}

describe("buildMovementDataset", () => {
  it("extracts action sequences grouped by trajectory, sorted by ts", () => {
    const dataset = buildMovementDataset([
      replay("t1", [
        [30, "mouse.move", "(10,10)"],
        [10, "mouse.move", "(0,0)"],
        [20, "mouse.click", "left"],
      ]),
    ]);

    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]!.tokens).toEqual([
      tok("mouse.move", "(0,0)"),
      tok("mouse.click", "left"),
      tok("mouse.move", "(10,10)"),
    ]);
  });

  it("ignores non-action events and builds a sorted vocabulary", () => {
    const dataset = buildMovementDataset([
      {
        sessionId: "s1",
        trajectoryIds: ["t1"],
        events: [
          { kind: "transcript", ts: 1 },
          { kind: "observation", ts: 2 },
          actionEvent("t1", 3, "key.press", "a"),
          actionEvent("t1", 4, "key.press", "b"),
        ],
      },
    ]);

    expect(dataset.sequences[0]!.tokens).toEqual([tok("key.press", "a"), tok("key.press", "b")]);
    expect(dataset.vocabulary).toEqual([tok("key.press", "a"), tok("key.press", "b")].sort());
  });

  it("supports a custom tokenizer and session grouping", () => {
    const dataset = buildMovementDataset(
      [
        {
          sessionId: "s1",
          trajectoryIds: ["t1", "t2"],
          events: [actionEvent("t1", 1, "mouse.move", "(3,7)"), actionEvent("t2", 2, "mouse.move", "(9,9)")],
        },
      ],
      { groupBy: "session", tokenize: (a) => a.tool },
    );

    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]!.id).toBe("s1");
    expect(dataset.sequences[0]!.tokens).toEqual(["mouse.move", "mouse.move"]);
  });
});

describe("MarkovMovementBackend training + replay", () => {
  const backend = new MarkovMovementBackend();

  it("repeats a recorded movement exactly (high fidelity)", () => {
    const dataset = buildMovementDataset([
      replay("t1", [
        [1, "open", "app"],
        [2, "type", "hello"],
        [3, "save", "file"],
        [4, "close", "app"],
      ]),
    ]);
    const model = backend.train(dataset, { order: 2 });

    const generated = model.generate([tok("open", "app")], 10);
    expect(generated).toEqual([tok("type", "hello"), tok("save", "file"), tok("close", "app")]);

    const fidelity = evaluateReplayFidelity(model, dataset.sequences);
    expect(fidelity.fidelity).toBe(1);
    expect(fidelity.exactMatches).toBe(1);
  });

  it("predicts every conditioned next token correctly on its training data", () => {
    const dataset = buildMovementDataset([
      replay("t1", [
        [1, "a", ""],
        [2, "b", ""],
        [3, "c", ""],
      ]),
      replay("t2", [
        [1, "a", ""],
        [2, "b", ""],
        [3, "c", ""],
      ]),
    ]);
    const model = backend.train(dataset, { order: 2 });
    const accuracy = evaluateNextTokenAccuracy(model, dataset.sequences);
    // Only the from-nothing first token of each sequence (empty context) is an
    // unconditioned global-mode guess; every context-conditioned token is exact.
    expect(accuracy.correct).toBe(accuracy.total - dataset.sequences.length);
  });

  it("terminates generation at the learned end of sequence", () => {
    const dataset = buildMovementDataset([replay("t1", [[1, "a", ""], [2, "b", ""]])]);
    const model = backend.train(dataset, { order: 2 });
    // maxSteps is generous; generation must still stop at END.
    expect(model.generate([tok("a")], 50)).toEqual([tok("b")]);
  });

  it("generalizes to an unseen-but-related prefix via backoff", () => {
    // Two trajectories share the sub-movement b -> c. A novel prefix "x, b"
    // was never recorded, but the model should still predict c by backing off
    // from the (unseen) bigram context to the learned unigram context "b".
    const dataset = buildMovementDataset([
      replay("t1", [[1, "a", ""], [2, "b", ""], [3, "c", ""]]),
      replay("t2", [[1, "d", ""], [2, "b", ""], [3, "c", ""]]),
    ]);
    const model = backend.train(dataset, { order: 2 });

    const prediction = model.predictNext(["x", tok("b")]);
    expect(prediction?.token).toBe(tok("c"));
    // Backed off to a length-1 context ("b"), not the full length-2 context.
    expect(prediction?.order).toBe(1);
  });

  it("predicts the most frequent continuation and breaks ties deterministically", () => {
    // After "a": c appears twice, b once -> c wins.
    const dataset = buildMovementDataset([
      replay("t1", [[1, "a", ""], [2, "c", ""]]),
      replay("t2", [[1, "a", ""], [2, "c", ""]]),
      replay("t3", [[1, "a", ""], [2, "b", ""]]),
    ]);
    const model = backend.train(dataset, { order: 1 });

    const afterA = model.predictNext([tok("a")]);
    expect(afterA?.token).toBe(tok("c"));
    expect(afterA!.probability).toBeCloseTo(2 / 3);
  });

  it("returns undefined for an empty model", () => {
    const model = backend.train({ version: 1, sequences: [], vocabulary: [] });
    expect(model.predictNext(["anything"])).toBeUndefined();
    expect(model.generate(["anything"], 5)).toEqual([]);
  });

  it("prunes rare transitions when minCount is set", () => {
    const dataset = buildMovementDataset([
      replay("t1", [[1, "a", ""], [2, "b", ""]]),
      replay("t2", [[1, "a", ""], [2, "b", ""]]),
      replay("t3", [[1, "a", ""], [2, "z", ""]]),
    ]);
    const model = backend.train(dataset, { order: 1, minCount: 2 });
    // "z" was seen once after "a"; pruned. "b" (seen twice) remains.
    expect(model.predictNext([tok("a")])?.token).toBe(tok("b"));
  });
});

describe("serialization round-trip", () => {
  it("restores an identical model from its serialized form", () => {
    const backend = new MarkovMovementBackend();
    const dataset = buildMovementDataset([
      replay("t1", [[1, "open", "x"], [2, "type", "y"], [3, "close", "x"]]),
    ]);
    const model = backend.train(dataset, { order: 2 });
    const restored = MarkovMovementBackend.fromSerialized(model.serialize());

    expect(restored.order).toBe(model.order);
    expect(restored.vocabulary).toEqual(model.vocabulary);
    expect(restored.generate([tok("open", "x")], 10)).toEqual(model.generate([tok("open", "x")], 10));
  });
});

describe("evaluateReplayFidelity on held-out sequences", () => {
  it("measures generalization to related but unseen trajectories", () => {
    const backend = new MarkovMovementBackend();
    // Train on a canonical workflow repeated a few times.
    const train = buildMovementDataset([
      replay("t1", [[1, "launch", ""], [2, "navigate", ""], [3, "submit", ""]]),
      replay("t2", [[1, "launch", ""], [2, "navigate", ""], [3, "submit", ""]]),
    ]);
    const model = backend.train(train, { order: 2 });

    // Held-out sequence starts identically -> should replay the learned tail.
    const heldOut: MovementSequence[] = [
      { id: "h1", tokens: [tok("launch"), tok("navigate"), tok("submit")] },
    ];
    const report = evaluateReplayFidelity(model, heldOut);
    expect(report.fidelity).toBe(1);
    expect(report.meanTokenOverlap).toBe(1);
  });
});

describe("defaultTokenize / END sentinel", () => {
  it("is stable and distinct per tool+summary", () => {
    expect(defaultTokenize({ tool: "mouse.move", summary: "(1,2)" })).toBe(
      `mouse.move${TOKEN_FIELD_SEPARATOR}(1,2)`,
    );
    expect(defaultTokenize({ tool: "mouse.move", summary: "(1,2)" })).not.toBe(
      defaultTokenize({ tool: "mouse.move", summary: "(3,4)" }),
    );
  });

  it("avoids tool/summary boundary collisions via the field separator", () => {
    expect(defaultTokenize({ tool: "a", summary: "bc" })).not.toBe(
      defaultTokenize({ tool: "ab", summary: "c" }),
    );
  });

  it("never emits the END sentinel as a produced token", () => {
    const backend = new MarkovMovementBackend();
    const dataset = buildMovementDataset([replay("t1", [[1, "a", ""], [2, "b", ""]])]);
    const model = backend.train(dataset, { order: 2 });
    expect(model.generate([tok("a")], 20)).not.toContain(MOVEMENT_END_TOKEN);
  });
});
