import { describe, expect, it } from "vitest";
import type { ReplayTimelineEvent } from "../capture/replay.js";
import {
  generateSyntheticReplays,
  partitionReplays,
} from "../capture/synthetic-stream.js";
import {
  buildMovementDataset,
  evaluateMovementModel,
  MOVEMENT_END_TOKEN,
  NgramMovementBackend,
  tokenizeMovementEvent,
  type MovementDataset,
} from "./movement-model.js";

function actionEvent(tool: string, ts: number): ReplayTimelineEvent {
  return { kind: "action", ts, trajectoryId: "t", tool, summary: `do ${tool}` };
}

describe("tokenizeMovementEvent", () => {
  it("encodes each event kind with its primary identifier", () => {
    expect(tokenizeMovementEvent(actionEvent("mouse.click", 1))).toBe("action:mouse.click");
    expect(
      tokenizeMovementEvent({ kind: "observation", ts: 1, trajectoryId: "t", source: "window.focus", summary: "" }),
    ).toBe("observation:window.focus");
    expect(
      tokenizeMovementEvent({ kind: "transcript", ts: 1, messageId: "m", role: "user", content: "" }),
    ).toBe("transcript:user");
  });
});

describe("buildMovementDataset", () => {
  it("builds one sequence per replay and appends the end token", () => {
    const dataset = buildMovementDataset([
      { sessionId: "s1", events: [actionEvent("a", 1), actionEvent("b", 2)] },
    ]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0].tokens).toEqual(["action:a", "action:b", MOVEMENT_END_TOKEN]);
  });

  it("can filter to specific event kinds and omit the end token", () => {
    const dataset = buildMovementDataset(
      [
        {
          sessionId: "s1",
          events: [
            { kind: "observation", ts: 1, trajectoryId: "t", source: "win", summary: "" },
            actionEvent("a", 2),
          ],
        },
      ],
      { includeKinds: ["action"], appendEndToken: false },
    );
    expect(dataset.sequences[0].tokens).toEqual(["action:a"]);
  });
});

describe("NgramMovementBackend", () => {
  it("learns to reproduce a recorded movement sequence", async () => {
    const backend = new NgramMovementBackend({ order: 2 });
    const dataset = buildMovementDataset([
      {
        sessionId: "s1",
        events: [actionEvent("open", 1), actionEvent("click", 2), actionEvent("type", 3), actionEvent("save", 4)],
      },
    ]);
    const model = await backend.train(dataset);
    const generated = model.generate(["action:open"]);
    expect(generated).toEqual(["action:click", "action:type", "action:save"]);
  });

  it("predicts the most likely next movement with a probability and candidates", async () => {
    const backend = new NgramMovementBackend({ order: 1 });
    // "click" is followed by "type" twice and "save" once.
    const dataset: MovementDataset = {
      sequences: [
        { tokens: ["action:click", "action:type"], id: "a" },
        { tokens: ["action:click", "action:type"], id: "b" },
        { tokens: ["action:click", "action:save"], id: "c" },
      ],
    };
    const model = await backend.train(dataset);
    const prediction = model.predictNext(["action:click"]);
    expect(prediction.token).toBe("action:type");
    expect(prediction.probability).toBeCloseTo(2 / 3, 5);
    expect(prediction.candidates.map((c) => c.token)).toEqual(["action:type", "action:save"]);
    expect(prediction.backoffOrder).toBe(1);
  });

  it("generalises to unseen contexts via backoff", async () => {
    const backend = new NgramMovementBackend({ order: 2 });
    const dataset = buildMovementDataset([
      { sessionId: "s1", events: [actionEvent("a", 1), actionEvent("b", 2), actionEvent("c", 3)] },
    ]);
    const model = await backend.train(dataset);
    // The bigram context ["action:x","action:b"] was never seen, but backoff to
    // the unigram context "action:b" still predicts "action:c".
    const prediction = model.predictNext(["action:x", "action:b"]);
    expect(prediction.token).toBe("action:c");
    expect(prediction.backoffOrder).toBe(1);
  });

  it("returns an empty prediction for a model with no data", async () => {
    const backend = new NgramMovementBackend();
    const model = await backend.train({ sequences: [] });
    const prediction = model.predictNext(["anything"]);
    expect(prediction.token).toBeUndefined();
    expect(prediction.candidates).toEqual([]);
  });

  it("is deterministic across retrains and tie-breaks by count then token", async () => {
    const backend = new NgramMovementBackend({ order: 1 });
    const dataset: MovementDataset = {
      sequences: [
        { tokens: ["x", "b"], id: "1" },
        { tokens: ["x", "a"], id: "2" },
      ],
    };
    const first = await backend.train(dataset);
    const second = await backend.train(dataset);
    // Equal counts (1 each) -> lexicographic tie-break picks "a".
    expect(first.predictNext(["x"]).token).toBe("a");
    expect(second.predictNext(["x"]).token).toBe("a");
    expect(first.serialize()).toEqual(second.serialize());
  });

  it("round-trips through serialize/restore with identical behaviour", async () => {
    const backend = new NgramMovementBackend({ order: 2 });
    const dataset = buildMovementDataset(
      generateSyntheticReplays({ seed: 7, sequenceCount: 6 }).map((r) => ({ sessionId: r.sessionId, events: r.events })),
    );
    const model = await backend.train(dataset);
    const state = model.serialize();
    const restored = backend.restore(state);
    expect(restored.serialize()).toEqual(state);
    const context = dataset.sequences[0].tokens.slice(0, 2);
    expect(restored.predictNext(context)).toEqual(model.predictNext(context));
    expect(restored.generate(context.slice(0, 1))).toEqual(model.generate(context.slice(0, 1)));
  });

  it("caps generation length via maxSteps even without an end token", async () => {
    const backend = new NgramMovementBackend({ order: 1 });
    // A self-loop with no end token would generate forever without the cap.
    const model = await backend.train({ sequences: [{ tokens: ["a", "a", "a"], id: "loop" }] });
    const generated = model.generate(["a"], { maxSteps: 5, stopAtEndToken: false });
    expect(generated).toHaveLength(5);
    expect(new Set(generated)).toEqual(new Set(["a"]));
  });
});

describe("evaluateMovementModel", () => {
  it("scores perfect accuracy on the training distribution", async () => {
    const backend = new NgramMovementBackend({ order: 2 });
    const dataset = buildMovementDataset([
      { sessionId: "s1", events: [actionEvent("a", 1), actionEvent("b", 2), actionEvent("c", 3)] },
    ]);
    const model = await backend.train(dataset);
    const result = evaluateMovementModel(model, dataset);
    expect(result.accuracy).toBe(1);
    expect(result.predictionCount).toBeGreaterThan(0);
    expect(result.perSequence[0].correct).toBe(result.perSequence[0].predictions);
  });

  it("measures generalisation on held-out but related synthetic movements", async () => {
    const replays = generateSyntheticReplays({ seed: 42, sequenceCount: 30 });
    const { train, heldOut } = partitionReplays(replays, 0.7);
    const backend = new NgramMovementBackend({ order: 2 });
    const model = await backend.train(
      buildMovementDataset(train.map((r) => ({ sessionId: r.sessionId, events: r.events }))),
    );
    const result = evaluateMovementModel(
      model,
      buildMovementDataset(heldOut.map((r) => ({ sessionId: r.sessionId, events: r.events }))),
      { topK: 3 },
    );
    // The templates share structure, so a backoff n-gram should generalise well
    // above chance on unseen sequences.
    expect(result.sequenceCount).toBe(heldOut.length);
    expect(result.accuracy).toBeGreaterThan(0.5);
    expect(result.topKAccuracy).toBeGreaterThanOrEqual(result.accuracy);
  });
});
