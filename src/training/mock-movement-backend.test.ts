import { describe, expect, it } from "vitest";
import { buildMovementDataset, type MovementReplaySource } from "./movement-model.js";
import { MockMarkovMovementBackend } from "./mock-movement-backend.js";
import type { ReplayTimelineEvent } from "../capture/replay.js";

function seq(sessionId: string, tools: string[]): MovementReplaySource {
  return {
    sessionId,
    events: tools.map<ReplayTimelineEvent>((tool, ts) => ({ kind: "action", ts, trajectoryId: sessionId, tool, summary: tool })),
  };
}

async function trainOn(sources: MovementReplaySource[], order: number) {
  const backend = new MockMarkovMovementBackend();
  const dataset = buildMovementDataset(sources);
  const model = await backend.train({ dataset, config: { order } });
  return { backend, model };
}

describe("MockMarkovMovementBackend", () => {
  it("records tokenCount and vocabulary in metadata", async () => {
    const { model } = await trainOn([seq("a", ["x", "y", "z"]), seq("b", ["x", "y"])], 2);
    expect(model.metadata.backend).toBe("mock-markov");
    expect(model.metadata.order).toBe(2);
    expect(model.metadata.sequenceCount).toBe(2);
    expect(model.metadata.tokenCount).toBe(5);
    expect(model.metadata.vocabulary).toEqual(["action:x", "action:y", "action:z"]);
  });

  it("predicts the most frequent continuation with a normalized distribution", async () => {
    // After "x": three sequences go x->y, one goes x->z.
    const { backend, model } = await trainOn(
      [seq("a", ["x", "y"]), seq("b", ["x", "y"]), seq("c", ["x", "y"]), seq("d", ["x", "z"])],
      1,
    );
    const prediction = backend.predictNext(model, ["action:x"]);
    expect(prediction.token).toBe("action:y");
    expect(prediction.probability).toBeCloseTo(0.75);
    expect(prediction.contextOrderUsed).toBe(1);
    const sum = prediction.distribution.reduce((total, entry) => total + entry.probability, 0);
    expect(sum).toBeCloseTo(1);
  });

  it("backs off to shorter context when the full context is unseen", async () => {
    // Order-2 model. Context ["action:p","action:x"] never occurred, but "action:x" -> "action:y" did.
    const { backend, model } = await trainOn([seq("a", ["q", "x", "y"])], 2);
    const prediction = backend.predictNext(model, ["action:p", "action:x"]);
    expect(prediction.token).toBe("action:y");
    expect(prediction.contextOrderUsed).toBe(1);
  });

  it("falls back to the unigram distribution for a wholly unseen context", async () => {
    const { backend, model } = await trainOn([seq("a", ["x", "y", "y"])], 2);
    const prediction = backend.predictNext(model, ["action:unseen"]);
    // Unigram: y appears twice, x once -> y wins.
    expect(prediction.token).toBe("action:y");
    expect(prediction.contextOrderUsed).toBe(0);
  });

  it("returns a null token when the model has no data at all", async () => {
    const { backend, model } = await trainOn([], 2);
    const prediction = backend.predictNext(model, ["action:x"]);
    expect(prediction.token).toBeNull();
    expect(prediction.distribution).toEqual([]);
  });

  it("breaks probability ties deterministically by token order", async () => {
    // "x" is followed once by "b" and once by "a" — equal probability, "a" should win the tie-break.
    const { backend, model } = await trainOn([seq("a", ["x", "b"]), seq("b", ["x", "a"])], 1);
    const prediction = backend.predictNext(model, ["action:x"]);
    expect(prediction.token).toBe("action:a");
  });

  it("produces a JSON-serializable model artifact", async () => {
    const { model } = await trainOn([seq("a", ["x", "y"])], 1);
    const roundTripped = JSON.parse(JSON.stringify(model));
    expect(roundTripped).toEqual(model);
  });
});
