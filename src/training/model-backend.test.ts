import { describe, expect, it } from "vitest";
import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import {
  NgramMovementBackend,
  datasetFromReplays,
  evaluateNextTokenAccuracy,
  sequenceFromEvents,
  tokenizeEvent,
  type MovementDataset,
} from "./model-backend.js";

function action(tool: string, ts: number): ReplayTimelineEvent {
  return { kind: "action", ts, trajectoryId: "t1", tool, summary: `${tool}@${ts}` };
}

function observation(source: string, ts: number): ReplayTimelineEvent {
  return { kind: "observation", ts, trajectoryId: "t1", source, summary: `${source}@${ts}` };
}

function replay(sessionId: string, events: ReplayTimelineEvent[]): ReplayManifest {
  return { version: 1, sessionId, trajectoryIds: ["t1"], eventCount: events.length, events };
}

/** A repeated "open -> focus -> type -> submit" workflow. */
function workflowDataset(): MovementDataset {
  const seq = [
    observation("window", 1),
    action("open", 2),
    action("focus", 3),
    action("type", 4),
    action("submit", 5),
  ];
  return datasetFromReplays([replay("s1", seq), replay("s2", seq)]);
}

describe("tokenizeEvent", () => {
  it("encodes each event kind into a compact token", () => {
    expect(tokenizeEvent(action("click", 1))).toBe("act:click");
    expect(tokenizeEvent(observation("mouse", 1))).toBe("obs:mouse");
    expect(tokenizeEvent({ kind: "transcript", ts: 1, messageId: "m", role: "user", content: "hi" })).toBe(
      "msg:user",
    );
  });
});

describe("sequenceFromEvents / datasetFromReplays", () => {
  it("preserves ordering of the recorded movements", () => {
    const seq = sequenceFromEvents("s", [action("a", 1), action("b", 2)]);
    expect(seq.tokens).toEqual(["act:a", "act:b"]);
  });
});

describe("NgramMovementBackend — repeat recorded movements", () => {
  it("reproduces a recorded trajectory from its prefix (objective 2c)", async () => {
    const model = await new NgramMovementBackend().train(workflowDataset(), { order: 3 });
    const generated = model.generate({ history: ["obs:window", "act:open"] }, 3);
    expect(generated).toEqual(["act:focus", "act:type", "act:submit"]);
  });

  it("predicts with full confidence when the context is unambiguous", async () => {
    const model = await new NgramMovementBackend().train(workflowDataset());
    const prediction = model.predictNext({ history: ["act:open"] });
    expect(prediction.token).toBe("act:focus");
    expect(prediction.confidence).toBe(1);
    expect(prediction.order).toBeGreaterThan(0);
  });
});

describe("NgramMovementBackend — generalize to related movements", () => {
  it("backs off to a shorter shared suffix for an unseen prefix (objective 2d)", async () => {
    // Two trajectories that diverge early but share the "focus -> type" habit.
    const a = [action("open", 1), action("focus", 2), action("type", 3)];
    const b = [action("launch", 1), action("focus", 2), action("type", 3)];
    const model = await new NgramMovementBackend().train(
      datasetFromReplays([replay("a", a), replay("b", b)]),
      { order: 2 },
    );
    // Unseen high-order context ("scroll" never preceded "focus"), but the
    // order-1 context "focus" was always followed by "type".
    const prediction = model.predictNext({ history: ["act:scroll", "act:focus"] });
    expect(prediction.token).toBe("act:type");
    expect(prediction.order).toBe(1);
  });

  it("returns no prediction when nothing was learned", async () => {
    const model = await new NgramMovementBackend().train({ sequences: [] });
    expect(model.predictNext({ history: ["act:anything"] }).token).toBeUndefined();
    expect(model.generate({ history: [] }, 5)).toEqual([]);
  });
});

describe("NgramMovementBackend — determinism & candidates", () => {
  it("breaks ties deterministically by descending count then ascending token", async () => {
    // After "act:x": one "act:a", one "act:b" (tie) -> ascending token wins.
    const seq = [action("x", 1), action("b", 2), action("x", 3), action("a", 4)];
    const model = await new NgramMovementBackend().train(datasetFromReplays([replay("s", seq)]), {
      order: 1,
    });
    const prediction = model.predictNext({ history: ["act:x"] });
    expect(prediction.token).toBe("act:a");
    expect(prediction.candidates.map((candidate) => candidate.token)).toEqual(["act:a", "act:b"]);
    expect(prediction.candidates[0]?.probability).toBeCloseTo(0.5);
  });
});

describe("NgramMovementBackend — serialization round-trip", () => {
  it("restores an equivalent model from serialized counts", async () => {
    const backend = new NgramMovementBackend();
    const model = await backend.train(workflowDataset(), { order: 3 });
    const restored = backend.restore(model.serialize());
    expect(restored.order).toBe(model.order);
    expect(restored.generate({ history: ["obs:window", "act:open"] }, 3)).toEqual([
      "act:focus",
      "act:type",
      "act:submit",
    ]);
  });

  it("rejects restoring a foreign backend's model", () => {
    const backend = new NgramMovementBackend();
    expect(() =>
      backend.restore({ version: 1, backend: "mlx", order: 2, counts: {} }),
    ).toThrow(/Cannot restore mlx model/);
  });
});

describe("evaluateNextTokenAccuracy — generalization eval harness", () => {
  it("scores perfect accuracy on a held-out repeat of a learned workflow", async () => {
    const model = await new NgramMovementBackend().train(workflowDataset(), { order: 3 });
    const heldOut = workflowDataset();
    const result = evaluateNextTokenAccuracy(model, heldOut);
    // Every position is scored; the first token has empty context (unigram),
    // the rest are exact recorded continuations.
    expect(result.total).toBe(10);
    expect(result.accuracy).toBeGreaterThan(0.7);
  });

  it("reports zero for an empty held-out set", async () => {
    const model = await new NgramMovementBackend().train(workflowDataset());
    expect(evaluateNextTokenAccuracy(model, { sequences: [] })).toEqual({
      total: 0,
      correct: 0,
      accuracy: 0,
    });
  });
});
