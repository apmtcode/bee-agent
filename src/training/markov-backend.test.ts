import { describe, expect, it } from "vitest";
import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import {
  buildMovementDataset,
  extractMovementSequence,
  movementActionToken,
  type MovementDataset,
} from "./model-backend.js";
import { MARKOV_BACKEND_ID, MarkovMovementBackend } from "./markov-backend.js";

function action(tool: string, summary: string, ts: number): ReplayTimelineEvent {
  return { kind: "action", ts, trajectoryId: "t1", tool, summary };
}

function replayFrom(events: ReplayTimelineEvent[], trajectoryId = "t1"): ReplayManifest {
  return {
    version: 1,
    sessionId: "s1",
    trajectoryIds: [trajectoryId],
    eventCount: events.length,
    events,
  };
}

function datasetFromSequences(sequences: string[][]): MovementDataset {
  return {
    samples: sequences.map((tokens, index) => ({ trajectoryId: `t${index}`, tokens })),
    vocabulary: [...new Set(sequences.flat())].sort(),
  };
}

describe("movement dataset extraction", () => {
  it("extracts only action events as a stable token sequence", () => {
    const replay = replayFrom([
      { kind: "transcript", ts: 1, messageId: "m1", role: "user", content: "go" },
      action("device", "tapped Login", 2),
      { kind: "observation", ts: 3, trajectoryId: "t1", source: "device", summary: "screen" },
      action("device", "typed into Email", 4),
    ]);
    expect(extractMovementSequence(replay)).toEqual(["device:tapped login", "device:typed into email"]);
  });

  it("normalizes summary whitespace and case when tokenizing", () => {
    const token = movementActionToken({ kind: "action", ts: 1, trajectoryId: "t1", tool: "device", summary: "  Swiped   Up  " });
    expect(token).toBe("device:swiped up");
  });

  it("builds a dataset with a sorted vocabulary and drops empty sequences", () => {
    const withMoves = replayFrom([action("device", "b move", 1), action("device", "a move", 2)]);
    const noMoves = replayFrom([{ kind: "observation", ts: 1, trajectoryId: "t2", source: "device", summary: "idle" }], "t2");
    const dataset = buildMovementDataset([withMoves, noMoves]);
    expect(dataset.samples).toHaveLength(1);
    expect(dataset.vocabulary).toEqual(["device:a move", "device:b move"]);
  });
});

describe("MarkovMovementBackend — repeat recorded movements (objective 2c)", () => {
  it("reproduces a recorded movement sequence exactly from its seed", async () => {
    const sequence = ["open:app", "click:menu", "click:file", "click:save", "type:done"];
    const model = await new MarkovMovementBackend().train(datasetFromSequences([sequence]));
    const generated = model.generate([sequence[0]!], sequence.length - 1);
    expect(generated).toEqual(sequence.slice(1));
  });

  it("predicts the recorded continuation with full confidence at the highest matching order", async () => {
    const sequence = ["a", "b", "c", "d"];
    const model = await new MarkovMovementBackend(3).train(datasetFromSequences([sequence]));
    const prediction = model.predict(["a", "b", "c"]);
    expect(prediction.token).toBe("d");
    expect(prediction.confidence).toBe(1);
    expect(prediction.matchedOrder).toBe(3);
  });
});

describe("MarkovMovementBackend — generalize to related movements (objective 2d)", () => {
  it("backs off to a shorter context for an unseen-but-related prefix", async () => {
    // Two related flows that both end "type -> save". The high-order prefix
    // "scroll type" was never recorded, so the model must generalize via the
    // lower-order "type -> save" transition it did learn.
    const dataset = datasetFromSequences([
      ["open", "type", "save"],
      ["click", "type", "save"],
    ]);
    const model = await new MarkovMovementBackend(3).train(dataset);
    const prediction = model.predict(["scroll", "type"]);
    expect(prediction.token).toBe("save");
    expect(prediction.matchedOrder).toBe(1);
  });

  it("falls back to the unconditional prior for a wholly unseen context", async () => {
    const model = await new MarkovMovementBackend(2).train(datasetFromSequences([["x", "y"], ["x", "y"], ["x", "z"]]));
    const prediction = model.predict(["unseen-token"]);
    // Order-0 prior is the marginal token frequency: x=3, y=2, z=1 (total 6) ->
    // "x" is the most common movement overall and wins deterministically.
    expect(prediction.token).toBe("x");
    expect(prediction.matchedOrder).toBe(0);
    expect(prediction.candidates[0]).toEqual({ token: "x", probability: 3 / 6 });
  });

  it("returns a null prediction when trained on no movements", async () => {
    const model = await new MarkovMovementBackend().train({ samples: [], vocabulary: [] });
    expect(model.predict(["anything"])).toEqual({ token: null, confidence: 0, matchedOrder: 0, candidates: [] });
    expect(model.generate(["anything"], 5)).toEqual([]);
  });
});

describe("MarkovMovementBackend — determinism and persistence", () => {
  it("produces identical serialized weights for the same dataset", async () => {
    const dataset = datasetFromSequences([["a", "b", "c"], ["a", "b", "d"]]);
    const backend = new MarkovMovementBackend();
    const first = (await backend.train(dataset)).serialize();
    const second = (await backend.train(dataset)).serialize();
    expect(second).toEqual(first);
    expect(first.backendId).toBe(MARKOV_BACKEND_ID);
  });

  it("breaks prediction ties by ascending token so results are stable", async () => {
    // After context "a": "m" and "z" each occur once -> "m" wins by token order.
    const model = await new MarkovMovementBackend(1).train(datasetFromSequences([["a", "z"], ["a", "m"]]));
    const prediction = model.predict(["a"]);
    expect(prediction.token).toBe("m");
    expect(prediction.candidates.map((c) => c.token)).toEqual(["m", "z"]);
  });

  it("round-trips through serialize/restore with identical predictions", async () => {
    const dataset = datasetFromSequences([["open", "type", "save"], ["click", "type", "save"]]);
    const backend = new MarkovMovementBackend(3);
    const model = await backend.train(dataset);
    const restored = backend.restore(model.serialize());
    for (const context of [["open", "type"], ["type"], ["scroll", "type"], ["unseen"]]) {
      expect(restored.predict(context)).toEqual(model.predict(context));
    }
    expect(restored.serialize()).toEqual(model.serialize());
  });

  it("rejects restoring a model from a different backend", () => {
    expect(() => new MarkovMovementBackend().restore({ backendId: "other", version: 1 })).toThrow(/cannot restore/);
  });
});
