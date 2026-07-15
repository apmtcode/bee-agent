import { describe, expect, it } from "vitest";
import { MarkovMovementBackend } from "./markov-backend.js";
import { evaluateMovementModel, splitMovementDataset } from "./movement-eval.js";
import {
  MOVEMENT_END,
  MOVEMENT_START,
  MovementModelRegistry,
  buildMovementDataset,
  tokenizeReplayEvents,
  type MovementDataset,
} from "./movement-model.js";
import {
  buildSyntheticReplay,
  buildSyntheticReplays,
  sampleDocumentWorkflows,
} from "./synthetic-movements.js";

function datasetFrom(sequences: string[][]): MovementDataset {
  return { version: 1, sequences: sequences.map((tokens) => ({ tokens })) };
}

describe("tokenizeReplayEvents", () => {
  it("reduces actions/observations to target-agnostic verb tokens, dropping transcript by default", () => {
    const replay = buildSyntheticReplay({
      sessionId: "s1",
      trajectoryId: "t1",
      steps: [{ observe: ["os", "opened"], act: ["device", "tapped"] }],
    });
    const withTranscript = [
      ...replay.events,
      { kind: "transcript" as const, ts: 99, messageId: "m", role: "user" as const, content: "hi" },
    ];
    expect(tokenizeReplayEvents(withTranscript)).toEqual(["obs:os:opened", "act:device:tapped"]);
    expect(tokenizeReplayEvents(withTranscript, { includeTranscript: true })).toContain("msg:user");
  });
});

describe("MarkovMovementBackend", () => {
  const backend = new MarkovMovementBackend({ order: 2 });

  it("repeats a single recorded movement trajectory exactly (objective 2c)", () => {
    const dataset = datasetFrom([["a", "b", "c", "d"]]);
    const model = backend.train(dataset);
    expect(model.generate()).toEqual(["a", "b", "c", "d"]);
    expect(model.vocabulary).toEqual(["a", "b", "c", "d"]);
  });

  it("generalizes by stitching transitions across trajectories into a novel sequence (objective 2d)", () => {
    // "a b c" and "x b y" were each recorded; "x b c" was never seen verbatim.
    const model = new MarkovMovementBackend({ order: 1 }).train(datasetFrom([
      ["a", "b", "c"],
      ["x", "b", "y"],
    ]));
    expect(model.generate({ seed: ["x"] })).toEqual(["x", "b", "c"]);
    const afterB = model.predictNext(["b"]);
    expect(afterB.distribution.map((c) => c.token).sort()).toEqual(["c", "y"]);
  });

  it("backs off to a shorter context when the full context is unseen", () => {
    const model = backend.train(datasetFrom([["a", "b", "c"], ["z", "b", "c"]]));
    // Context ["q","b"] (order 2) was never seen; backs off to ["b"] -> c.
    const prediction = model.predictNext(["q", "b"]);
    expect(prediction.token).toBe("c");
    expect(prediction.matchedOrder).toBe(1);
  });

  it("returns an empty prediction from an empty, untrained model", () => {
    const model = backend.train(datasetFrom([]));
    const prediction = model.predictNext(["anything"]);
    expect(prediction.token).toBeUndefined();
    expect(prediction.distribution).toEqual([]);
  });

  it("emits START/END sentinels only when explicitly requested", () => {
    const model = backend.train(datasetFrom([["a", "b"]]));
    // Generation stops *at* END without emitting it, so only the START seed appears.
    expect(model.generate({ includeSentinels: true })).toEqual([MOVEMENT_START, "a", "b"]);
  });

  it("round-trips through serialization via the registry", () => {
    const registry = new MovementModelRegistry().register(new MarkovMovementBackend());
    const original = registry.get("markov").train(datasetFrom([["a", "b", "c"], ["a", "b", "d"]]));
    const restored = registry.load(JSON.parse(JSON.stringify(original.toJSON())));
    expect(restored.vocabulary).toEqual(original.vocabulary);
    expect(restored.predictNext(["a", "b"]).distribution).toEqual(original.predictNext(["a", "b"]).distribution);
    expect(restored.generate()).toEqual(original.generate());
  });
});

describe("MovementModelRegistry", () => {
  it("is pluggable and reports unknown backends clearly", () => {
    const registry = new MovementModelRegistry();
    expect(registry.has("markov")).toBe(false);
    registry.register(new MarkovMovementBackend());
    expect(registry.list()).toEqual(["markov"]);
    expect(() => registry.get("real-mlx")).toThrow(/Unknown movement-model backend/);
  });
});

describe("synthetic loop + generalization eval", () => {
  it("trains on real capture-shaped replays and predicts a held-out related workflow", () => {
    const dataset = buildMovementDataset(buildSyntheticReplays(sampleDocumentWorkflows()));
    expect(dataset.sequences.length).toBe(3);
    // Hold out the third workflow; train on the other two.
    const train = { version: 1 as const, sequences: dataset.sequences.slice(0, 2) };
    const heldOut = dataset.sequences.slice(2);
    const model = new MarkovMovementBackend({ order: 2 }).train(train);

    const result = evaluateMovementModel(model, heldOut);
    expect(result.predictions).toBe(5);
    // The novel middle verb ("formatted") is unpredictable, but the shared
    // open/tap/save scaffolding transfers: the model gets the exact next token
    // right on the scaffolding steps (top-1 0.4) and puts the true next token
    // in-distribution 80% of the time (coverage) — real generalization, not memorization.
    expect(result.top1Accuracy).toBeGreaterThanOrEqual(0.4);
    expect(result.coverage).toBeGreaterThan(0.5);
  });

  it("splits a dataset deterministically for held-out evaluation", () => {
    const dataset = datasetFrom([["a"], ["b"], ["c"], ["d"], ["e"], ["f"]]);
    const { train, heldOut } = splitMovementDataset(dataset, 3);
    expect(train.sequences.map((s) => s.tokens[0])).toEqual(["a", "b", "d", "e"]);
    expect(heldOut.map((s) => s.tokens[0])).toEqual(["c", "f"]);
  });
});
