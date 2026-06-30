import { describe, expect, it } from "vitest";
import {
  DeterministicMovementBackend,
  MOVEMENT_END_TOKEN,
  buildMovementDataset,
  canonicalMovementToken,
  evaluateMovementModel,
  generateSyntheticMovementSequences,
  replayManifestToMovementSequences,
  trajectoryToMovementSequence,
  type MovementSequence,
} from "./movement-model.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import type { ReplayManifest } from "../capture/replay.js";

function seq(id: string, tokens: string[]): MovementSequence {
  return { id, tokens };
}

describe("canonicalMovementToken", () => {
  it("normalizes tool + summary so repeats collapse to one token", () => {
    expect(canonicalMovementToken({ tool: "Key", summary: "  Cmd+S  press " })).toBe(
      "key::cmd+s press",
    );
    expect(canonicalMovementToken({ tool: "wait", summary: "" })).toBe("wait");
  });
});

describe("tokenization from trajectories and replays", () => {
  it("orders trajectory actions by timestamp", () => {
    const trajectory = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [
        { kind: "action", tool: "b", summary: "second", ts: 20 },
        { kind: "action", tool: "a", summary: "first", ts: 10 },
      ],
    });
    expect(trajectoryToMovementSequence(trajectory).tokens).toEqual(["a::first", "b::second"]);
  });

  it("groups replay action events by trajectory in time order", () => {
    const manifest: ReplayManifest = {
      version: 1,
      sessionId: "s1",
      trajectoryIds: ["t1", "t2"],
      eventCount: 3,
      events: [
        { kind: "action", ts: 30, trajectoryId: "t2", tool: "x", summary: "go" },
        { kind: "action", ts: 20, trajectoryId: "t1", tool: "b", summary: "two" },
        { kind: "action", ts: 10, trajectoryId: "t1", tool: "a", summary: "one" },
      ],
    };
    expect(replayManifestToMovementSequences(manifest)).toEqual([
      { id: "t1", tokens: ["a::one", "b::two"] },
      { id: "t2", tokens: ["x::go"] },
    ]);
  });

  it("drops trajectories with no actions from the dataset", () => {
    const empty = buildTrajectorySpan({ id: "empty", sessionId: "s1", actions: [] });
    const full = buildTrajectorySpan({
      id: "full",
      sessionId: "s1",
      actions: [{ kind: "action", tool: "a", summary: "x", ts: 1 }],
    });
    expect(buildMovementDataset([empty, full]).map((s) => s.id)).toEqual(["full"]);
  });
});

describe("DeterministicMovementBackend", () => {
  const backend = new DeterministicMovementBackend();

  it("repeats a recorded movement sequence exactly when generating from scratch", () => {
    const dataset = [seq("a", ["focus::editor", "key::cmd+s", "observe::saved"])];
    const model = backend.train(dataset, { order: 2 });
    expect(model.generate([], 10)).toEqual(["focus::editor", "key::cmd+s", "observe::saved"]);
  });

  it("predicts the next recorded movement from a known prefix", () => {
    const dataset = [seq("a", ["focus::editor", "key::cmd+s", "observe::saved"])];
    const model = backend.train(dataset, { order: 2 });
    const prediction = model.predictNext(["focus::editor"]);
    expect(prediction?.token).toBe("key::cmd+s");
    expect(prediction?.probability).toBeGreaterThan(0);
  });

  it("favors the more frequent continuation when a context branches", () => {
    const dataset = [
      seq("a", ["open", "save"]),
      seq("b", ["open", "save"]),
      seq("c", ["open", "close"]),
    ];
    const model = backend.train(dataset, { order: 1 });
    expect(model.predictNext(["open"])?.token).toBe("save");
  });

  it("is deterministic: identical datasets yield identical models and rollouts", () => {
    const dataset = [seq("a", ["one", "two", "three"]), seq("b", ["one", "two", "four"])];
    const a = backend.train(dataset, { order: 2 });
    const b = backend.train(dataset, { order: 2 });
    expect(a.toJSON()).toEqual(b.toJSON());
    expect(a.generate([], 10)).toEqual(b.generate([], 10));
  });

  it("generalizes to an unseen prefix via back-off to a shorter context", () => {
    // No sequence starts with "warmup", but every "key::cmd+s" is followed by
    // "observe::saved", so the model should still predict it after a novel lead-in.
    const dataset = [
      seq("a", ["focus::editor", "key::cmd+s", "observe::saved"]),
      seq("b", ["focus::terminal", "key::cmd+s", "observe::saved"]),
    ];
    const model = backend.train(dataset, { order: 2 });
    const prediction = model.predictNext(["warmup", "key::cmd+s"]);
    expect(prediction?.token).toBe("observe::saved");
    expect(prediction?.contextOrder).toBe(1); // backed off from 2 to 1
  });

  it("round-trips through JSON serialization", () => {
    const dataset = [seq("a", ["one", "two", "three"])];
    const model = backend.train(dataset, { order: 2 });
    const restored = backend.load(model.toJSON());
    expect(restored.toJSON()).toEqual(model.toJSON());
    expect(restored.generate([], 10)).toEqual(model.generate([], 10));
  });

  it("returns undefined predictions for an untrained model", () => {
    const model = backend.train([], { order: 2 });
    expect(model.predictNext(["anything"])).toBeUndefined();
    expect(model.generate([], 5)).toEqual([]);
  });

  it("rejects negative smoothing", () => {
    expect(() => backend.train([seq("a", ["x"])], { order: 1, smoothing: -1 })).toThrow();
  });
});

describe("evaluateMovementModel", () => {
  const backend = new DeterministicMovementBackend();

  it("scores perfect accuracy when held-out equals training data", () => {
    const dataset = [seq("a", ["focus::editor", "key::cmd+s", "observe::saved"])];
    const model = backend.train(dataset, { order: 2 });
    const result = evaluateMovementModel(model, dataset);
    expect(result.accuracy).toBe(1);
    expect(result.correct).toBe(result.predictions);
    // Smoothing dilutes the argmax probability, so perplexity stays above 1 even
    // with perfect top-1 accuracy; it should still be far below the vocab size.
    expect(Number.isFinite(result.perplexity)).toBe(true);
    expect(result.perplexity).toBeLessThan(model.vocabulary.length);
  });

  it("counts the END token, so a model that never stops is penalized", () => {
    const model = backend.train([seq("a", ["x", "x", "x", "x"])], { order: 1 });
    // Held-out is shorter; the model over-predicts "x" instead of END at the end.
    const result = evaluateMovementModel(model, [seq("h", ["x"])]);
    expect(result.predictions).toBe(2); // one real token + END
    expect(result.accuracy).toBeLessThan(1);
  });

  it("generalizes: held-out related sequences beat chance", () => {
    const train = generateSyntheticMovementSequences({ seed: 7, count: 40 });
    const held = generateSyntheticMovementSequences({ seed: 99, count: 20 });
    const model = backend.train(train, { order: 3 });
    const result = evaluateMovementModel(model, held);
    expect(result.accuracy).toBeGreaterThan(0.5);
  });
});

describe("generateSyntheticMovementSequences", () => {
  it("is deterministic for a given seed and varies across seeds", () => {
    const a = generateSyntheticMovementSequences({ seed: 1, count: 5 });
    const b = generateSyntheticMovementSequences({ seed: 1, count: 5 });
    const c = generateSyntheticMovementSequences({ seed: 2, count: 5 });
    expect(a).toEqual(b);
    expect(a).not.toEqual(c);
  });

  it("produces the requested count of non-empty sequences", () => {
    const sequences = generateSyntheticMovementSequences({ seed: 3, count: 12 });
    expect(sequences).toHaveLength(12);
    expect(sequences.every((s) => s.tokens.length > 0)).toBe(true);
  });

  it("honors a custom workflow grammar", () => {
    const sequences = generateSyntheticMovementSequences({
      seed: 5,
      count: 3,
      workflow: [{ variants: ["only::step"] }],
    });
    expect(sequences.every((s) => s.tokens.length === 1 && s.tokens[0] === "only::step")).toBe(true);
  });
});
