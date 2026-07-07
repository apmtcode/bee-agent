import { describe, expect, it } from "vitest";
import {
  MarkovMovementBackend,
  MOVEMENT_END,
  movementSequenceFromReplay,
  movementSequenceFromTrajectory,
  movementSequencesFromReplays,
  movementTokenForAction,
  type MovementModelBackend,
  type MovementSequence,
} from "./movement-model.js";
import { evaluateMovementModel } from "./movement-eval.js";
import {
  DEFAULT_MOVEMENT_TEMPLATES,
  synthesizeMovementSequences,
} from "./movement-synth.js";
import type { ReplayManifest } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

const backend: MovementModelBackend = new MarkovMovementBackend();

describe("MarkovMovementBackend", () => {
  it("exactly replays a memorized movement sequence (round-trip)", () => {
    const sequence: MovementSequence = {
      id: "s1",
      tokens: ["browser#open", "browser#click deploy", "browser#confirm", "browser#observe"],
    };
    const model = backend.train([sequence], { order: 3 });

    // From an empty prefix the model should regenerate the recorded movement.
    expect(model.generate([])).toEqual(sequence.tokens);
  });

  it("predicts the recorded continuation for a seen prefix", () => {
    const model = backend.train(
      [{ id: "s", tokens: ["a", "b", "c", "d"] }],
      { order: 3 },
    );
    expect(model.predictNext(["a"])?.token).toBe("b");
    expect(model.predictNext(["a", "b"])?.token).toBe("c");
    expect(model.predictNext(["b", "c"])?.token).toBe("d");
    // End of a fully-seen sequence predicts the stop token.
    expect(model.predictNext(["b", "c", "d"])?.token).toBe(MOVEMENT_END);
  });

  it("generalizes to a novel prefix via backoff to shorter contexts", () => {
    // Two related sequences: both go X -> deploy after some opener. A novel
    // opener the model never saw before ("open-4") should still route to
    // "deploy" by backing off the unseen high-order context to the shared
    // lower-order pattern.
    const model = backend.train(
      [
        { id: "a", tokens: ["open-1", "select", "deploy", "done"] },
        { id: "b", tokens: ["open-2", "select", "deploy", "done"] },
        { id: "c", tokens: ["open-3", "select", "deploy", "done"] },
      ],
      { order: 3 },
    );
    // "select" -> "deploy" is the dominant transition regardless of opener.
    const prediction = model.predictNext(["open-4", "select"]);
    expect(prediction?.token).toBe("deploy");
    expect(prediction?.matchedContextLength).toBeGreaterThan(0);
  });

  it("is deterministic: identical training yields identical snapshots", () => {
    const seqs: MovementSequence[] = [
      { id: "a", tokens: ["x", "y", "z"] },
      { id: "b", tokens: ["x", "y", "w"] },
    ];
    const a = backend.train(seqs, { order: 2 }).toJSON();
    const b = backend.train(seqs, { order: 2 }).toJSON();
    expect(JSON.stringify(a)).toBe(JSON.stringify(b));
  });

  it("round-trips through serialization (save + load reproduce predictions)", () => {
    const model = backend.train(
      [{ id: "s", tokens: ["a", "b", "c"] }],
      { order: 2 },
    );
    const snapshot = model.toJSON();
    const reloaded = backend.load(JSON.parse(JSON.stringify(snapshot)));
    expect(reloaded.generate([])).toEqual(model.generate([]));
    expect(reloaded.predictNext(["a"])?.token).toBe(model.predictNext(["a"])?.token);
  });

  it("ranks candidates by frequency with deterministic tie-breaking", () => {
    const model = backend.train(
      [
        { id: "1", tokens: ["ctx", "common"] },
        { id: "2", tokens: ["ctx", "common"] },
        { id: "3", tokens: ["ctx", "rare"] },
      ],
      { order: 2 },
    );
    const ranked = model.rank(["ctx"]);
    expect(ranked[0].token).toBe("common");
    expect(ranked.find((entry) => entry.token === "rare")).toBeDefined();
    expect(ranked[0].score).toBeGreaterThan(
      ranked.find((entry) => entry.token === "rare")!.score,
    );
  });

  it("returns undefined for an empty model", () => {
    const model = backend.train([], { order: 2 });
    expect(model.predictNext(["anything"])).toBeUndefined();
    expect(model.generate([])).toEqual([]);
  });
});

describe("movement tokenization", () => {
  it("normalizes volatile ids/numbers so related actions share a token", () => {
    const a = movementTokenForAction({ tool: "browser", summary: "opened tab 42" });
    const b = movementTokenForAction({ tool: "browser", summary: "opened tab 99" });
    expect(a).toBe(b);
    expect(a).toContain("browser#");
  });

  it("derives a structured token from gesture metadata", () => {
    const token = movementTokenForAction({
      tool: "device",
      summary: "swiped up",
      metadata: { gesture: "swipe", direction: "up" },
    });
    expect(token).toBe("device#swipe:up");
  });

  it("builds a sequence from trajectory actions in timestamp order", () => {
    const trajectory: TrajectorySpan = {
      id: "t1",
      sessionId: "sess",
      createdAt: "2026-01-01T00:00:00.000Z",
      captureTier: "operator",
      observations: [],
      actions: [
        { kind: "action", tool: "browser", summary: "second", ts: 20 },
        { kind: "action", tool: "browser", summary: "first", ts: 10 },
      ],
    };
    const sequence = movementSequenceFromTrajectory(trajectory);
    expect(sequence.tokens).toEqual(["browser#first", "browser#second"]);
  });

  it("extracts only action events from a replay manifest", () => {
    const replay: ReplayManifest = {
      version: 1,
      sessionId: "sess",
      trajectoryIds: ["t1"],
      eventCount: 3,
      events: [
        { kind: "observation", ts: 1, trajectoryId: "t1", source: "browser", summary: "saw page" },
        { kind: "action", ts: 2, trajectoryId: "t1", tool: "browser", summary: "clicked deploy" },
        { kind: "action", ts: 3, trajectoryId: "t1", tool: "browser", summary: "confirmed" },
      ],
    };
    const sequence = movementSequenceFromReplay(replay);
    expect(sequence.tokens).toEqual(["browser#clicked deploy", "browser#confirmed"]);
    expect(movementSequencesFromReplays([replay])).toHaveLength(1);
  });
});

describe("movement generalization eval harness", () => {
  it("scores exact replay of a memorized sequence at 100% accuracy", () => {
    // A single trained sequence is unambiguously reproducible at every step,
    // including step 0 (empty context conditions on START padding).
    const [sequence] = synthesizeMovementSequences({
      templates: DEFAULT_MOVEMENT_TEMPLATES,
      perTemplate: 1,
      seed: 7,
    });
    const model = backend.train([sequence], { order: 4 });
    const report = evaluateMovementModel(model, [sequence], { contextWindow: 4 });
    expect(report.accuracy).toBe(1);
  });

  it("generalizes to a held-out split of the same template family", () => {
    // Split synthetic data: train on the first half, evaluate on the rest.
    const all = synthesizeMovementSequences({
      templates: DEFAULT_MOVEMENT_TEMPLATES,
      perTemplate: 8,
      seed: 3,
    });
    const train = all.filter((_, i) => i % 2 === 0);
    const heldOut = all.filter((_, i) => i % 2 === 1);
    const model = backend.train(train, { order: 4 });
    const report = evaluateMovementModel(model, heldOut, { contextWindow: 4 });
    // Related-but-unseen sequences should be mostly predictable via backoff.
    expect(report.accuracy).toBeGreaterThan(0.6);
    expect(report.macroAccuracy).toBeGreaterThan(0.6);
  });
});

describe("synthetic movement generator", () => {
  it("is reproducible for a fixed seed", () => {
    const opts = { templates: DEFAULT_MOVEMENT_TEMPLATES, perTemplate: 5, seed: 42 };
    expect(synthesizeMovementSequences(opts)).toEqual(synthesizeMovementSequences(opts));
  });

  it("varies output across different seeds", () => {
    const a = synthesizeMovementSequences({ templates: DEFAULT_MOVEMENT_TEMPLATES, perTemplate: 5, seed: 1 });
    const b = synthesizeMovementSequences({ templates: DEFAULT_MOVEMENT_TEMPLATES, perTemplate: 5, seed: 2 });
    expect(JSON.stringify(a)).not.toBe(JSON.stringify(b));
  });
});
