import { describe, expect, it } from "vitest";
import { buildTrajectorySpan, type TrajectorySpan } from "../capture/trajectory.js";
import {
  MovementLearner,
  NGramMovementBackend,
  buildMovementSequences,
  buildMovementSequencesFromReplay,
  evaluateMovementGeneralization,
  movementClassKey,
  movementTokenFromAction,
  movementTokenKey,
  type MovementModelBackend,
  type MovementSequence,
} from "./movement-model.js";
import { buildReplayManifest } from "../capture/replay.js";

let clock = 0;
function nextTs(): number {
  clock += 1000;
  return clock;
}

function action(tool: string, metadata: Record<string, unknown>, summary = tool): TrajectorySpan["actions"][number] {
  return { kind: "action", tool, summary, ts: nextTs(), metadata };
}

function gestureAction(gesture: string, target: string, direction?: string) {
  return action("device", { gesture, target, ...(direction ? { direction } : {}) }, `${gesture} ${target}`);
}

/** A synthetic "open note and type" episode, parameterized by the note target. */
function openNoteSequence(id: string, note: string): TrajectorySpan {
  return buildTrajectorySpan({
    id,
    sessionId: `session-${id}`,
    actions: [
      gestureAction("tap", "notes-app-icon"),
      gestureAction("tap", `note-${note}`),
      gestureAction("type", "note-body"),
      gestureAction("shortcut", "save"),
    ],
    outcome: { status: "success", summary: "note saved" },
  });
}

describe("movement token extraction", () => {
  it("recovers structured tokens from action metadata", () => {
    const token = movementTokenFromAction(gestureAction("swipe", "gallery", "left"));
    expect(token).toEqual({ tool: "device", gesture: "swipe", target: "gallery", direction: "left" });
  });

  it("falls back to the summary when no structured metadata exists", () => {
    const token = movementTokenFromAction({ kind: "action", tool: "shell", summary: "ls -la", ts: 1, metadata: {} });
    expect(token).toEqual({ tool: "shell", target: "ls -la" });
  });

  it("derives distinct exact keys but a shared class key across targets", () => {
    const a = movementTokenFromAction(gestureAction("tap", "note-alpha"));
    const b = movementTokenFromAction(gestureAction("tap", "note-beta"));
    expect(movementTokenKey(a)).not.toBe(movementTokenKey(b));
    expect(movementClassKey(a)).toBe(movementClassKey(b));
  });
});

describe("buildMovementSequences", () => {
  it("orders actions by timestamp and drops empty trajectories", () => {
    const sequences = buildMovementSequences([
      openNoteSequence("t1", "alpha"),
      buildTrajectorySpan({ id: "empty", sessionId: "s", actions: [] }),
    ]);
    expect(sequences).toHaveLength(1);
    expect(sequences[0].tokens.map((t) => t.gesture)).toEqual(["tap", "tap", "type", "shortcut"]);
    expect(sequences[0].context.outcome).toBe("success");
  });

  it("builds sequences from a replay manifest action timeline", () => {
    const trajectory = openNoteSequence("replay-1", "alpha");
    const replay = buildReplayManifest({ sessionId: "sess", transcript: [], trajectories: [trajectory] });
    const sequences = buildMovementSequencesFromReplay(replay);
    expect(sequences).toHaveLength(1);
    expect(sequences[0].tokens).toHaveLength(4);
    expect(sequences[0].tokens.every((t) => t.tool === "device")).toBe(true);
  });
});

describe("NGramMovementBackend — repeat recorded movements", () => {
  it("predicts the exact next movement it was trained on", () => {
    const dataset = buildMovementSequences([
      openNoteSequence("t1", "alpha"),
      openNoteSequence("t2", "alpha"),
    ]);
    const learner = new MovementLearner();
    learner.train(dataset, { order: 3 });

    const predictions = learner.predictNext({ history: [{ tool: "device", gesture: "tap", target: "notes-app-icon" }] });
    expect(predictions[0].token).toEqual({ tool: "device", gesture: "tap", target: "note-alpha" });
    expect(predictions[0].generalized).toBe(false);
  });

  it("predicts the first movement from an empty history via the start boundary", () => {
    const dataset = buildMovementSequences([openNoteSequence("t1", "alpha")]);
    const learner = new MovementLearner();
    learner.train(dataset);
    const [top] = learner.predictNext({ history: [] });
    expect(top.token).toEqual({ tool: "device", gesture: "tap", target: "notes-app-icon" });
  });

  it("autoregressively replays a full recorded movement run", () => {
    const dataset = buildMovementSequences([
      openNoteSequence("t1", "alpha"),
      openNoteSequence("t2", "alpha"),
    ]);
    const learner = new MovementLearner();
    learner.train(dataset, { order: 4 });
    const rollout = learner.rollout({ history: [] }, 4);
    expect(rollout.map((t) => `${t.gesture}:${t.target}`)).toEqual([
      "tap:notes-app-icon",
      "tap:note-alpha",
      "type:note-body",
      "shortcut:save",
    ]);
  });
});

describe("NGramMovementBackend — generalize to new but related movements", () => {
  it("predicts a novel target's movement class via class backoff", () => {
    // Train only on note-alpha / note-beta; hold out note-gamma.
    const dataset = buildMovementSequences([
      openNoteSequence("t1", "alpha"),
      openNoteSequence("t2", "beta"),
    ]);
    const backend = new NGramMovementBackend();
    const snapshot = backend.train(dataset, { order: 3 });

    // History leads with a *never-seen* concrete target, so exact n-gram context
    // misses and the model must generalize through the movement-class layer.
    const predictions = backend.predictNext(snapshot, {
      history: [
        { tool: "device", gesture: "tap", target: "notes-app-icon" },
        { tool: "device", gesture: "tap", target: "note-gamma" },
      ],
    });
    const top = predictions[0];
    // Concrete target unknown, but the *class* (type note-body) is predicted.
    expect(top.token.gesture).toBe("type");
    expect(movementClassKey(top.token)).toBe(movementClassKey({ tool: "device", gesture: "type", target: "note-body" }));
    expect(top.generalized).toBe(true);
  });

  it("scores measurable generalization on held-out related sequences", () => {
    const train = buildMovementSequences([
      openNoteSequence("t1", "alpha"),
      openNoteSequence("t2", "beta"),
      openNoteSequence("t3", "delta"),
    ]);
    const heldOut = buildMovementSequences([openNoteSequence("h1", "gamma")]);
    const backend = new NGramMovementBackend();
    const snapshot = backend.train(train, { order: 3 });

    const report = evaluateMovementGeneralization(backend, snapshot, heldOut);
    expect(report.evaluatedSequences).toBe(1);
    expect(report.evaluatedSteps).toBe(4);
    // Class accuracy should be strong even though the concrete note target is new.
    expect(report.classAccuracy).toBeGreaterThanOrEqual(0.75);
    // At least some correct class predictions came from the generalization path.
    expect(report.generalizationRate).toBeGreaterThan(0);
  });
});

describe("determinism and pluggability", () => {
  it("produces identical snapshots and predictions across runs", () => {
    const dataset = buildMovementSequences([openNoteSequence("t1", "alpha"), openNoteSequence("t2", "beta")]);
    const backend = new NGramMovementBackend();
    const a = backend.train(dataset, { order: 3 });
    const b = backend.train(dataset, { order: 3 });
    expect(JSON.stringify(a)).toBe(JSON.stringify(b));
    const ctx = { history: [{ tool: "device", gesture: "tap", target: "notes-app-icon" }] };
    expect(backend.predictNext(a, ctx)).toEqual(backend.predictNext(b, ctx));
  });

  it("accepts a custom backend through the learner seam", () => {
    const fixed: MovementModelBackend = {
      name: "fixed",
      train: () => ({ backend: "fixed", version: 1, order: 1, trainedSequences: 0, trainedTokens: 0, parameters: {} }),
      predictNext: () => [{ token: { tool: "noop" }, score: 1, backoffOrder: 0, generalized: false }],
    };
    const learner = new MovementLearner(fixed);
    learner.train([]);
    expect(learner.predictNext({ history: [] })[0].token.tool).toBe("noop");
  });

  it("throws if used before training", () => {
    const learner = new MovementLearner();
    expect(() => learner.predictNext({ history: [] })).toThrow(/train\(\)/);
  });
});

describe("snapshot round-trip", () => {
  it("survives JSON serialization for on-device persistence", () => {
    const dataset: MovementSequence[] = buildMovementSequences([openNoteSequence("t1", "alpha")]);
    const backend = new NGramMovementBackend();
    const snapshot = backend.train(dataset, { order: 3 });
    const restored = JSON.parse(JSON.stringify(snapshot));
    const fromLive = backend.predictNext(snapshot, { history: [] });
    const fromRestored = backend.predictNext(restored, { history: [] });
    expect(fromRestored).toEqual(fromLive);
  });
});
