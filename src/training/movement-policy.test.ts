import { describe, expect, it } from "vitest";
import { buildReplayManifest } from "../capture/replay.js";
import { buildTrajectorySpan, type TrajectoryAction, type TrajectorySpan } from "../capture/trajectory.js";
import {
  buildMovementDatasetFromReplays,
  buildMovementDatasetFromTrajectories,
  deserializeMovementModel,
  evaluateMovementModel,
  MOVEMENT_END_TOKEN,
  NgramMovementBackend,
  tokenizeMovementAction,
  trainMovementModel,
} from "./movement-policy.js";

function action(tool: string, summary: string, ts: number, metadata?: Record<string, unknown>): TrajectoryAction {
  return { kind: "action", tool, summary, ts, ...(metadata ? { metadata } : {}) };
}

function span(id: string, actions: TrajectoryAction[]): TrajectorySpan {
  return buildTrajectorySpan({ id, sessionId: "session-1", actions });
}

/** Synthetic event stream: a repeated "open -> type -> save" movement pattern. */
function syntheticTrajectories(count: number): TrajectorySpan[] {
  return Array.from({ length: count }, (_unused, index) =>
    span(`traj-${index}`, [
      action("device", "opened editor", index * 100 + 1, { gesture: "tap", target: "editor" }),
      action("device", "typed text", index * 100 + 2, { gesture: "type", target: "editor" }),
      action("device", "saved file", index * 100 + 3, { gesture: "shortcut", target: "save" }),
    ]),
  );
}

describe("tokenizeMovementAction", () => {
  it("collapses semantically identical movements to one token via metadata", () => {
    const a = tokenizeMovementAction(action("device", "tapped the editor pane", 1, { gesture: "tap", target: "editor" }));
    const b = tokenizeMovementAction(action("device", "a totally different summary", 2, { gesture: "tap", target: "editor" }));
    expect(a).toBe(b);
    expect(a).toBe("device:tap.editor");
  });

  it("falls back to a slug of the summary when no movement metadata is present", () => {
    expect(tokenizeMovementAction(action("keyboard", "Press Cmd+S", 1))).toBe("keyboard:press-cmd-s");
  });
});

describe("dataset construction", () => {
  it("builds one ordered token sequence per trajectory", () => {
    const dataset = buildMovementDatasetFromTrajectories(syntheticTrajectories(2));
    expect(dataset.sequenceCount).toBe(2);
    expect(dataset.sequences[0]).toEqual(["device:tap.editor", "device:type.editor", "device:shortcut.save"]);
    expect(dataset.vocabulary).toEqual(["device:shortcut.save", "device:tap.editor", "device:type.editor"]);
    expect(dataset.tokenCount).toBe(6);
  });

  it("sorts actions by timestamp regardless of input order", () => {
    const dataset = buildMovementDatasetFromTrajectories([
      span("traj-x", [action("device", "third", 30, { gesture: "tap", target: "c" }), action("device", "first", 10, { gesture: "tap", target: "a" }), action("device", "second", 20, { gesture: "tap", target: "b" })]),
    ]);
    expect(dataset.sequences[0]).toEqual(["device:tap.a", "device:tap.b", "device:tap.c"]);
  });

  it("derives an ordered, trainable dataset from replay manifests", () => {
    // Replay manifests carry only tool + summary (no movement metadata), so the
    // replay path tokenizes from the summary slug. It must still yield one
    // ordered sequence per trajectory that round-trips through training.
    const trajectories = syntheticTrajectories(2);
    const replay = buildReplayManifest({ sessionId: "session-1", transcript: [], trajectories });
    const fromReplay = buildMovementDatasetFromReplays([replay]);
    expect(fromReplay.sequenceCount).toBe(2);
    expect(fromReplay.sequences[0]).toEqual(["device:opened-editor", "device:typed-text", "device:saved-file"]);
    const model = trainMovementModel(fromReplay, { order: 2 });
    expect(model.generate(["device:opened-editor"], 10)).toEqual(["device:typed-text", "device:saved-file"]);
  });
});

describe("NgramMovementBackend training + inference", () => {
  it("repeats a recorded movement pattern exactly from a seed", () => {
    const dataset = buildMovementDatasetFromTrajectories(syntheticTrajectories(5));
    const model = trainMovementModel(dataset, { order: 2 });
    const rollout = model.generate(["device:tap.editor"], 10);
    expect(rollout).toEqual(["device:type.editor", "device:shortcut.save"]);
  });

  it("predicts the next movement deterministically", () => {
    const dataset = buildMovementDatasetFromTrajectories(syntheticTrajectories(3));
    const model = trainMovementModel(dataset, { order: 2 });
    const prediction = model.predictNext(["device:tap.editor"]);
    expect(prediction.token).toBe("device:type.editor");
    expect(prediction.probability).toBeGreaterThan(0.5);
  });

  it("terminates rollouts at the end sentinel", () => {
    const dataset = buildMovementDatasetFromTrajectories(syntheticTrajectories(3));
    const model = trainMovementModel(dataset, { order: 2 });
    const prediction = model.predictNext(["device:shortcut.save"]);
    expect(prediction.token).toBe(MOVEMENT_END_TOKEN);
  });

  it("generalizes to a new but related sequence via backoff", () => {
    // Two patterns share the "type -> save" tail. A context never seen at full
    // order must back off and still predict the shared continuation.
    const dataset = buildMovementDatasetFromTrajectories([
      span("a", [action("device", "open", 1, { gesture: "tap", target: "editor" }), action("device", "type", 2, { gesture: "type", target: "editor" }), action("device", "save", 3, { gesture: "shortcut", target: "save" })]),
      span("b", [action("device", "focus", 1, { gesture: "tap", target: "terminal" }), action("device", "type", 2, { gesture: "type", target: "editor" }), action("device", "save", 3, { gesture: "shortcut", target: "save" })]),
    ]);
    const model = trainMovementModel(dataset, { order: 3 });
    // Unseen full-order context; backoff on the shared "type" token should still
    // recover the "save" continuation.
    const prediction = model.predictNext(["device:tap.unseen", "device:type.editor"]);
    expect(prediction.token).toBe("device:shortcut.save");
    expect(prediction.backoffOrder).toBeLessThan(2);
  });
});

describe("serialization round-trip", () => {
  it("reproduces predictions after serialize/deserialize", () => {
    const dataset = buildMovementDatasetFromTrajectories(syntheticTrajectories(4));
    const model = trainMovementModel(dataset, { order: 2 });
    const snapshot = model.serialize();
    const restored = deserializeMovementModel(JSON.parse(JSON.stringify(snapshot)));
    expect(restored.generate(["device:tap.editor"], 10)).toEqual(model.generate(["device:tap.editor"], 10));
    expect(restored.predictNext(["device:tap.editor"]).token).toBe(model.predictNext(["device:tap.editor"]).token);
  });
});

describe("evaluateMovementModel", () => {
  it("scores perfect fidelity when held-out matches the trained pattern", () => {
    const dataset = buildMovementDatasetFromTrajectories(syntheticTrajectories(6));
    const model = new NgramMovementBackend().train(dataset, { order: 2 });
    const heldOut = buildMovementDatasetFromTrajectories(syntheticTrajectories(2)).sequences;
    const result = evaluateMovementModel(model, heldOut, { seedLength: 1 });
    expect(result.sequenceCount).toBe(2);
    expect(result.nextTokenAccuracy).toBe(1);
    expect(result.rolloutFidelity).toBe(1);
    expect(result.exactRolloutRate).toBe(1);
  });

  it("returns a zeroed result for an empty held-out set", () => {
    const model = trainMovementModel(buildMovementDatasetFromTrajectories(syntheticTrajectories(1)));
    expect(evaluateMovementModel(model, [])).toEqual({
      sequenceCount: 0,
      nextTokenAccuracy: 0,
      rolloutFidelity: 0,
      exactRolloutRate: 0,
    });
  });
});
