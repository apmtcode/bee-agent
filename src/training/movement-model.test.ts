import { describe, expect, it } from "vitest";
import { buildReplayManifest } from "../capture/replay.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import {
  MovementInferenceEngine,
  NGramMovementBackend,
  canonicalMovementLabel,
  sequenceFromReplay,
  sequenceFromTrajectory,
  sequencesFromTrajectories,
  type MovementSequence,
  type MovementToken,
} from "./movement-model.js";

function action(tool: string, summary: string, ts: number): MovementToken {
  return { kind: "action", label: canonicalMovementLabel({ kind: "action", tool }), tool, summary, ts };
}

function seq(id: string, tokens: MovementToken[]): MovementSequence {
  return { trajectoryId: id, tokens };
}

describe("NGramMovementBackend + MovementInferenceEngine", () => {
  it("repeats a recorded movement exactly", () => {
    const recorded = seq("rec-1", [
      action("window", "open editor", 1),
      action("keyboard", "type hello", 2),
      action("mouse", "click save", 3),
    ]);
    const engine = new MovementInferenceEngine(new NGramMovementBackend(3));
    engine.train([recorded]);

    // Seeding the first movement replays the remaining two in order, each with
    // full confidence (unambiguous continuation for this trajectory).
    const rollout = engine.rollout([recorded.tokens[0]!], { maxSteps: 2 });
    expect(rollout.map((prediction) => prediction.token.summary)).toEqual(["type hello", "click save"]);
    expect(rollout.every((prediction) => prediction.confidence === 1)).toBe(true);
  });

  it("generalizes to a new-but-related prefix via backoff", () => {
    // Two related trajectories: open → type → click <button> → done.
    const submit = seq("submit", [
      action("window", "open form", 1),
      action("keyboard", "type name", 2),
      action("mouse", "click submit", 3),
      action("window", "show confirmation", 4),
    ]);
    const cancel = seq("cancel", [
      action("window", "open form", 1),
      action("keyboard", "type name", 2),
      action("mouse", "click cancel", 3),
      action("window", "show dismissal", 4),
    ]);
    const engine = new MovementInferenceEngine(new NGramMovementBackend(3));
    engine.train([submit, cancel]);

    // A NEW seed the model never saw at full length, but structurally related:
    // "open a *different* form, then type". The model should still predict the
    // learned continuation: a mouse click.
    const seed: MovementToken[] = [
      action("window", "open settings form", 10),
      action("keyboard", "type query", 11),
    ];
    const prediction = engine.predictNext(seed);
    expect(prediction).toBeDefined();
    expect(prediction!.token.label).toBe("action:mouse");
    // Backed off from the unseen full context to the shared "type → click" bigram.
    expect(prediction!.backoffOrder).toBeGreaterThanOrEqual(1);
  });

  it("falls back to the unconditional most-frequent movement for an unknown context", () => {
    const engine = new MovementInferenceEngine(new NGramMovementBackend(2));
    engine.train([
      seq("a", [action("mouse", "click", 1), action("mouse", "click", 2), action("keyboard", "type", 3)]),
    ]);
    const prediction = engine.predictNext([action("browser", "navigate", 99)]);
    expect(prediction).toBeDefined();
    // "click" appears most, so the order-0 backoff picks it.
    expect(prediction!.backoffOrder).toBe(0);
    expect(prediction!.token.summary).toBe("click");
  });

  it("returns undefined when the model is untrained", () => {
    const engine = new MovementInferenceEngine(new NGramMovementBackend());
    expect(engine.predictNext([action("mouse", "click", 1)])).toBeUndefined();
    expect(engine.rollout([action("mouse", "click", 1)])).toEqual([]);
  });

  it("stops a rollout on a terminal label", () => {
    const recorded = seq("rec", [
      action("mouse", "click", 1),
      action("keyboard", "type", 2),
      action("window", "close", 3),
    ]);
    const engine = new MovementInferenceEngine(new NGramMovementBackend(3));
    engine.train([recorded, recorded]);
    const rollout = engine.rollout([recorded.tokens[0]!], {
      maxSteps: 10,
      stopLabels: [canonicalMovementLabel({ kind: "action", tool: "window" })],
    });
    expect(rollout.at(-1)!.token.label).toBe("action:window");
    expect(rollout).toHaveLength(2);
  });

  it("halts runaway self-repeating predictions via maxRepeat", () => {
    const engine = new MovementInferenceEngine(new NGramMovementBackend(1));
    // A degenerate loop: click always follows click.
    engine.train([seq("loop", [action("mouse", "click", 1), action("mouse", "click", 2), action("mouse", "click", 3)])]);
    const rollout = engine.rollout([action("mouse", "click", 0)], { maxSteps: 100, maxRepeat: 3 });
    expect(rollout.length).toBe(3);
  });

  it("scores next-step accuracy on held-out sequences", () => {
    const train = seq("t", [action("mouse", "click", 1), action("keyboard", "type", 2), action("window", "close", 3)]);
    const engine = new MovementInferenceEngine(new NGramMovementBackend(3));
    engine.train([train]);
    // Held-out sequence identical in structure → perfect next-step accuracy.
    const evalResult = engine.evaluateNextStep([train]);
    expect(evalResult.evaluated).toBe(2);
    expect(evalResult.accuracy).toBe(1);
  });
});

describe("movement sequence adapters", () => {
  it("derives an ordered movement sequence from a trajectory span", () => {
    const trajectory = buildTrajectorySpan({
      id: "traj-1",
      sessionId: "sess-1",
      observations: [{ kind: "observation", source: "screen", summary: "form visible", ts: 20 }],
      actions: [
        { kind: "action", tool: "keyboard", summary: "type name", ts: 30 },
        { kind: "action", tool: "mouse", summary: "click submit", ts: 10 },
      ],
    });
    const sequence = sequenceFromTrajectory(trajectory);
    expect(sequence.tokens.map((token) => token.summary)).toEqual(["click submit", "form visible", "type name"]);
    expect(sequence.tokens.map((token) => token.label)).toEqual([
      "action:mouse",
      "observation:screen",
      "action:keyboard",
    ]);
  });

  it("derives a movement sequence from a replay manifest (transcript events dropped)", () => {
    const trajectory = buildTrajectorySpan({
      id: "traj-2",
      sessionId: "sess-2",
      observations: [{ kind: "observation", source: "browser", summary: "page loaded", ts: 20 }],
      actions: [{ kind: "action", tool: "browser", summary: "click link", ts: 40 }],
    });
    const replay = buildReplayManifest({
      sessionId: "sess-2",
      transcript: [{ id: "m1", message: { role: "user", content: "go", timestamp: 5 } }],
      trajectories: [trajectory],
    });
    const sequence = sequenceFromReplay(replay);
    expect(sequence.tokens.every((token) => token.kind !== undefined)).toBe(true);
    expect(sequence.tokens.map((token) => token.label)).toEqual(["observation:browser", "action:browser"]);
  });

  it("filters empty sequences when batching trajectories", () => {
    const empty = buildTrajectorySpan({ id: "empty", sessionId: "s" });
    const filled = buildTrajectorySpan({
      id: "filled",
      sessionId: "s",
      actions: [{ kind: "action", tool: "mouse", summary: "click", ts: 1 }],
    });
    expect(sequencesFromTrajectories([empty, filled]).map((sequence) => sequence.trajectoryId)).toEqual(["filled"]);
  });
});
