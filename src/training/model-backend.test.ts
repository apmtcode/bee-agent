import { describe, expect, it } from "vitest";
import {
  MarkovMovementBackend,
  buildMovementDataset,
  createDefaultMovementBackends,
  movementSequenceFromReplay,
  movementSequenceFromTrajectory,
  tokenKey,
  type MovementSequence,
  type MovementToken,
} from "./model-backend.js";
import type { ReplayManifest } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

function action(label: string): MovementToken {
  return { kind: "action", label };
}

function observation(label: string): MovementToken {
  return { kind: "observation", label };
}

function seq(id: string, tokens: MovementToken[]): MovementSequence {
  return { id, tokens };
}

describe("MarkovMovementBackend", () => {
  it("repeats a recorded movement sequence it was trained on", () => {
    const recorded = [action("open-app"), action("focus-field"), action("type-text"), action("submit")];
    const model = new MarkovMovementBackend().train(buildMovementDataset([seq("r1", recorded)]), { order: 2 });

    const rollout = model.generate([action("open-app")], 5);

    expect(rollout).toEqual([action("focus-field"), action("type-text"), action("submit")]);
  });

  it("predicts the most likely next movement with a confidence", () => {
    // "click" is followed by "type" twice and "scroll" once.
    const dataset = buildMovementDataset([
      seq("a", [action("click"), action("type")]),
      seq("b", [action("click"), action("type")]),
      seq("c", [action("click"), action("scroll")]),
    ]);
    const model = new MarkovMovementBackend().train(dataset, { order: 1 });

    const prediction = model.predictNext([action("click")]);

    expect(prediction.token).toEqual(action("type"));
    expect(prediction.confidence).toBeCloseTo(2 / 3);
    expect(prediction.candidates.map((c) => tokenKey(c.token))).toEqual(["action:type", "action:scroll"]);
  });

  it("generalises to a new-but-related context via back-off", () => {
    // The bigram [drag>drop] was never seen, but "drop" is always followed by
    // "confirm" — back-off to the order-1 context recovers that.
    const dataset = buildMovementDataset([
      seq("a", [action("select"), action("drop"), action("confirm")]),
      seq("b", [action("copy"), action("drop"), action("confirm")]),
    ]);
    const model = new MarkovMovementBackend().train(dataset, { order: 2 });

    const prediction = model.predictNext([action("drag"), action("drop")]);

    expect(prediction.token).toEqual(action("confirm"));
    expect(prediction.backoffOrder).toBe(1);
  });

  it("returns an empty prediction for a wholly unknown context", () => {
    const model = new MarkovMovementBackend().train(buildMovementDataset([]), { order: 2 });

    const prediction = model.predictNext([action("mystery")]);

    expect(prediction.token).toBeUndefined();
    expect(prediction.confidence).toBe(0);
    expect(prediction.candidates).toEqual([]);
  });

  it("breaks ties deterministically by token key", () => {
    const dataset = buildMovementDataset([
      seq("a", [action("start"), action("zzz")]),
      seq("b", [action("start"), action("aaa")]),
    ]);
    const model = new MarkovMovementBackend().train(dataset, { order: 1 });

    const first = model.predictNext([action("start")]);
    const second = model.predictNext([action("start")]);

    expect(first.token).toEqual(action("aaa"));
    expect(second.token).toEqual(first.token);
  });

  it("round-trips through serialize/restore", () => {
    const dataset = buildMovementDataset([seq("a", [action("one"), action("two"), action("three")])]);
    const backend = new MarkovMovementBackend();
    const original = backend.train(dataset, { order: 2 });

    const snapshot = original.serialize();
    const restored = backend.restore(snapshot);

    expect(restored.order).toBe(2);
    expect(restored.generate([action("one")], 3)).toEqual(original.generate([action("one")], 3));
    // Snapshot is JSON-serialisable.
    expect(JSON.parse(JSON.stringify(snapshot))).toEqual(snapshot);
  });

  it("stops generating at maxSteps", () => {
    const model = new MarkovMovementBackend().train(
      buildMovementDataset([seq("a", [action("a"), action("b"), action("c"), action("d")])]),
      { order: 1 },
    );

    expect(model.generate([action("a")], 2)).toEqual([action("b"), action("c")]);
  });

  it("is registered as a default in-process backend", () => {
    const backends = createDefaultMovementBackends();
    expect(backends["markov-movement"]).toBeInstanceOf(MarkovMovementBackend);
  });
});

describe("movement dataset extraction", () => {
  it("extracts an ordered sequence from a replay manifest, dropping transcript turns", () => {
    const manifest: ReplayManifest = {
      version: 1,
      sessionId: "session-1",
      trajectoryIds: ["t1"],
      eventCount: 3,
      events: [
        { kind: "transcript", ts: 1, messageId: "m1", role: "user", content: "go" },
        { kind: "observation", ts: 2, trajectoryId: "t1", source: "screen", summary: "form visible" },
        { kind: "action", ts: 3, trajectoryId: "t1", tool: "click", summary: "clicked submit" },
      ],
    };

    const sequence = movementSequenceFromReplay(manifest);

    expect(sequence.id).toBe("session-1");
    expect(sequence.tokens).toEqual([observation("screen"), action("click")]);
  });

  it("extracts a time-ordered sequence from a trajectory span", () => {
    const span: TrajectorySpan = {
      id: "traj-1",
      sessionId: "session-1",
      createdAt: "2026-01-01T00:00:00.000Z",
      captureTier: "operator",
      observations: [{ kind: "observation", source: "screen", summary: "seen", ts: 10 }],
      actions: [
        { kind: "action", tool: "type", summary: "typed", ts: 20 },
        { kind: "action", tool: "click", summary: "clicked", ts: 5 },
      ],
    };

    const sequence = movementSequenceFromTrajectory(span);

    // Sorted by ts: click(5), screen(10), type(20).
    expect(sequence.tokens).toEqual([action("click"), observation("screen"), action("type")]);
  });

  it("trains directly on trajectory-derived sequences", () => {
    const span: TrajectorySpan = {
      id: "traj-1",
      sessionId: "s",
      createdAt: "2026-01-01T00:00:00.000Z",
      captureTier: "operator",
      observations: [],
      actions: [
        { kind: "action", tool: "open", summary: "", ts: 1 },
        { kind: "action", tool: "read", summary: "", ts: 2 },
        { kind: "action", tool: "close", summary: "", ts: 3 },
      ],
    };
    const model = new MarkovMovementBackend().train(
      buildMovementDataset([movementSequenceFromTrajectory(span)]),
      { order: 2 },
    );

    expect(model.generate([action("open")], 3)).toEqual([action("read"), action("close")]);
  });
});
