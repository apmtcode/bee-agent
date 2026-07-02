import { describe, expect, it } from "vitest";
import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  MarkovMovementBackend,
  MOVEMENT_STOP_TOKEN,
  actionToken,
  buildMovementDataset,
  evaluateGeneralization,
  evaluateReplayFidelity,
  generateMovementSequence,
  isActionToken,
  movementDatasetFromTrajectories,
  movementSequenceFromReplay,
  movementSequenceFromTrajectory,
  observationToken,
  type MovementSequence,
} from "./movement-model.js";
import type { ReplayManifest } from "../capture/replay.js";

function trajectory(id: string, actionSummaries: string[], observation = "window focused"): TrajectorySpan {
  let ts = 0;
  return {
    id,
    sessionId: `session-${id}`,
    createdAt: "2026-07-02T00:00:00.000Z",
    captureTier: "full",
    observations: [{ kind: "observation", source: "os.window", summary: observation, ts: ts++ }],
    actions: actionSummaries.map((summary) => ({
      kind: "action",
      tool: "mouse",
      summary,
      ts: ts++,
    })),
  };
}

describe("tokenization", () => {
  it("produces stable, slugged tokens", () => {
    expect(actionToken("Mouse.Move", "Drag Window!")).toBe("act:mouse-move:drag-window");
    expect(observationToken("os.window", "Window Focused")).toBe("obs:os-window:window-focused");
  });

  it("classifies action and stop tokens", () => {
    expect(isActionToken(actionToken("mouse", "click"))).toBe(true);
    expect(isActionToken(MOVEMENT_STOP_TOKEN)).toBe(true);
    expect(isActionToken(observationToken("os", "x"))).toBe(false);
  });

  it("orders trajectory steps by timestamp with observations before actions on ties", () => {
    const sequence = movementSequenceFromTrajectory(trajectory("t1", ["down", "up"]));
    expect(sequence.steps.map((step) => step.kind)).toEqual(["observation", "action", "action"]);
    expect(sequence.steps.at(-1)?.token).toBe(actionToken("mouse", "up"));
  });

  it("tokenizes a replay manifest timeline", () => {
    const replay: ReplayManifest = {
      version: 1,
      sessionId: "s1",
      trajectoryIds: ["traj-1"],
      eventCount: 3,
      events: [
        { kind: "transcript", ts: 0, messageId: "m0", role: "user", content: "open editor" },
        { kind: "observation", ts: 1, trajectoryId: "traj-1", source: "os.window", summary: "editor open" },
        { kind: "action", ts: 2, trajectoryId: "traj-1", tool: "keyboard", summary: "type hello" },
      ],
    };
    const sequence = movementSequenceFromReplay(replay);
    expect(sequence.trajectoryId).toBe("traj-1");
    expect(sequence.steps.map((step) => step.kind)).toEqual(["observation", "observation", "action"]);
  });
});

describe("dataset", () => {
  it("collects vocabulary and action tokens including the stop token", () => {
    const dataset = movementDatasetFromTrajectories([trajectory("t1", ["down", "up"])], { order: 2 });
    expect(dataset.order).toBe(2);
    expect(dataset.actionTokens).toContain(MOVEMENT_STOP_TOKEN);
    expect(dataset.actionTokens).toContain(actionToken("mouse", "down"));
    expect(dataset.vocabulary).toContain(observationToken("os.window", "window focused"));
    // vocabulary is sorted and de-duplicated
    expect([...dataset.vocabulary].sort()).toEqual(dataset.vocabulary);
  });

  it("clamps order to at least 1", () => {
    const dataset = buildMovementDataset([], { order: 0 });
    expect(dataset.order).toBe(1);
  });
});

describe("MarkovMovementBackend training and replay", () => {
  const backend = new MarkovMovementBackend();

  it("reproduces recorded movements exactly (objective 2c)", () => {
    const dataset = movementDatasetFromTrajectories(
      [trajectory("t1", ["move-a", "move-b", "move-c"]), trajectory("t2", ["move-a", "move-b", "move-c"])],
      { order: 3 },
    );
    const model = backend.train(dataset);
    const report = evaluateReplayFidelity(backend, model, dataset.sequences);
    expect(report.accuracy).toBe(1);
    expect(report.correct).toBe(report.totalActions);
    expect(report.totalActions).toBeGreaterThan(0);
  });

  it("generates the recorded movement chain from its opening context", () => {
    const dataset = movementDatasetFromTrajectories([trajectory("t1", ["down", "drag", "up"])], { order: 3 });
    const model = backend.train(dataset);
    const seed = [observationToken("os.window", "window focused")];
    const result = generateMovementSequence(backend, model, seed, { maxSteps: 10 });
    expect(result.stopped).toBe(true);
    expect(result.tokens).toEqual([
      actionToken("mouse", "down"),
      actionToken("mouse", "drag"),
      actionToken("mouse", "up"),
    ]);
  });

  it("stops generation at the step cap when the model never emits stop", () => {
    // A self-looping single-action model: after 'spin' the next action is 'spin'.
    const looping: MovementSequence = {
      trajectoryId: "loop",
      steps: [
        { kind: "action", token: actionToken("mouse", "spin"), ts: 0 },
        { kind: "action", token: actionToken("mouse", "spin"), ts: 1 },
        { kind: "action", token: actionToken("mouse", "spin"), ts: 2 },
      ],
    };
    const dataset = buildMovementDataset([looping], { order: 1 });
    const model = backend.train(dataset);
    const result = generateMovementSequence(backend, model, [actionToken("mouse", "spin")], { maxSteps: 4 });
    // order-1 'spin -> spin' dominates over the single terminal 'spin -> stop'.
    expect(result.tokens.length).toBe(4);
    expect(result.stopped).toBe(false);
  });

  it("returns an unknown prediction for an empty model", () => {
    const dataset = buildMovementDataset([], { order: 2 });
    const model = backend.train(dataset);
    const prediction = backend.predict(model, ["anything"]);
    expect(prediction.token).toBeUndefined();
    expect(prediction.source).toBe("unknown");
  });

  it("ranks candidates deterministically with a lexicographic tie-break", () => {
    // Two equally-frequent continuations of the same context -> lexicographic order.
    const seqA: MovementSequence = {
      trajectoryId: "a",
      steps: [
        { kind: "observation", token: "obs:ctx:x", ts: 0 },
        { kind: "action", token: actionToken("mouse", "zulu"), ts: 1 },
      ],
    };
    const seqB: MovementSequence = {
      trajectoryId: "b",
      steps: [
        { kind: "observation", token: "obs:ctx:x", ts: 0 },
        { kind: "action", token: actionToken("mouse", "alpha"), ts: 1 },
      ],
    };
    const dataset = buildMovementDataset([seqA, seqB], { order: 1 });
    const model = backend.train(dataset);
    const prediction = backend.predict(model, ["obs:ctx:x"]);
    expect(prediction.candidates.map((c) => c.token)).toEqual([
      actionToken("mouse", "alpha"),
      actionToken("mouse", "zulu"),
    ]);
    expect(prediction.confidence).toBeCloseTo(0.5);
  });
});

describe("generalization to new-but-related movements (objective 2d)", () => {
  const backend = new MarkovMovementBackend();

  it("predicts held-out next actions via shared shorter contexts", () => {
    // All trajectories share the tail pattern down -> up -> close -> done.
    // The held-out trajectory has a novel intro action (unpredictable) but the
    // same local tail, so the model must generalize via backoff, not exact recall.
    const sequences: MovementSequence[] = [
      seq("train-1", ["intro-a", "down", "up", "close", "done"]),
      seq("train-2", ["intro-b", "down", "up", "close", "done"]),
      seq("train-3", ["intro-c", "down", "up", "close", "done"]),
      seq("holdout", ["intro-z", "down", "up", "close", "done"]),
    ];
    const dataset = buildMovementDataset(sequences, { order: 3 });
    const report = evaluateGeneralization(backend, dataset, { holdoutCount: 1 });
    expect(report.holdoutSequenceCount).toBe(1);
    expect(report.trainSequenceCount).toBe(3);
    // The shared 'down -> up -> close' tail is predictable on the unseen trajectory.
    expect(report.accuracy).toBeGreaterThan(0.5);
    // ...and those correct predictions come from backoff/prior, not exact recall.
    expect(report.backoffShare).toBeGreaterThan(0);
  });

  it("clamps holdout count to leave at least one training sequence", () => {
    const dataset = buildMovementDataset([seq("a", ["x"]), seq("b", ["y"])], { order: 2 });
    const report = evaluateGeneralization(backend, dataset, { holdoutCount: 99 });
    expect(report.trainSequenceCount).toBeGreaterThanOrEqual(1);
    expect(report.holdoutSequenceCount).toBeGreaterThanOrEqual(1);
  });
});

function seq(id: string, actions: string[]): MovementSequence {
  let ts = 0;
  return {
    trajectoryId: id,
    steps: actions.map((summary) => ({ kind: "action" as const, token: actionToken("mouse", summary), ts: ts++ })),
  };
}
