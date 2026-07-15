import { describe, expect, it } from "vitest";
import type { ReplayManifest } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  MOVEMENT_END,
  MarkovMovementBackend,
  MovementModelTrainer,
  evaluateReplayFidelity,
  generateMovements,
  movementSequenceFromTrajectory,
  movementSequencesFromReplayManifest,
  synthesizeMovementSequences,
  tokenizeMovementAction,
  type MovementSequence,
} from "./movement-model.js";

function span(id: string, actions: Array<{ tool: string; summary: string; ts: number; metadata?: Record<string, unknown> }>): TrajectorySpan {
  return {
    id,
    sessionId: "session-1",
    createdAt: "2026-01-01T00:00:00.000Z",
    captureTier: "full",
    observations: [],
    actions: actions.map((a) => ({ kind: "action", ...a })),
  };
}

describe("tokenizeMovementAction", () => {
  it("prefers structured gesture metadata for a canonical channel:verb:target token", () => {
    const token = tokenizeMovementAction({
      kind: "action",
      tool: "device",
      summary: "tapped Send button",
      ts: 1,
      metadata: { gesture: "tap", target: "sendButton" },
    });
    expect(token).toBe("device:tap:sendbutton");
  });

  it("falls back to the summary when metadata is absent", () => {
    const token = tokenizeMovementAction({
      kind: "action",
      tool: "keyboard",
      summary: "type hello",
      ts: 1,
    });
    expect(token).toBe("keyboard:type:hello");
  });

  it("produces stable tokens regardless of casing/whitespace", () => {
    const a = tokenizeMovementAction({ kind: "action", tool: "Device", summary: "x", ts: 1, metadata: { gesture: "Swipe", direction: "UP" } });
    const b = tokenizeMovementAction({ kind: "action", tool: "device", summary: "x", ts: 1, metadata: { gesture: "swipe ", direction: " up" } });
    expect(a).toBe(b);
    expect(a).toBe("device:swipe:up");
  });
});

describe("sequence extraction", () => {
  it("orders trajectory actions by timestamp", () => {
    const s = movementSequenceFromTrajectory(
      span("t1", [
        { tool: "device", summary: "b", ts: 30, metadata: { gesture: "tap", target: "b" } },
        { tool: "device", summary: "a", ts: 10, metadata: { gesture: "tap", target: "a" } },
        { tool: "device", summary: "c", ts: 20, metadata: { gesture: "tap", target: "c" } },
      ]),
    );
    expect(s.tokens).toEqual(["device:tap:a", "device:tap:c", "device:tap:b"]);
  });

  it("groups replay-manifest action events by trajectory", () => {
    const manifest: ReplayManifest = {
      version: 1,
      sessionId: "session-1",
      trajectoryIds: ["t1", "t2"],
      eventCount: 4,
      events: [
        { kind: "observation", ts: 1, trajectoryId: "t1", source: "device", summary: "screen" },
        { kind: "action", ts: 2, trajectoryId: "t1", tool: "device", summary: "tap a" },
        { kind: "action", ts: 3, trajectoryId: "t2", tool: "device", summary: "tap x" },
        { kind: "action", ts: 4, trajectoryId: "t1", tool: "device", summary: "tap b" },
      ],
    };
    const sequences = movementSequencesFromReplayManifest(manifest);
    expect(sequences).toEqual([
      { id: "t1", tokens: ["device:tap:a", "device:tap:b"] },
      { id: "t2", tokens: ["device:tap:x"] },
    ]);
  });
});

describe("MarkovMovementBackend replay", () => {
  it("reproduces a recorded movement exactly after training on it once", () => {
    const backend = new MarkovMovementBackend(2);
    const recorded: MovementSequence = {
      id: "rec",
      tokens: ["device:tap:menu", "device:tap:new", "keyboard:type:title", "device:tap:save"],
    };
    const state = backend.train([recorded]);
    const generated = generateMovements(backend, state, { maxSteps: 16 });
    expect(generated).toEqual(recorded.tokens);
  });

  it("terminates generation at the end sentinel and never emits it", () => {
    const backend = new MarkovMovementBackend(1);
    const state = backend.train([{ id: "r", tokens: ["a:b:c"] }]);
    const generated = generateMovements(backend, state, { maxSteps: 100 });
    expect(generated).toEqual(["a:b:c"]);
    expect(generated).not.toContain(MOVEMENT_END);
  });

  it("is deterministic across independent trainings", () => {
    const backend = new MarkovMovementBackend(2);
    const data: MovementSequence[] = [
      { id: "1", tokens: ["a:x:1", "a:x:2", "a:x:3"] },
      { id: "2", tokens: ["a:x:1", "a:x:2", "a:x:9"] },
    ];
    const first = generateMovements(backend, backend.train(data), {});
    const second = generateMovements(backend, backend.train([...data].reverse()), {});
    expect(first).toEqual(second);
  });
});

describe("MarkovMovementBackend generalization (stupid-backoff)", () => {
  it("predicts a plausible continuation for an unseen high-order prefix by backing off", () => {
    const backend = new MarkovMovementBackend(2);
    // Every recorded flow ends `...:open -> :confirm`. A novel prefix ending in
    // :open should still predict :confirm via the order-1 backoff context.
    const state = backend.train([
      { id: "1", tokens: ["app:click:a", "app:click:open", "app:click:confirm"] },
      { id: "2", tokens: ["app:click:b", "app:click:open", "app:click:confirm"] },
    ]);
    const prediction = backend.predictNext(state, ["app:click:z", "app:click:open"]);
    expect(prediction?.token).toBe("app:click:confirm");
    expect(prediction?.order).toBe(1); // matched the shorter, generalizing context
  });

  it("generalizes: performs the shared suffix of a movement it was never trained on", () => {
    const backend = new MarkovMovementBackend(2);
    // Several recorded flows enter differently but converge on the same
    // review->submit tail. The model has never seen the `entry:d` prefix.
    const state = backend.train([
      { id: "1", tokens: ["nav:entry:a", "form:tap:review", "form:tap:submit"] },
      { id: "2", tokens: ["nav:entry:b", "form:tap:review", "form:tap:submit"] },
      { id: "3", tokens: ["nav:entry:c", "form:tap:review", "form:tap:submit"] },
    ]);
    const novel: MovementSequence = {
      id: "novel",
      tokens: ["nav:entry:d", "form:tap:review", "form:tap:submit"],
    };
    // Once the novel prefix reaches the shared token, backoff recovers the tail.
    expect(backend.predictNext(state, ["nav:entry:d", "form:tap:review"])?.token).toBe("form:tap:submit");

    const report = evaluateReplayFidelity(backend, state, [novel]);
    // Teacher-forced, the shared-suffix steps (submit via order-1, END via
    // order-2) are recovered; only the novel entry token and the token
    // immediately following it are unrecoverable — a real Markov limitation.
    expect(report.correct).toBeGreaterThanOrEqual(2);
    expect(report.accuracy).toBeGreaterThanOrEqual(0.5);
  });
});

describe("evaluateReplayFidelity", () => {
  it("scores perfect fidelity when evaluating on the trained sequence", () => {
    const backend = new MarkovMovementBackend(2);
    const seq: MovementSequence = { id: "s", tokens: ["a:b:1", "a:b:2", "a:b:3"] };
    const state = backend.train([seq]);
    const report = evaluateReplayFidelity(backend, state, [seq]);
    expect(report.accuracy).toBe(1);
    expect(report.correct).toBe(report.steps);
    expect(report.perSequence[0]?.id).toBe("s");
  });
});

describe("MovementModelTrainer", () => {
  it("holds out every Nth trajectory deterministically and evaluates it", () => {
    const spans = Array.from({ length: 4 }, (_, i) =>
      span(`t${i}`, [
        { tool: "device", summary: "open", ts: 1, metadata: { gesture: "tap", target: "open" } },
        { tool: "device", summary: "save", ts: 2, metadata: { gesture: "tap", target: "save" } },
      ]),
    );
    const trainer = new MovementModelTrainer(new MarkovMovementBackend(1));
    const result = trainer.trainFromTrajectories(spans, { holdoutEvery: 2 });
    expect(result.train.map((s) => s.id)).toEqual(["t0", "t2"]);
    expect(result.holdout.map((s) => s.id)).toEqual(["t1", "t3"]);
    expect(result.evaluation.accuracy).toBe(1);
  });

  it("produces a serializable state artifact that round-trips through JSON", () => {
    const backend = new MarkovMovementBackend(2);
    const state = backend.train([{ id: "1", tokens: ["a:b:c", "a:b:d"] }]);
    const restored = JSON.parse(JSON.stringify(state));
    expect(generateMovements(backend, restored, {})).toEqual(generateMovements(backend, state, {}));
    expect(restored.vocabulary).toEqual(["a:b:c", "a:b:d"]);
  });
});

describe("synthesizeMovementSequences", () => {
  it("generates the requested count of structurally-related variants", () => {
    const sequences = synthesizeMovementSequences(
      { channel: "device", steps: [{ verb: "tap", targets: ["a", "b"] }] },
      3,
    );
    expect(sequences).toHaveLength(3);
    expect(sequences.map((s) => s.tokens[0])).toEqual(["device:tap:a", "device:tap:b", "device:tap:a"]);
  });

  it("rejects a negative count", () => {
    expect(() => synthesizeMovementSequences({ channel: "d", steps: [] }, -1)).toThrow();
  });
});
