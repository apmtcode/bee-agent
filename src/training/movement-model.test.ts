import { describe, expect, it } from "vitest";
import { MarkovMovementBackend } from "./markov-backend.js";
import {
  buildMovementDatasetFromTrajectories,
  buildMovementSequenceFromReplay,
  defaultMovementTokenizer,
  evaluateMovementModel,
  type MovementSequence,
} from "./movement-model.js";
import type { ReplayManifest } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

function trajectory(overrides: Partial<TrajectorySpan> & Pick<TrajectorySpan, "id">): TrajectorySpan {
  return {
    id: overrides.id,
    sessionId: overrides.sessionId ?? "session-1",
    createdAt: overrides.createdAt ?? "2026-07-15T00:00:00.000Z",
    captureTier: overrides.captureTier ?? "operator",
    observations: overrides.observations ?? [],
    actions: overrides.actions ?? [],
    ...(overrides.outcome ? { outcome: overrides.outcome } : {}),
    ...(overrides.review ? { review: overrides.review } : {}),
  };
}

describe("buildMovementDatasetFromTrajectories", () => {
  it("orders actions by timestamp and tokenizes them", () => {
    const dataset = buildMovementDatasetFromTrajectories([
      trajectory({
        id: "t1",
        actions: [
          { kind: "action", tool: "mouse.click", summary: "", ts: 30 },
          { kind: "action", tool: "focus.window", summary: "", ts: 10 },
          { kind: "action", tool: "mouse.move", summary: "", ts: 20 },
        ],
      }),
    ]);

    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]!.tokens).toEqual(["focus.window", "mouse.move", "mouse.click"]);
    expect(dataset.sequences[0]!.sourceId).toBe("t1");
  });

  it("skips trajectories with no actions", () => {
    const dataset = buildMovementDatasetFromTrajectories([trajectory({ id: "empty" })]);
    expect(dataset.sequences).toHaveLength(0);
  });

  it("filters to approved trajectories when requireApproved is set", () => {
    const approved = trajectory({
      id: "ok",
      actions: [{ kind: "action", tool: "a", summary: "", ts: 1 }],
      review: { status: "approved", reviewedAt: "", reviewedBy: "me" },
    });
    const pending = trajectory({
      id: "pending",
      actions: [{ kind: "action", tool: "b", summary: "", ts: 1 }],
    });

    const dataset = buildMovementDatasetFromTrajectories([approved, pending], { requireApproved: true });
    expect(dataset.sequences.map((sequence) => sequence.sourceId)).toEqual(["ok"]);
  });

  it("prefers redacted reviewed actions over raw actions", () => {
    const dataset = buildMovementDatasetFromTrajectories([
      trajectory({
        id: "t1",
        actions: [{ kind: "action", tool: "raw", summary: "", ts: 1 }],
        review: {
          status: "approved",
          reviewedAt: "",
          reviewedBy: "me",
          redactedActions: [{ ts: 1, tool: "redacted", summary: "" }],
        },
      }),
    ]);

    expect(dataset.sequences[0]!.tokens).toEqual(["redacted"]);
  });

  it("carries terminal reward through to the sequence", () => {
    const dataset = buildMovementDatasetFromTrajectories([
      trajectory({
        id: "t1",
        actions: [{ kind: "action", tool: "a", summary: "", ts: 1 }],
        outcome: { status: "success", summary: "done", reward: 0.9 },
      }),
    ]);

    expect(dataset.sequences[0]!.reward).toBe(0.9);
  });

  it("supports a custom tokenizer", () => {
    const dataset = buildMovementDatasetFromTrajectories(
      [
        trajectory({
          id: "t1",
          actions: [{ kind: "action", tool: "mouse.move", summary: "to (10,20)", ts: 1 }],
        }),
      ],
      { tokenizer: (action) => `${action.tool}|${action.summary}` },
    );
    expect(dataset.sequences[0]!.tokens).toEqual(["mouse.move|to (10,20)"]);
  });
});

describe("buildMovementSequenceFromReplay", () => {
  it("extracts only action events in order", () => {
    const replay: ReplayManifest = {
      version: 1,
      sessionId: "s1",
      trajectoryIds: ["t1"],
      eventCount: 3,
      events: [
        { kind: "observation", ts: 1, trajectoryId: "t1", source: "os", summary: "" },
        { kind: "action", ts: 2, trajectoryId: "t1", tool: "mouse.move", summary: "" },
        { kind: "action", ts: 3, trajectoryId: "t1", tool: "mouse.click", summary: "" },
      ],
    };

    const sequence = buildMovementSequenceFromReplay(replay);
    expect(sequence.sourceId).toBe("s1");
    expect(sequence.tokens).toEqual(["mouse.move", "mouse.click"]);
  });
});

describe("evaluateMovementModel", () => {
  it("reports perfect accuracy replaying the training data", () => {
    const backend = new MarkovMovementBackend({ maxOrder: 3 });
    const sequences: MovementSequence[] = [
      { sourceId: "a", tokens: ["open", "edit", "save", "close"] },
    ];
    const model = backend.train({ version: 1, sequences });

    const result = evaluateMovementModel(backend, model, sequences);
    expect(result.accuracy).toBe(1);
    expect(result.totalPredictions).toBe(3);
    expect(result.correct).toBe(3);
  });

  it("measures generalization on a held-out but related sequence", () => {
    const backend = new MarkovMovementBackend({ maxOrder: 2 });
    const train: MovementSequence[] = [
      { sourceId: "1", tokens: ["open", "edit", "save", "close"] },
      { sourceId: "2", tokens: ["launch", "edit", "save", "quit"] },
    ];
    const model = backend.train({ version: 1, sequences: train });

    // Held-out flow shares the "edit -> save" bigram; the model should predict
    // "save" after "edit" even though this exact flow was never recorded.
    const heldOut: MovementSequence[] = [{ sourceId: "3", tokens: ["boot", "edit", "save", "exit"] }];
    const result = evaluateMovementModel(backend, model, heldOut);
    expect(result.accuracy).toBeGreaterThan(0);
    expect(result.averageConfidence).toBeGreaterThan(0);
  });
});

describe("defaultMovementTokenizer", () => {
  it("uses the tool name", () => {
    expect(defaultMovementTokenizer({ tool: "mouse.move", summary: "x" })).toBe("mouse.move");
  });
});
