import { describe, expect, it } from "vitest";
import { buildTrajectorySpan, type TrajectoryAction } from "../capture/trajectory.js";
import type { ExportedReplayManifest } from "./export-manifest.js";
import {
  MOVEMENT_END_TOKEN,
  MarkovMovementBackend,
  createMovementBackend,
  datasetFromReplays,
  datasetFromReviewedExport,
  datasetFromTrajectories,
  evaluateMovementModel,
  replayFidelity,
  tokenizeAction,
  type MovementDataset,
  type MovementSequence,
} from "./movement-model.js";

function action(tool: string, summary: string, ts: number, metadata?: Record<string, unknown>): TrajectoryAction {
  return { kind: "action", tool, summary, ts, ...(metadata ? { metadata } : {}) };
}

describe("tokenizeAction", () => {
  it("builds a structured token from tool + gesture + target", () => {
    expect(tokenizeAction(action("device", "tapped submit", 1, { gesture: "tap", target: "Submit Button" }))).toBe(
      "device:tap:submit-button",
    );
  });

  it("falls back to the summary when there is no gesture metadata", () => {
    expect(tokenizeAction(action("browser", "Clicked the Login link", 1))).toBe("browser:clicked-the-login-link");
  });

  it("prefers direction when no target is present", () => {
    expect(tokenizeAction(action("device", "scrolled", 1, { gesture: "scroll", direction: "down" }))).toBe(
      "device:scroll:down",
    );
  });
});

describe("dataset builders", () => {
  it("orders trajectory actions by timestamp and drops empty spans", () => {
    const withActions = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [action("device", "b", 20, { gesture: "type" }), action("device", "a", 10, { gesture: "tap" })],
    });
    const empty = buildTrajectorySpan({ id: "t2", sessionId: "s1" });
    const dataset = datasetFromTrajectories([withActions, empty]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]).toEqual({ id: "t1", tokens: ["device:tap", "device:type"] });
  });

  it("builds a dataset from exported replay manifests", () => {
    const replay: ExportedReplayManifest = {
      sessionId: "s1",
      trajectoryIds: ["t1"],
      eventCount: 3,
      events: [
        { kind: "action", ts: 2, trajectoryId: "t1", tool: "device", summary: "second" },
        { kind: "observation", ts: 1, trajectoryId: "t1", source: "device", summary: "noise" },
        { kind: "action", ts: 1, trajectoryId: "t1", tool: "device", summary: "first" },
      ],
    };
    const dataset = datasetFromReplays([replay]);
    expect(dataset.sequences[0]).toEqual({ id: "t1", tokens: ["device:first", "device:second"] });
  });

  it("datasetFromReviewedExport reads the manifest replays", () => {
    const dataset = datasetFromReviewedExport({
      version: 1,
      createdAt: "2026-01-01T00:00:00.000Z",
      reviewedBy: "tester",
      purpose: "test",
      targetPlatform: "apple-silicon",
      modes: ["sft"],
      rawCaptureIncluded: false,
      promotedSkills: [],
      executableSkills: [],
      executableSkillRuns: [],
      memories: [],
      trajectories: [],
      replays: [
        {
          sessionId: "s1",
          trajectoryIds: ["t1"],
          eventCount: 1,
          events: [{ kind: "action", ts: 1, trajectoryId: "t1", tool: "device", summary: "go" }],
        },
      ],
    });
    expect(dataset.sequences[0]?.tokens).toEqual(["device:go"]);
  });
});

describe("MarkovMovementBackend — repeat recorded movements", () => {
  it("reproduces a recorded sequence exactly (replay fidelity = 1)", () => {
    const dataset: MovementDataset = {
      sequences: [{ id: "t1", tokens: ["open", "click-field", "type-text", "click-submit"] }],
    };
    const model = new MarkovMovementBackend(3).train(dataset);
    const fidelity = replayFidelity(model, dataset.sequences[0]!);
    expect(fidelity.fidelity).toBe(1);
    expect(fidelity.generated).toEqual(["open", "click-field", "type-text", "click-submit"]);
  });

  it("learns the end sentinel so generation stops", () => {
    const model = new MarkovMovementBackend(3).train({ sequences: [{ id: "t1", tokens: ["a", "b"] }] });
    // Predicting after the full sequence should yield the end sentinel.
    expect(model.predictNext(["a", "b"]).token).toBe(MOVEMENT_END_TOKEN);
    expect(model.generate(["a"], 10)).toEqual(["a", "b"]);
    expect(model.vocabulary()).toEqual(["a", "b"]);
  });

  it("is deterministic across runs", () => {
    const dataset: MovementDataset = {
      sequences: [
        { id: "t1", tokens: ["a", "b", "c"] },
        { id: "t2", tokens: ["a", "b", "d"] },
      ],
    };
    const a = new MarkovMovementBackend(2).train(dataset).predictNext(["a", "b"]);
    const b = new MarkovMovementBackend(2).train(dataset).predictNext(["a", "b"]);
    expect(a).toEqual(b);
    // Tie between c and d (count 1 each) resolves lexicographically → c.
    expect(a.token).toBe("c");
    expect(a.alternatives.map((entry) => entry.token)).toEqual(["c", "d"]);
  });
});

describe("MarkovMovementBackend — generalize to new but related movements", () => {
  it("predicts a related continuation for a novel prefix via back-off", () => {
    // Two workflows that share a middle: (start) → click-field → type-text → submit.
    const dataset: MovementDataset = {
      sequences: [
        { id: "t1", tokens: ["open-app", "click-field", "type-text", "click-submit"] },
        { id: "t2", tokens: ["open-browser", "click-field", "type-text", "click-submit"] },
      ],
    };
    const model = new MarkovMovementBackend(3).train(dataset);

    // A never-before-seen opening movement, but a familiar tail.
    const prediction = model.predictNext(["open-settings", "click-field"]);
    expect(prediction.token).toBe("type-text");
    // High-order context (open-settings, click-field) is unseen → backs off to order 1.
    expect(prediction.order).toBe(1);
    expect(prediction.probability).toBe(1);
  });

  it("evaluateMovementModel scores next-token accuracy on held-out sequences", () => {
    const train: MovementDataset = {
      sequences: [
        { id: "t1", tokens: ["open-app", "click-field", "type-text", "click-submit"] },
        { id: "t2", tokens: ["open-browser", "click-field", "type-text", "click-submit"] },
      ],
    };
    const model = new MarkovMovementBackend(3).train(train);
    // Held-out but related: unseen opener, then the shared tail the model generalizes over.
    const heldOut: MovementSequence[] = [
      { id: "eval", tokens: ["open-settings", "click-field", "type-text", "click-submit"] },
    ];
    const result = evaluateMovementModel(model, heldOut);
    // 3 predictions (positions 1..3); only position 1 (novel opener → click-field) is unknowable.
    expect(result.predictions).toBe(3);
    expect(result.correct).toBe(2);
    expect(result.accuracy).toBeCloseTo(2 / 3);
    expect(result.perSequence[0]?.id).toBe("eval");
  });
});

describe("createMovementBackend registry", () => {
  it("markov backend conditions on higher-order context", () => {
    const backend = createMovementBackend("markov", { order: 2 });
    const model = backend.train({
      sequences: [
        { id: "t1", tokens: ["a", "x", "p"] },
        { id: "t2", tokens: ["b", "x", "q"] },
      ],
    });
    // Order-2 context distinguishes which "x" we mean.
    expect(model.predictNext(["a", "x"]).token).toBe("p");
    expect(model.predictNext(["b", "x"]).token).toBe("q");
  });

  it("most-frequent backend is an order-0 control baseline", () => {
    const backend = createMovementBackend("most-frequent");
    expect(backend.name).toBe("most-frequent");
    const model = backend.train({
      sequences: [
        { id: "t1", tokens: ["a", "x", "x", "p"] },
        { id: "t2", tokens: ["b", "x", "q"] },
      ],
    });
    expect(model.order).toBe(0);
    // Ignores context entirely; "x" is the single most frequent movement.
    expect(model.predictNext(["a", "x"]).token).toBe("x");
    expect(model.predictNext(["b", "x"]).token).toBe("x");
  });
});
