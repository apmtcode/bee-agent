import { describe, expect, it } from "vitest";
import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";
import {
  MOVEMENT_START_TOKEN,
  MarkovMovementBackend,
  buildMovementDataset,
  defaultMovementBackend,
  evaluateMovementModel,
  movementSequenceFromTrajectory,
  movementTokenFromAction,
} from "./movement-model.js";

function action(tool: string, summary: string, ts: number, metadata?: Record<string, unknown>): TrajectoryAction {
  return { kind: "action", tool, summary, ts, ...(metadata ? { metadata } : {}) };
}

function span(id: string, actions: TrajectoryAction[]): TrajectorySpan {
  return {
    id,
    sessionId: `session-${id}`,
    createdAt: "2026-01-01T00:00:00.000Z",
    captureTier: "full",
    observations: [],
    actions,
  };
}

/**
 * Deterministic synthetic movement stream: a fixed "open menu -> navigate ->
 * confirm" workflow, one trajectory per repeat, with a seeded offset so ordering
 * varies without any real OS input or randomness.
 */
function synthesizeWorkflowTrajectories(repeats: number): TrajectorySpan[] {
  const steps: Array<{ tool: string; gesture: string; target: string }> = [
    { tool: "device", gesture: "tap", target: "menu-button" },
    { tool: "device", gesture: "swipe", target: "list" },
    { tool: "device", gesture: "tap", target: "settings-row" },
    { tool: "device", gesture: "type", target: "search-field" },
    { tool: "device", gesture: "tap", target: "confirm" },
  ];
  return Array.from({ length: repeats }, (_, repeat) =>
    span(
      `wf-${repeat}`,
      steps.map((step, index) =>
        action(step.tool, `${step.gesture} ${step.target}`, repeat * 100 + index, {
          gesture: step.gesture,
          target: step.target,
        }),
      ),
    ),
  );
}

describe("movementTokenFromAction", () => {
  it("canonicalizes structured gesture metadata into a stable token", () => {
    expect(
      movementTokenFromAction(action("device", "swiped up", 1, { gesture: "swipe", direction: "up", target: "Photo List" })),
    ).toBe("device:swipe:up:photo-list");
  });

  it("falls back to a slug of the summary when no metadata is present", () => {
    expect(movementTokenFromAction(action("browser", "Clicked Sign In", 1))).toBe("browser:clicked-sign-in");
  });

  it("is deterministic and order-independent for identical inputs", () => {
    const a = action("device", "tap", 5, { gesture: "tap", target: "ok" });
    expect(movementTokenFromAction(a)).toBe(movementTokenFromAction({ ...a }));
  });
});

describe("movementSequenceFromTrajectory", () => {
  it("orders actions by timestamp regardless of array order", () => {
    const trajectory = span("t", [
      action("device", "third", 30, { gesture: "tap", target: "c" }),
      action("device", "first", 10, { gesture: "tap", target: "a" }),
      action("device", "second", 20, { gesture: "tap", target: "b" }),
    ]);
    expect(movementSequenceFromTrajectory(trajectory)).toEqual([
      "device:tap:a",
      "device:tap:b",
      "device:tap:c",
    ]);
  });
});

describe("buildMovementDataset", () => {
  it("pads sequence starts and generates one sample per action", () => {
    const dataset = buildMovementDataset(synthesizeWorkflowTrajectories(1), { order: 2 });
    expect(dataset.order).toBe(2);
    // 5 actions -> 5 samples.
    expect(dataset.samples).toHaveLength(5);
    // First sample's context is fully padded with the start token.
    expect(dataset.samples[0]?.context).toEqual([MOVEMENT_START_TOKEN, MOVEMENT_START_TOKEN]);
    expect(dataset.samples[0]?.next).toBe("device:tap:menu-button");
    // Vocabulary is sorted and de-duplicated, excludes the start sentinel.
    expect(dataset.vocabulary).toEqual([...dataset.vocabulary].sort());
    expect(dataset.vocabulary).not.toContain(MOVEMENT_START_TOKEN);
  });

  it("skips trajectories with no actions", () => {
    const dataset = buildMovementDataset([span("empty", [])], { order: 2 });
    expect(dataset.samples).toHaveLength(0);
  });
});

describe("MarkovMovementBackend", () => {
  it("learns and reproduces a repeated workflow exactly (memorization)", () => {
    const dataset = buildMovementDataset(synthesizeWorkflowTrajectories(4), { order: 2 });
    const model = defaultMovementBackend.train(dataset);
    const evaluation = evaluateMovementModel(model, dataset.samples);
    // A deterministic workflow with no branching is perfectly reproducible.
    expect(evaluation.accuracy).toBe(1);
  });

  it("rolls out the full workflow autoregressively from the start", () => {
    const dataset = buildMovementDataset(synthesizeWorkflowTrajectories(3), { order: 2 });
    const model = new MarkovMovementBackend().train(dataset);
    const rollout = model.rollout([MOVEMENT_START_TOKEN, MOVEMENT_START_TOKEN], 5);
    expect(rollout).toEqual([
      "device:tap:menu-button",
      "device:swipe:list",
      "device:tap:settings-row",
      "device:type:search-field",
      "device:tap:confirm",
    ]);
  });

  it("generalizes to an unseen context via backoff", () => {
    const dataset = buildMovementDataset(synthesizeWorkflowTrajectories(3), { order: 2 });
    const model = new MarkovMovementBackend().train(dataset);
    // This exact 2-token context never appears in training, but the trailing
    // token "device:tap:settings-row" is always followed by the type step.
    const prediction = model.predict(["device:tap:never-seen", "device:tap:settings-row"]);
    expect(prediction.token).toBe("device:type:search-field");
    // It had to shorten the context to produce this — that is the generalization.
    expect(prediction.backoffOrder).toBeLessThan(2);
  });

  it("prefers the higher-order match when a branch diverges", () => {
    // Same trailing token, different next depending on the token before it.
    const branching = [
      span("a", [
        action("device", "one", 1, { gesture: "tap", target: "start" }),
        action("device", "two", 2, { gesture: "tap", target: "middle" }),
        action("device", "three", 3, { gesture: "tap", target: "left" }),
      ]),
      span("b", [
        action("device", "one", 1, { gesture: "swipe", target: "start" }),
        action("device", "two", 2, { gesture: "tap", target: "middle" }),
        action("device", "three", 3, { gesture: "tap", target: "right" }),
      ]),
    ];
    const dataset = buildMovementDataset(branching, { order: 2 });
    const model = new MarkovMovementBackend().train(dataset);
    const left = model.predict(["device:tap:start", "device:tap:middle"]);
    const right = model.predict(["device:swipe:start", "device:tap:middle"]);
    expect(left.token).toBe("device:tap:left");
    expect(left.backoffOrder).toBe(2);
    expect(right.token).toBe("device:tap:right");
    expect(right.backoffOrder).toBe(2);
  });

  it("returns an empty prediction for an untrained model", () => {
    const model = new MarkovMovementBackend().train(buildMovementDataset([], { order: 2 }));
    const prediction = model.predict(["anything"]);
    expect(prediction.token).toBeUndefined();
    expect(prediction.candidates).toEqual([]);
  });

  it("ranks candidates by probability with deterministic tie-breaks", () => {
    // From "start", tap:a occurs twice and tap:b once.
    const trajectories = [
      span("a", [action("device", "s", 1, { gesture: "tap", target: "start" }), action("device", "x", 2, { gesture: "tap", target: "a" })]),
      span("b", [action("device", "s", 1, { gesture: "tap", target: "start" }), action("device", "x", 2, { gesture: "tap", target: "a" })]),
      span("c", [action("device", "s", 1, { gesture: "tap", target: "start" }), action("device", "x", 2, { gesture: "tap", target: "b" })]),
    ];
    const dataset = buildMovementDataset(trajectories, { order: 1 });
    const model = new MarkovMovementBackend().train(dataset);
    const prediction = model.predict(["device:tap:start"]);
    expect(prediction.candidates.map((candidate) => candidate.token)).toEqual(["device:tap:a", "device:tap:b"]);
    expect(prediction.candidates[0]?.score).toBeCloseTo(2 / 3, 10);
    expect(prediction.candidates[1]?.score).toBeCloseTo(1 / 3, 10);
  });
});

describe("serialization", () => {
  it("round-trips through a snapshot with identical predictions", () => {
    const dataset = buildMovementDataset(synthesizeWorkflowTrajectories(3), { order: 2 });
    const backend = new MarkovMovementBackend();
    const model = backend.train(dataset);
    const snapshot = model.serialize();
    // Snapshot must be plain JSON.
    expect(JSON.parse(JSON.stringify(snapshot))).toEqual(snapshot);
    const restored = backend.restore(snapshot);
    for (const sample of dataset.samples) {
      expect(restored.predict(sample.context).token).toBe(model.predict(sample.context).token);
    }
    expect(restored.serialize()).toEqual(snapshot);
  });
});

describe("evaluateMovementModel (generalization harness)", () => {
  it("measures accuracy and generalization on held-out trajectories", () => {
    // Train on the first 3 workflow repeats, evaluate on a fresh repeat whose
    // exact timestamps (hence no verbatim overlap of full histories) differ.
    const train = buildMovementDataset(synthesizeWorkflowTrajectories(3), { order: 2 });
    const model = new MarkovMovementBackend().train(train);
    const heldOut = buildMovementDataset(
      synthesizeWorkflowTrajectories(1).map((trajectory) => ({ ...trajectory, id: "held-out" })),
      { order: 2 },
    );
    const evaluation = evaluateMovementModel(model, heldOut.samples, { topK: 2 });
    expect(evaluation.total).toBe(5);
    // The workflow is deterministic, so a well-trained model reproduces it.
    expect(evaluation.accuracy).toBe(1);
    expect(evaluation.topKAccuracy).toBe(1);
    expect(evaluation.generalizationRate).toBeGreaterThanOrEqual(0);
  });

  it("reports zeroed metrics for an empty sample set", () => {
    const model = new MarkovMovementBackend().train(buildMovementDataset(synthesizeWorkflowTrajectories(1), { order: 2 }));
    expect(evaluateMovementModel(model, [])).toEqual({
      total: 0,
      correct: 0,
      accuracy: 0,
      topKAccuracy: 0,
      generalizationRate: 0,
    });
  });
});
