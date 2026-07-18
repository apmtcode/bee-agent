import { describe, expect, it } from "vitest";
import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  DEFAULT_MOVEMENT_PATTERNS,
  MovementPolicyRunner,
  NearestNeighborPolicyBackend,
  buildPolicyExamples,
  buildPolicyExamplesFromTrajectory,
  evaluateMovementPolicy,
  generateSyntheticTrajectories,
  observationFeatureVector,
  type PolicyTrainingExample,
} from "./movement-policy.js";

function span(id: string, steps: Array<{ source: string; summary: string; tool: string; actionSummary: string }>): TrajectorySpan {
  let ts = 100;
  const observations = [] as TrajectorySpan["observations"];
  const actions = [] as TrajectorySpan["actions"];
  for (const step of steps) {
    observations.push({ kind: "observation", source: step.source, summary: step.summary, ts });
    ts += 1;
    actions.push({ kind: "action", tool: step.tool, summary: step.actionSummary, ts });
    ts += 1;
  }
  return {
    id,
    sessionId: "s1",
    createdAt: new Date(0).toISOString(),
    captureTier: "operator",
    observations,
    actions,
  };
}

describe("observationFeatureVector", () => {
  it("produces an L2-normalized bag-of-tokens vector", () => {
    const vector = observationFeatureVector({ source: "ui.pointer", summary: "click the save button" });
    const magnitude = Math.sqrt(Object.values(vector).reduce((sum, value) => sum + value * value, 0));
    expect(magnitude).toBeCloseTo(1, 10);
    expect(vector.click).toBeGreaterThan(0);
    expect(vector.save).toBeGreaterThan(0);
  });

  it("returns an empty vector for token-free observations", () => {
    expect(observationFeatureVector({ source: "", summary: "!!! ---" })).toEqual({});
  });
});

describe("buildPolicyExamplesFromTrajectory", () => {
  it("pairs each action with the most recent preceding observation", () => {
    const trajectory = span("t1", [
      { source: "ui.pointer", summary: "click save button", tool: "click", actionSummary: "click save" },
      { source: "ui.keyboard", summary: "type into name field", tool: "type", actionSummary: "type name" },
    ]);
    const examples = buildPolicyExamplesFromTrajectory(trajectory);
    expect(examples).toHaveLength(2);
    expect(examples[0]?.observation.summary).toBe("click save button");
    expect(examples[0]?.action.tool).toBe("click");
    expect(examples[1]?.action.tool).toBe("type");
    expect(examples[0]?.trajectoryId).toBe("t1");
  });

  it("prefers reviewed/redacted observations and actions when present", () => {
    const trajectory = span("t2", [
      { source: "ui.pointer", summary: "raw secret click", tool: "click", actionSummary: "raw" },
    ]);
    trajectory.review = {
      status: "approved",
      reviewedAt: new Date(0).toISOString(),
      reviewedBy: "reviewer",
      redactedObservations: [{ ts: 100, source: "ui.pointer", summary: "click redacted button" }],
      redactedActions: [{ ts: 101, tool: "click", summary: "click redacted" }],
    };
    const [example] = buildPolicyExamplesFromTrajectory(trajectory);
    expect(example?.observation.summary).toBe("click redacted button");
    expect(example?.action.summary).toBe("click redacted");
  });

  it("returns no examples when observations or actions are absent", () => {
    const empty: TrajectorySpan = {
      id: "empty",
      sessionId: "s1",
      createdAt: new Date(0).toISOString(),
      captureTier: "operator",
      observations: [],
      actions: [],
    };
    expect(buildPolicyExamplesFromTrajectory(empty)).toEqual([]);
  });
});

describe("NearestNeighborPolicyBackend", () => {
  const examples: PolicyTrainingExample[] = [
    { observation: { source: "ui.pointer", summary: "click the save button" }, action: { tool: "click", summary: "click save" } },
    { observation: { source: "ui.keyboard", summary: "type text into the name field" }, action: { tool: "type", summary: "type name" } },
    { observation: { source: "ui.window", summary: "open the settings window" }, action: { tool: "open", summary: "open settings" } },
  ];

  it("replays a recorded movement exactly for a familiar observation", () => {
    const model = new NearestNeighborPolicyBackend().train(examples);
    const prediction = model.predict({ source: "ui.pointer", summary: "click the save button" });
    expect(prediction.action.tool).toBe("click");
    expect(prediction.source).toBe("exact");
    expect(prediction.confidence).toBeCloseTo(1, 10);
  });

  it("generalizes to a novel but related observation", () => {
    const model = new NearestNeighborPolicyBackend().train(examples);
    // Never-seen target noun ("submit"), same pattern tokens as the click example.
    const prediction = model.predict({ source: "ui.pointer", summary: "click the submit button" });
    expect(prediction.action.tool).toBe("click");
    expect(prediction.source).toBe("generalized");
    expect(prediction.confidence).toBeGreaterThan(0);
    expect(prediction.confidence).toBeLessThan(1);
  });

  it("falls back to the modal action when there is no token overlap", () => {
    const model = new NearestNeighborPolicyBackend().train(examples);
    const prediction = model.predict({ source: "", summary: "zzz qqq" });
    expect(prediction.source).toBe("fallback");
    expect(prediction.confidence).toBe(0);
    expect(prediction.matchedEntryIndex).toBe(-1);
  });

  it("round-trips through a serialized snapshot", () => {
    const backend = new NearestNeighborPolicyBackend();
    const model = backend.train(examples);
    const snapshot = JSON.parse(JSON.stringify(model.toJSON()));
    const restored = backend.restore(snapshot);
    expect(restored.exampleCount).toBe(model.exampleCount);
    const before = model.predict({ source: "ui.keyboard", summary: "type text into the email field" });
    const after = restored.predict({ source: "ui.keyboard", summary: "type text into the email field" });
    expect(after.action.tool).toBe(before.action.tool);
    expect(after.confidence).toBeCloseTo(before.confidence, 10);
  });
});

describe("MovementPolicyRunner", () => {
  it("trains from trajectories and predicts a movement sequence", () => {
    const runner = new MovementPolicyRunner();
    const trajectory = span("t1", [
      { source: "ui.pointer", summary: "click the save button", tool: "click", actionSummary: "click save" },
      { source: "ui.keyboard", summary: "type text into the name field", tool: "type", actionSummary: "type name" },
    ]);
    const model = runner.trainFromTrajectories([trajectory]);
    expect(runner.backendName).toBe("nearest-neighbor");
    const predictions = runner.predictSequence(model, [
      { source: "ui.pointer", summary: "click the save button" },
      { source: "ui.keyboard", summary: "type text into the email field" },
    ]);
    expect(predictions.map((prediction) => prediction.action.tool)).toEqual(["click", "type"]);
  });
});

describe("generateSyntheticTrajectories", () => {
  it("is deterministic for a fixed seed", () => {
    const a = generateSyntheticTrajectories({ count: 3, seed: 42 });
    const b = generateSyntheticTrajectories({ count: 3, seed: 42 });
    expect(a).toEqual(b);
  });

  it("produces distinct streams for different seeds", () => {
    const a = generateSyntheticTrajectories({ count: 5, seed: 1 });
    const b = generateSyntheticTrajectories({ count: 5, seed: 2 });
    expect(a).not.toEqual(b);
  });

  it("emits well-formed observation/action pairs", () => {
    const [trajectory] = generateSyntheticTrajectories({ count: 1, seed: 7, actionsPerTrajectory: 4 });
    expect(trajectory?.observations).toHaveLength(4);
    expect(trajectory?.actions).toHaveLength(4);
    const tools = new Set(DEFAULT_MOVEMENT_PATTERNS.map((pattern) => pattern.tool));
    for (const action of trajectory?.actions ?? []) {
      expect(tools.has(action.tool)).toBe(true);
    }
  });
});

describe("evaluateMovementPolicy (generalization harness)", () => {
  it("recovers recorded movements with perfect fidelity on the training set", () => {
    const trajectories = generateSyntheticTrajectories({ count: 20, seed: 11, actionsPerTrajectory: 4 });
    const runner = new MovementPolicyRunner();
    const model = runner.trainFromTrajectories(trajectories);
    const evaluation = evaluateMovementPolicy(model, buildPolicyExamples(trajectories));
    expect(evaluation.toolAccuracy).toBe(1);
    expect(evaluation.total).toBe(80);
  });

  it("generalizes to held-out target nouns of the same movement patterns", () => {
    // Train on one set of targets; evaluate on entirely unseen targets that
    // share each pattern's keyword tokens -> the policy must generalize.
    const trainTrajectories = generateSyntheticTrajectories({
      count: 40,
      seed: 3,
      actionsPerTrajectory: 4,
      targets: ["save", "submit", "cancel"],
    });
    const heldOut = generateSyntheticTrajectories({
      count: 20,
      seed: 99,
      actionsPerTrajectory: 4,
      targets: ["export", "publish", "archive"],
      startTs: 9_000_000,
    });

    const runner = new MovementPolicyRunner();
    const model = runner.trainFromTrajectories(trainTrajectories);
    const evaluation = evaluateMovementPolicy(model, buildPolicyExamples(heldOut));

    // Held-out targets were never seen, so every correct call is a generalization.
    expect(evaluation.generalizedMatches).toBeGreaterThan(0);
    expect(evaluation.toolAccuracy).toBe(1);
    expect(evaluation.generalizationAccuracy).toBe(1);
    expect(evaluation.exactMatches).toBe(0);
  });

  it("reports zero accuracy on an empty eval set without dividing by zero", () => {
    const runner = new MovementPolicyRunner();
    const model = runner.trainFromExamples([]);
    const evaluation = evaluateMovementPolicy(model, []);
    expect(evaluation.toolAccuracy).toBe(0);
    expect(evaluation.generalizationAccuracy).toBe(0);
    expect(evaluation.averageConfidence).toBe(0);
  });
});
