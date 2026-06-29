import { describe, expect, it } from "vitest";
import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  NearestNeighborMovementBackend,
  deriveMovementDataset,
  deriveMovementExamples,
  evaluateMovementModel,
  jaccard,
  replayTrajectory,
  tokenizeContext,
  trainMovementModel,
} from "./movement-model.js";

function trajectory(
  id: string,
  steps: { obs?: { source: string; summary: string }; act?: { tool: string; summary: string } }[],
  overrides: Partial<TrajectorySpan> = {},
): TrajectorySpan {
  const observations = [];
  const actions = [];
  let ts = 0;
  for (const step of steps) {
    ts += 1;
    if (step.obs) {
      observations.push({ kind: "observation" as const, ts, source: step.obs.source, summary: step.obs.summary });
    }
    if (step.act) {
      actions.push({ kind: "action" as const, ts, tool: step.act.tool, summary: step.act.summary });
    }
  }
  return {
    id,
    sessionId: `session-${id}`,
    createdAt: "2026-06-29T00:00:00.000Z",
    captureTier: "full",
    observations,
    actions,
    ...overrides,
  };
}

// A small synthetic "open the file menu then save" movement, recorded twice
// with slightly different wording so the model has related-but-not-identical
// contexts to generalize over.
const openAndSave = trajectory("open-save", [
  { obs: { source: "screen", summary: "editor window focused with unsaved document" } },
  { act: { tool: "mouse.click", summary: "click File menu in the top bar" } },
  { obs: { source: "screen", summary: "File menu open showing Save entry" } },
  { act: { tool: "mouse.click", summary: "click Save in the File menu" } },
  { obs: { source: "screen", summary: "save dialog requesting a filename" } },
  { act: { tool: "keyboard.type", summary: "type the document filename" } },
]);

describe("deriveMovementExamples", () => {
  it("emits one example per action with the preceding observations and actions", () => {
    const examples = deriveMovementExamples(openAndSave);
    expect(examples).toHaveLength(3);

    // First decision: only the first observation is visible, no prior actions.
    expect(examples[0]?.context.observations).toEqual([
      { source: "screen", summary: "editor window focused with unsaved document" },
    ]);
    expect(examples[0]?.context.recentActions).toEqual([]);
    expect(examples[0]?.action.tool).toBe("mouse.click");

    // Third decision sees all three prior observations and two prior actions.
    expect(examples[2]?.context.observations).toHaveLength(3);
    expect(examples[2]?.context.recentActions).toHaveLength(2);
    expect(examples[2]?.action).toEqual({ tool: "keyboard.type", summary: "type the document filename" });
    expect(examples[2]?.trajectoryId).toBe("open-save");
  });

  it("respects observation and action windows", () => {
    const examples = deriveMovementExamples(openAndSave, { observationWindow: 1, actionWindow: 1 });
    expect(examples[2]?.context.observations).toHaveLength(1);
    expect(examples[2]?.context.recentActions).toHaveLength(1);
  });

  it("can filter to approved trajectories only", () => {
    const pending = deriveMovementExamples(openAndSave, { approvedOnly: true });
    expect(pending).toHaveLength(0);

    const approved = deriveMovementExamples(
      trajectory(
        "approved",
        [{ obs: { source: "screen", summary: "a" }, act: { tool: "mouse.click", summary: "b" } }],
        { review: { status: "approved", reviewedAt: "2026-06-29T00:00:00.000Z", reviewedBy: "tester" } },
      ),
      { approvedOnly: true },
    );
    expect(approved).toHaveLength(1);
  });
});

describe("tokenizeContext / jaccard", () => {
  it("namespaces tokens so observation words never collide with tool names", () => {
    const tokens = tokenizeContext({
      observations: [{ source: "screen", summary: "click here" }],
      recentActions: [{ tool: "mouse.click", summary: "done" }],
    });
    expect(tokens.has("obs:click")).toBe(true);
    expect(tokens.has("act.tool:mouse.click")).toBe(true);
    // The observation "click" and the tool "click" are distinct features.
    expect(tokens.has("obs:click") && tokens.has("act.tool:mouse.click")).toBe(true);
  });

  it("computes a symmetric similarity in [0, 1]", () => {
    const a = new Set(["x", "y", "z"]);
    const b = new Set(["x", "y", "w"]);
    expect(jaccard(a, b)).toBeCloseTo(2 / 4);
    expect(jaccard(a, b)).toBe(jaccard(b, a));
    expect(jaccard(a, a)).toBe(1);
    expect(jaccard(new Set(), new Set())).toBe(1);
    expect(jaccard(new Set(["x"]), new Set(["y"]))).toBe(0);
  });
});

describe("NearestNeighborMovementBackend", () => {
  it("recalls recorded movements exactly (objective 2c)", async () => {
    const model = await trainMovementModel(new NearestNeighborMovementBackend(), [openAndSave]);
    expect(model.backend).toBe("nearest-neighbor");
    expect(model.exampleCount).toBe(3);

    const replay = replayTrajectory(model, openAndSave);
    expect(replay.every((step) => step.match)).toBe(true);
    expect(replay.every((step) => step.predicted.source === "recall")).toBe(true);
    expect(replay.every((step) => step.predicted.confidence === 1)).toBe(true);
  });

  it("generalizes to a new-but-related context (objective 2d)", async () => {
    const model = await trainMovementModel(new NearestNeighborMovementBackend(), [openAndSave]);

    // Unseen wording, same situation as the first recorded decision point.
    const prediction = model.predict({
      observations: [{ source: "screen", summary: "code editor focused, document has unsaved edits" }],
      recentActions: [],
    });
    expect(prediction.tool).toBe("mouse.click");
    expect(prediction.summary).toBe("click File menu in the top bar");
    expect(prediction.source).toBe("generalize");
    expect(prediction.confidence).toBeGreaterThan(0);
    expect(prediction.confidence).toBeLessThan(1);
  });

  it("falls back to the most frequent action when nothing overlaps", async () => {
    const frequentClicks = trajectory("clicky", [
      { obs: { source: "screen", summary: "alpha" }, act: { tool: "mouse.click", summary: "tap alpha" } },
      { obs: { source: "screen", summary: "beta" }, act: { tool: "mouse.click", summary: "tap beta" } },
      { obs: { source: "screen", summary: "gamma" }, act: { tool: "keyboard.type", summary: "write" } },
    ]);
    const model = await trainMovementModel(new NearestNeighborMovementBackend(), [frequentClicks]);

    const prediction = model.predict({
      observations: [{ source: "audio", summary: "completely unrelated zzz qqq" }],
      recentActions: [],
    });
    expect(prediction.source).toBe("fallback");
    expect(prediction.tool).toBe("mouse.click");
    expect(prediction.confidence).toBe(0);
  });

  it("returns a 'none' prediction when untrained", async () => {
    const model = await new NearestNeighborMovementBackend().train([]);
    const prediction = model.predict({ observations: [], recentActions: [] });
    expect(prediction.source).toBe("none");
    expect(prediction.tool).toBe("");
  });

  it("is deterministic across retrains", async () => {
    const dataset = deriveMovementDataset([openAndSave]);
    const a = await new NearestNeighborMovementBackend().train(dataset);
    const b = await new NearestNeighborMovementBackend().train(dataset);
    const context = { observations: [{ source: "screen", summary: "save dialog asking for name" }], recentActions: [] };
    expect(a.predict(context)).toEqual(b.predict(context));
  });
});

describe("evaluateMovementModel", () => {
  it("scores held-out generalization and reports per-source breakdown", async () => {
    const train = trajectory("t1", [
      { obs: { source: "screen", summary: "login form with username field" } },
      { act: { tool: "mouse.click", summary: "click the username field" } },
      { obs: { source: "screen", summary: "username field focused" } },
      { act: { tool: "keyboard.type", summary: "type the username" } },
    ]);
    // Held-out trajectory: same task, paraphrased observations/actions.
    const heldOut = trajectory("t2", [
      { obs: { source: "screen", summary: "sign in form showing the username box" } },
      { act: { tool: "mouse.click", summary: "click the username field" } },
      { obs: { source: "screen", summary: "username box is now focused" } },
      { act: { tool: "keyboard.type", summary: "type the username" } },
    ]);

    const model = await trainMovementModel(new NearestNeighborMovementBackend(), [train]);
    const evaluation = evaluateMovementModel(model, deriveMovementDataset([heldOut]));

    expect(evaluation.total).toBe(2);
    expect(evaluation.toolAccuracy).toBe(1);
    expect(evaluation.exactAccuracy).toBe(1);
    expect(evaluation.bySource.generalize).toBeGreaterThan(0);
    expect(evaluation.bySource.recall).toBe(0);
    expect(evaluation.meanConfidence).toBeGreaterThan(0);
    expect(evaluation.meanConfidence).toBeLessThanOrEqual(1);
  });

  it("returns zeroed metrics for an empty eval set", async () => {
    const model = await trainMovementModel(new NearestNeighborMovementBackend(), [openAndSave]);
    const evaluation = evaluateMovementModel(model, []);
    expect(evaluation).toMatchObject({ total: 0, exactAccuracy: 0, toolAccuracy: 0, meanConfidence: 0 });
  });
});
