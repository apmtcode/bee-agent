import { describe, expect, it } from "vitest";
import type { ReplayManifest } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  StatisticalMovementBackend,
  actionToken,
  buildMovementDatasetFromReplays,
  buildMovementDatasetFromTrajectories,
  evaluateMovementModel,
} from "./movement-model.js";
import {
  createRng,
  generateSyntheticTrajectories,
  type SyntheticMovementRecipe,
} from "./synthetic-movements.js";

function trajectory(
  id: string,
  steps: Array<{ ts: number; observation: string; source?: string; tool: string; summary: string }>,
): TrajectorySpan {
  return {
    id,
    sessionId: `${id}-session`,
    createdAt: "2026-01-01T00:00:00.000Z",
    captureTier: "full",
    observations: steps.map((step) => ({
      kind: "observation",
      source: step.source ?? "screen",
      summary: step.observation,
      ts: step.ts,
    })),
    actions: steps.map((step) => ({
      kind: "action",
      tool: step.tool,
      summary: step.summary,
      ts: step.ts + 1,
    })),
    outcome: { status: "success", summary: "ok" },
  };
}

const SAVE_RECIPE: SyntheticMovementRecipe = {
  name: "save-document",
  slots: { doc: ["invoice", "report", "letter", "memo", "contract", "budget"] },
  steps: [
    { observation: "editor open with {doc} document", tool: "mouse.move", summary: "move cursor to file menu" },
    { observation: "file menu highlighted for {doc}", tool: "mouse.click", summary: "click file menu" },
    { observation: "file menu expanded for {doc}", tool: "mouse.click", summary: "click save entry" },
    { observation: "save dialog for {doc}", tool: "keyboard.type", summary: "confirm save" },
  ],
};

describe("movement dataset construction", () => {
  it("pairs each action with its preceding observation and the previous action", () => {
    const dataset = buildMovementDatasetFromTrajectories([
      trajectory("t1", [
        { ts: 10, observation: "button visible", tool: "mouse.click", summary: "click button" },
        { ts: 20, observation: "field focused", tool: "keyboard.type", summary: "type name" },
      ]),
    ]);
    expect(dataset.samples).toHaveLength(1);
    const [first, second] = dataset.samples[0]!.steps;
    expect(first!.context.observation).toBe("button visible");
    expect(first!.context.previousAction).toBeUndefined();
    expect(first!.action).toBe(actionToken("mouse.click", "click button"));
    expect(second!.context.observation).toBe("field focused");
    expect(second!.context.previousAction).toBe("mouse.click");
  });

  it("builds an equivalent dataset from a replay manifest", () => {
    const replay: ReplayManifest = {
      version: 1,
      sessionId: "s1",
      trajectoryIds: ["t1"],
      eventCount: 4,
      events: [
        { kind: "observation", ts: 10, trajectoryId: "t1", source: "screen", summary: "button visible" },
        { kind: "action", ts: 11, trajectoryId: "t1", tool: "mouse.click", summary: "click button" },
        { kind: "observation", ts: 20, trajectoryId: "t1", source: "screen", summary: "field focused" },
        { kind: "action", ts: 21, trajectoryId: "t1", tool: "keyboard.type", summary: "type name" },
      ],
    };
    const dataset = buildMovementDatasetFromReplays([replay]);
    expect(dataset.samples).toHaveLength(1);
    expect(dataset.samples[0]!.steps.map((step) => step.tool)).toEqual(["mouse.click", "keyboard.type"]);
    expect(dataset.samples[0]!.steps[1]!.context.previousAction).toBe("mouse.click");
  });
});

describe("StatisticalMovementBackend training + inference", () => {
  it("reproduces recorded movements exactly on the training set", () => {
    const trajectories = generateSyntheticTrajectories(SAVE_RECIPE, { count: 12, seed: 1 });
    const dataset = buildMovementDatasetFromTrajectories(trajectories);
    const model = new StatisticalMovementBackend().train(dataset);

    const report = evaluateMovementModel(model, dataset);
    expect(report.stepCount).toBeGreaterThan(0);
    expect(report.fidelity).toBe(1);
    // Every recorded step should be reproduced (some via exact memory, some via
    // token similarity when two variants share an identical context key).
    expect(report.byStrategy.exact + report.byStrategy.similar).toBe(report.stepCount);
  });

  it("generalizes to held-out but related movements (unseen slot values)", () => {
    // Train only on the first four documents; evaluate on two documents the model
    // has never seen — the environment differs but the movement is the same.
    const trainDocs = ["invoice", "report", "letter", "memo"];
    const holdoutDocs = ["contract", "budget"];
    const trainTrajectories = generateSyntheticTrajectories(SAVE_RECIPE, {
      count: 16,
      seed: 7,
      idPrefix: "train",
      slotFilter: (slot, values) => (slot === "doc" ? values.filter((v) => trainDocs.includes(v)) : values),
    });
    const holdoutTrajectories = generateSyntheticTrajectories(SAVE_RECIPE, {
      count: 6,
      seed: 99,
      idPrefix: "holdout",
      slotFilter: (slot, values) => (slot === "doc" ? values.filter((v) => holdoutDocs.includes(v)) : values),
    });

    const model = new StatisticalMovementBackend().train(
      buildMovementDatasetFromTrajectories(trainTrajectories),
    );
    const holdout = buildMovementDatasetFromTrajectories(holdoutTrajectories);
    const report = evaluateMovementModel(model, holdout);

    // The action summaries are stable across documents, so the model should recover
    // the correct movement for unseen contexts via the similarity backoff.
    expect(report.toolFidelity).toBe(1);
    expect(report.fidelity).toBe(1);
    expect(report.byStrategy.similar).toBeGreaterThan(0);
    expect(report.byStrategy.exact).toBe(0);
  });

  it("rollout threads predicted actions forward from observations only", () => {
    const trajectories = generateSyntheticTrajectories(SAVE_RECIPE, { count: 10, seed: 3 });
    const model = new StatisticalMovementBackend().train(
      buildMovementDatasetFromTrajectories(trajectories),
    );
    const predictions = model.rollout([
      { observation: "editor open with spreadsheet document", observationSource: "screen" },
      { observation: "file menu highlighted for spreadsheet", observationSource: "screen" },
      { observation: "file menu expanded for spreadsheet", observationSource: "screen" },
      { observation: "save dialog for spreadsheet", observationSource: "screen" },
    ]);
    expect(predictions.map((p) => p.tool)).toEqual([
      "mouse.move",
      "mouse.click",
      "mouse.click",
      "keyboard.type",
    ]);
  });

  it("falls back to the action→action transition then the global prior", () => {
    const dataset = buildMovementDatasetFromTrajectories([
      trajectory("t1", [
        { ts: 10, observation: "alpha", tool: "mouse.click", summary: "click" },
        { ts: 20, observation: "beta", tool: "keyboard.type", summary: "type" },
      ]),
    ]);
    const model = new StatisticalMovementBackend().train(dataset);
    // Unknown observation with no shared tokens, but a known previous action.
    const transition = model.predict({ observation: "zzz-unseen-9999", previousAction: "mouse.click" });
    expect(transition?.strategy).toBe("transition");
    expect(transition?.tool).toBe("keyboard.type");
    // No observation, no previous action → global prior.
    const prior = model.predict({});
    expect(prior?.strategy).toBe("prior");
  });

  it("returns undefined for an empty model", () => {
    const model = new StatisticalMovementBackend().train({ version: 1, samples: [] });
    expect(model.predict({ observation: "anything" })).toBeUndefined();
  });
});

describe("snapshot round-trip", () => {
  it("reloads to an identical model", () => {
    const trajectories = generateSyntheticTrajectories(SAVE_RECIPE, { count: 8, seed: 5 });
    const dataset = buildMovementDatasetFromTrajectories(trajectories);
    const backend = new StatisticalMovementBackend();
    const model = backend.train(dataset);
    const snapshot = model.snapshot();

    // Snapshot must be plain-JSON serializable.
    const roundTripped = backend.load(JSON.parse(JSON.stringify(snapshot)));
    expect(roundTripped.snapshot()).toEqual(snapshot);

    const context = { observation: "save dialog for invoice" };
    expect(roundTripped.predict(context)).toEqual(model.predict(context));
  });

  it("produces a deterministic snapshot regardless of sample order", () => {
    const trajectories = generateSyntheticTrajectories(SAVE_RECIPE, { count: 6, seed: 2 });
    const forward = new StatisticalMovementBackend()
      .train(buildMovementDatasetFromTrajectories(trajectories))
      .snapshot();
    const reversed = new StatisticalMovementBackend()
      .train(buildMovementDatasetFromTrajectories([...trajectories].reverse()))
      .snapshot();
    expect(reversed).toEqual(forward);
  });
});

describe("synthetic generator determinism", () => {
  it("is stable for a fixed seed and varies across seeds", () => {
    const a = generateSyntheticTrajectories(SAVE_RECIPE, { count: 4, seed: 42 });
    const b = generateSyntheticTrajectories(SAVE_RECIPE, { count: 4, seed: 42 });
    expect(b).toEqual(a);
    const c = generateSyntheticTrajectories(SAVE_RECIPE, { count: 4, seed: 43 });
    expect(c).not.toEqual(a);
  });

  it("createRng yields values in [0,1)", () => {
    const rng = createRng(123);
    for (let i = 0; i < 100; i += 1) {
      const value = rng();
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThan(1);
    }
  });
});
