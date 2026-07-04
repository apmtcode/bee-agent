import { describe, expect, it } from "vitest";
import {
  NearestContextMovementBackend,
  buildMovementDataset,
  type MovementDataset,
  type MovementStep,
} from "./movement-model.js";

const openReport: MovementStep[] = [
  { kind: "observation", source: "screen", summary: "Finder is focused" },
  { kind: "action", tool: "click", summary: "Open file report.txt" },
  { kind: "action", tool: "type", summary: "Save report.txt to disk" },
];

const openBudget: MovementStep[] = [
  { kind: "observation", source: "screen", summary: "Editor is focused" },
  { kind: "action", tool: "click", summary: "Delete row in budget.csv" },
];

const dataset: MovementDataset = {
  examples: [
    { trajectoryId: "traj-report", context: "open file report.txt in editor", steps: openReport },
    { trajectoryId: "traj-budget", context: "delete a row in budget.csv spreadsheet", steps: openBudget },
  ],
};

describe("NearestContextMovementBackend", () => {
  it("repeats a recorded movement verbatim when the context matches exactly (objective 2c)", () => {
    const model = new NearestContextMovementBackend().train(dataset);
    const prediction = model.predict("open file report.txt in editor");

    expect(prediction.generalized).toBe(false);
    expect(prediction.sourceTrajectoryId).toBe("traj-report");
    expect(prediction.confidence).toBe(1);
    expect(prediction.steps).toEqual(openReport);
  });

  it("generalizes to a new-but-related context by substituting the varying slot (objective 2d)", () => {
    const model = new NearestContextMovementBackend().train(dataset);
    const prediction = model.predict("open file invoice.txt in editor");

    expect(prediction.generalized).toBe(true);
    expect(prediction.sourceTrajectoryId).toBe("traj-report");
    expect(prediction.confidence).toBeGreaterThan(0);
    expect(prediction.confidence).toBeLessThan(1);
    expect(prediction.substitutions).toEqual([{ from: "report.txt", to: "invoice.txt" }]);
    // The recorded steps are adapted: report.txt → invoice.txt everywhere it appears.
    expect(prediction.steps).toEqual([
      { kind: "observation", source: "screen", summary: "Finder is focused" },
      { kind: "action", tool: "click", summary: "Open file invoice.txt" },
      { kind: "action", tool: "type", summary: "Save invoice.txt to disk" },
    ]);
  });

  it("selects the nearest trajectory when several are recorded", () => {
    const model = new NearestContextMovementBackend().train(dataset);
    const prediction = model.predict("delete a row in taxes.csv spreadsheet");

    expect(prediction.sourceTrajectoryId).toBe("traj-budget");
    // budget.csv → taxes.csv is substituted into the recorded action step.
    expect(prediction.steps[1]).toEqual({ kind: "action", tool: "click", summary: "Delete row in taxes.csv" });
  });

  it("is deterministic: same dataset and query yield identical predictions", () => {
    const backend = new NearestContextMovementBackend();
    const a = backend.train(dataset).predict("open file invoice.txt in editor");
    const b = backend.train(dataset).predict("open file invoice.txt in editor");
    expect(a).toEqual(b);
  });

  it("returns an empty prediction for an unrelated context", () => {
    const model = new NearestContextMovementBackend().train(dataset);
    const prediction = model.predict("quantum teleportation experiment");

    expect(prediction.steps).toEqual([]);
    expect(prediction.sourceTrajectoryId).toBeNull();
    expect(prediction.confidence).toBe(0);
  });

  it("handles an empty dataset without throwing", () => {
    const model = new NearestContextMovementBackend().train({ examples: [] });
    expect(model.exampleCount).toBe(0);
    const prediction = model.predict("anything");
    expect(prediction.steps).toEqual([]);
    expect(prediction.sourceTrajectoryId).toBeNull();
  });
});

describe("buildMovementDataset", () => {
  it("derives one example per trajectory from replay events, using user transcript as context", () => {
    const replays = [
      {
        events: [
          { kind: "transcript" as const, ts: 1, role: "user", content: "open the settings panel" },
          { kind: "transcript" as const, ts: 2, role: "assistant", content: "on it" },
          { kind: "observation" as const, ts: 3, trajectoryId: "t1", source: "screen", summary: "menu visible" },
          { kind: "action" as const, ts: 4, trajectoryId: "t1", tool: "click", summary: "click Settings" },
          { kind: "action" as const, ts: 5, trajectoryId: "t2", tool: "scroll", summary: "scroll down" },
        ],
      },
    ];

    const dataset = buildMovementDataset(replays);
    expect(dataset.examples).toHaveLength(2);

    const t1 = dataset.examples.find((example) => example.trajectoryId === "t1");
    expect(t1?.context).toBe("open the settings panel");
    expect(t1?.steps).toEqual([
      { kind: "observation", source: "screen", summary: "menu visible" },
      { kind: "action", tool: "click", summary: "click Settings" },
    ]);
  });

  it("round-trips through the backend: a dataset built from replays replays verbatim", () => {
    const replays = [
      {
        events: [
          { kind: "transcript" as const, ts: 1, role: "user", content: "rename the active tab" },
          { kind: "action" as const, ts: 2, trajectoryId: "t1", tool: "doubleclick", summary: "double-click tab" },
          { kind: "action" as const, ts: 3, trajectoryId: "t1", tool: "type", summary: "type new name" },
        ],
      },
    ];

    const dataset = buildMovementDataset(replays);
    const model = new NearestContextMovementBackend().train(dataset);
    const prediction = model.predict("rename the active tab");

    expect(prediction.generalized).toBe(false);
    expect(prediction.steps).toEqual([
      { kind: "action", tool: "doubleclick", summary: "double-click tab" },
      { kind: "action", tool: "type", summary: "type new name" },
    ]);
  });
});
