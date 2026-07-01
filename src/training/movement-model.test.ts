import { describe, expect, it } from "vitest";
import {
  NgramMovementBackend,
  NgramMovementModel,
  evaluateMovementModel,
  movementDatasetFromReplays,
  movementDatasetFromTrajectories,
  movementSequenceFromTrajectory,
  type MovementDataset,
  type MovementSequence,
  type MovementStep,
} from "./movement-model.js";
import type { TrajectorySpan } from "../capture/trajectory.js";
import type { ExportedReplayManifest } from "./export-manifest.js";

function action(channel: string, summary: string, ts: number): MovementStep {
  return { ts, type: "action", channel, summary };
}

function observation(channel: string, summary: string, ts: number): MovementStep {
  return { ts, type: "observation", channel, summary };
}

/**
 * A small synthetic "movement grammar": open an app, click a target, type text,
 * then submit. Two apps share the tail, so the model can generalise the tail to
 * a third, unseen app.
 */
function appFlow(app: string, base: number): MovementSequence {
  return {
    id: `flow-${app}`,
    // The app-launch channel is app-specific (`launcher.<app>`), so an unseen app
    // has an unseen full-order token context — but the shared tail lets the model
    // recover the next movement via suffix back-off.
    steps: [
      action(`launcher.${app}`, `open ${app}`, base + 0),
      observation("ui.tree", `${app} ready`, base + 1),
      action("mouse.click", "click primary button", base + 2),
      action("keyboard.type", "enter text", base + 3),
      action("keyboard.press", "submit", base + 4),
    ],
  };
}

describe("NgramMovementModel training + recall", () => {
  it("recalls a recorded sequence exactly (repeats the movement)", () => {
    const dataset: MovementDataset = { version: 1, sequences: [appFlow("mail", 0)] };
    const model = new NgramMovementBackend().train(dataset);

    const flow = appFlow("mail", 0).steps;
    const prediction = model.predictNext(flow.slice(0, 3));
    expect(prediction).toBeDefined();
    expect(prediction!.step).toEqual({ type: "action", channel: "keyboard.type", summary: "enter text" });
    expect(prediction!.matchedOrder).toBeGreaterThan(0);
    expect(prediction!.confidence).toBe(1);
    expect(prediction!.generalized).toBe(false);
  });

  it("returns undefined when the model is untrained", () => {
    const model = new NgramMovementBackend().train({ version: 1, sequences: [] });
    expect(model.predictNext([action("mouse.click", "x", 0)])).toBeUndefined();
  });

  it("falls back to the unconditional base rate with no matching context", () => {
    const dataset: MovementDataset = { version: 1, sequences: [appFlow("mail", 0), appFlow("chat", 100)] };
    const model = new NgramMovementBackend().train(dataset);
    // A context whose channel token was never seen -> only order-0 can match.
    const prediction = model.predictNext([action("never.seen", "???", 0)]);
    expect(prediction).toBeDefined();
    expect(prediction!.matchedOrder).toBe(0);
    expect(prediction!.generalized).toBe(true);
  });
});

describe("NgramMovementModel generalization (back-off)", () => {
  it("generalises the shared tail to an unseen app via suffix back-off", () => {
    // Train only on mail + chat; hold out a *new* app that shares the flow tail.
    const dataset: MovementDataset = {
      version: 1,
      sequences: [appFlow("mail", 0), appFlow("chat", 100)],
    };
    const model = new NgramMovementBackend().train(dataset, { maxOrder: 3 });

    const unseen = appFlow("notes", 500).steps;
    // Context ends with "click primary button" -> in training that is always
    // followed by "enter text", even though the FULL prefix (opening "notes") is new.
    const prediction = model.predictNext(unseen.slice(0, 3));
    expect(prediction).toBeDefined();
    expect(prediction!.step).toEqual({ type: "action", channel: "keyboard.type", summary: "enter text" });
    expect(prediction!.generalized).toBe(true);
    expect(prediction!.matchedOrder).toBeGreaterThanOrEqual(1);
  });

  it("scores held-out generalization with the eval harness", () => {
    const train: MovementDataset = { version: 1, sequences: [appFlow("mail", 0), appFlow("chat", 100)] };
    const model = new NgramMovementBackend().train(train, { maxOrder: 3 });

    // Held-out sequences: a brand-new app plus a repeat of a trained app.
    const heldOut = [appFlow("notes", 500), appFlow("mail", 900)];
    const result = evaluateMovementModel(model, heldOut, { minContext: 1 });

    expect(result.totalPredictions).toBeGreaterThan(0);
    // The shared grammar means the model should recover the right movement kind
    // for essentially every held-out step.
    expect(result.accuracy).toBeGreaterThanOrEqual(0.75);
    expect(result.generalizedPredictions).toBeGreaterThan(0);
    expect(result.generalizedCorrect).toBeGreaterThan(0);
    expect(result.perSequence).toHaveLength(2);
  });
});

describe("NgramMovementModel determinism + serialization", () => {
  it("produces identical predictions across independent trainings", () => {
    const dataset: MovementDataset = { version: 1, sequences: [appFlow("mail", 0), appFlow("chat", 100)] };
    const a = new NgramMovementBackend().train(dataset);
    const b = new NgramMovementBackend().train(dataset);
    const context = appFlow("mail", 0).steps.slice(0, 2);
    expect(a.predictNext(context)).toEqual(b.predictNext(context));
    expect(JSON.stringify(a.serialize())).toEqual(JSON.stringify(b.serialize()));
  });

  it("round-trips through a serialized snapshot artifact", () => {
    const dataset: MovementDataset = { version: 1, sequences: [appFlow("mail", 0), appFlow("chat", 100)] };
    const model = new NgramMovementBackend().train(dataset, { maxOrder: 2 });
    const snapshot = model.serialize();
    const restored = NgramMovementModel.restore(JSON.parse(JSON.stringify(snapshot)));

    const context = appFlow("chat", 100).steps.slice(0, 3);
    expect(restored.predictNext(context)).toEqual(model.predictNext(context));
    expect(restored.maxOrder).toBe(2);
    expect(restored.backendId).toBe("ngram-backoff");
  });
});

describe("dataset distillation", () => {
  it("builds a time-ordered sequence from a trajectory span", () => {
    const trajectory: TrajectorySpan = {
      id: "traj-1",
      sessionId: "sess-1",
      createdAt: "2026-07-01T00:00:00.000Z",
      captureTier: "app",
      observations: [{ kind: "observation", source: "ui.tree", summary: "ready", ts: 2 }],
      actions: [
        { kind: "action", tool: "mouse.click", summary: "click", ts: 1 },
        { kind: "action", tool: "keyboard.type", summary: "type", ts: 3 },
      ],
    };
    const sequence = movementSequenceFromTrajectory(trajectory);
    expect(sequence.id).toBe("traj-1");
    expect(sequence.steps.map((step) => step.channel)).toEqual(["mouse.click", "ui.tree", "keyboard.type"]);
    expect(sequence.steps.map((step) => step.ts)).toEqual([1, 2, 3]);
  });

  it("drops empty trajectories from the dataset", () => {
    const empty: TrajectorySpan = {
      id: "empty",
      sessionId: "s",
      createdAt: "2026-07-01T00:00:00.000Z",
      captureTier: "operator",
      observations: [],
      actions: [],
    };
    const dataset = movementDatasetFromTrajectories([empty]);
    expect(dataset.sequences).toHaveLength(0);
  });

  it("builds a dataset from replay manifests, ignoring transcript events", () => {
    const replay: ExportedReplayManifest = {
      sessionId: "sess-1",
      trajectoryIds: ["traj-9"],
      eventCount: 3,
      events: [
        { kind: "transcript", ts: 0, messageId: "m1", role: "user", content: "hi" },
        { kind: "observation", ts: 1, trajectoryId: "traj-9", source: "ui.tree", summary: "ready" },
        { kind: "action", ts: 2, trajectoryId: "traj-9", tool: "mouse.click", summary: "click" },
      ],
    };
    const dataset = movementDatasetFromReplays([replay]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0].id).toBe("traj-9");
    expect(dataset.sequences[0].steps.map((step) => step.type)).toEqual(["observation", "action"]);
  });
});
