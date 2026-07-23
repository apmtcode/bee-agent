import { describe, expect, it } from "vitest";
import type { ReplayManifest } from "../capture/replay.js";
import {
  MarkovMovementBackend,
  buildMovementDataset,
  evaluateMovementModel,
  restoreMovementModel,
  type MovementActionSample,
  type MovementSequence,
} from "./movement-model.js";

function actionEvent(trajectoryId: string, ts: number, tool: string, summary: string) {
  return { kind: "action" as const, ts, trajectoryId, tool, summary };
}

/** A synthetic replay manifest — stands in for real captured OS movements. */
function syntheticManifest(sessionId: string, trajectoryId: string, steps: Array<[string, string]>): ReplayManifest {
  const events = steps.map(([tool, summary], index) => actionEvent(trajectoryId, index + 1, tool, summary));
  return {
    version: 1,
    sessionId,
    trajectoryIds: [trajectoryId],
    eventCount: events.length,
    events,
  };
}

const DRAG_MOVEMENT: Array<[string, string]> = [
  ["mouse.move", "to icon (120,80)"],
  ["mouse.down", "press left"],
  ["mouse.move", "drag to folder (400,300)"],
  ["mouse.up", "release left"],
];

describe("buildMovementDataset", () => {
  it("groups action events per trajectory, ordered by timestamp, ignoring non-actions", () => {
    const manifest: ReplayManifest = {
      version: 1,
      sessionId: "s1",
      trajectoryIds: ["t1"],
      eventCount: 4,
      events: [
        { kind: "transcript", ts: 1, messageId: "m1", role: "user", content: "do the thing" },
        actionEvent("t1", 3, "mouse.up", "release"),
        { kind: "observation", ts: 2, trajectoryId: "t1", source: "screen", summary: "icon visible" },
        actionEvent("t1", 2, "mouse.down", "press"),
      ],
    };
    const dataset = buildMovementDataset([manifest]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]!.actions.map((a) => a.tool)).toEqual(["mouse.down", "mouse.up"]);
  });

  it("separates sequences by trajectory", () => {
    const dataset = buildMovementDataset([
      syntheticManifest("s1", "t1", DRAG_MOVEMENT),
      syntheticManifest("s1", "t2", [["key.press", "cmd+c"]]),
    ]);
    expect(dataset.sequences).toHaveLength(2);
  });
});

describe("MarkovMovementBackend — repetition (objective 2c)", () => {
  it("rolls out the exact recorded movement from a seed", async () => {
    const dataset = buildMovementDataset([syntheticManifest("s1", "t1", DRAG_MOVEMENT)]);
    const model = await new MarkovMovementBackend().train(dataset);

    const seed: MovementActionSample[] = [{ tool: "mouse.move", summary: "to icon (120,80)", ts: 1 }];
    const rolled = model.rollout({ seed, maxSteps: 3 });

    expect(rolled.map((a) => `${a.tool}|${a.summary}`)).toEqual([
      "mouse.down|press left",
      "mouse.move|drag to folder (400,300)",
      "mouse.up|release left",
    ]);
  });

  it("predicts the memorized next action with exact provenance", async () => {
    const dataset = buildMovementDataset([syntheticManifest("s1", "t1", DRAG_MOVEMENT)]);
    const model = await new MarkovMovementBackend().train(dataset, { order: 2 });

    const prediction = model.predictNext([
      { tool: "mouse.down", summary: "press left", ts: 2 },
      { tool: "mouse.move", summary: "drag to folder (400,300)", ts: 3 },
    ]);
    expect(prediction?.tool).toBe("mouse.up");
    expect(prediction?.source).toBe("exact");
    expect(prediction?.confidence).toBe(1);
  });
});

describe("MarkovMovementBackend — generalization (objective 2d)", () => {
  it("continues a novel context via lower-order backoff", async () => {
    // Two related movements share the pattern: after mouse.down comes a drag move.
    const dataset = buildMovementDataset([
      syntheticManifest("s1", "t1", [
        ["mouse.down", "press at A"],
        ["mouse.move", "drag from A"],
      ]),
      syntheticManifest("s1", "t2", [
        ["mouse.down", "press at B"],
        ["mouse.move", "drag from B"],
      ]),
    ]);
    const model = await new MarkovMovementBackend().train(dataset, { order: 2 });

    // A never-seen exact context (different summary) still generalizes on the tool.
    const prediction = model.predictNext([{ tool: "mouse.down", summary: "press at C (unseen)", ts: 5 }]);
    expect(prediction?.tool).toBe("mouse.move");
    // order-1 context matched (history length 1), so it is "exact" at that order;
    // the point is it did not need the full training summary to continue.
    expect(prediction).toBeDefined();
  });

  it("falls back to the global prior when no context matches", async () => {
    const dataset = buildMovementDataset([syntheticManifest("s1", "t1", DRAG_MOVEMENT)]);
    const model = await new MarkovMovementBackend().train(dataset, { order: 2 });

    const prediction = model.predictNext([{ tool: "totally.unknown.tool", summary: "n/a", ts: 9 }]);
    expect(prediction).toBeDefined();
    expect(prediction?.source).toBe("prior");
  });
});

describe("evaluateMovementModel — generalization eval harness", () => {
  it("scores perfect tool accuracy on held-out sequences drawn from the same distribution", async () => {
    const train = buildMovementDataset([
      syntheticManifest("s1", "t1", DRAG_MOVEMENT),
      syntheticManifest("s1", "t2", DRAG_MOVEMENT),
    ]);
    const model = await new MarkovMovementBackend().train(train, { order: 2 });

    const heldOut: MovementSequence[] = [{ trajectoryId: "ho", sessionId: "s2", actions: DRAG_MOVEMENT.map(([tool, summary], i) => ({ tool, summary, ts: i + 1 })) }];
    const result = evaluateMovementModel(model, heldOut);

    expect(result.predictedStepCount).toBe(3);
    expect(result.toolAccuracy).toBe(1);
    expect(result.exactAccuracy).toBe(1);
    expect(result.abstainCount).toBe(0);
  });

  it("reports zeroed metrics for an empty eval set without dividing by zero", async () => {
    const model = await new MarkovMovementBackend().train({ version: 1, sequences: [] });
    const result = evaluateMovementModel(model, []);
    expect(result.toolAccuracy).toBe(0);
    expect(result.exactAccuracy).toBe(0);
    expect(result.predictedStepCount).toBe(0);
  });
});

describe("serialization round-trip", () => {
  it("restores an identical model from its JSON form", async () => {
    const dataset = buildMovementDataset([syntheticManifest("s1", "t1", DRAG_MOVEMENT)]);
    const model = await new MarkovMovementBackend().train(dataset, { order: 2 });
    const restored = restoreMovementModel(model.toJSON());

    expect(restored.backendId).toBe(model.backendId);
    expect(restored.order).toBe(model.order);
    expect(restored.actionCount).toBe(model.actionCount);
    expect(restored.transitionCount).toBe(model.transitionCount);

    const history: MovementActionSample[] = [{ tool: "mouse.down", summary: "press left", ts: 2 }];
    expect(restored.predictNext(history)).toEqual(model.predictNext(history));
  });
});
