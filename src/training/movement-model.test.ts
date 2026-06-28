import { describe, expect, it } from "vitest";
import { buildTrajectorySpan, type TrajectoryAction, type TrajectorySpan } from "../capture/trajectory.js";
import type { ExportedReplayManifest } from "./export-manifest.js";
import {
  MOVEMENT_END_TOKEN,
  MarkovMovementBackend,
  buildMovementDataset,
  buildMovementDatasetFromReplays,
  evaluateMovementModel,
  tokenizeAction,
  trainMovementModel,
} from "./movement-model.js";

function gesture(
  ts: number,
  kind: string,
  fields: { target?: string; direction?: string; valueSummary?: string } = {},
): TrajectoryAction {
  return {
    kind: "action",
    tool: "device",
    summary: `${kind} ${fields.target ?? fields.direction ?? ""}`.trim(),
    ts,
    metadata: { gesture: kind, ...fields },
  };
}

/** Synthetic "open app, tap field, type, submit" movement trajectory. */
function synthTrajectory(id: string, target: string, base = 0): TrajectorySpan {
  return buildTrajectorySpan({
    id,
    sessionId: "session-synth",
    captureTier: "full",
    actions: [
      gesture(base + 1, "tap", { target: "app-icon" }),
      gesture(base + 2, "tap", { target }),
      gesture(base + 3, "type", { target, valueSummary: "text" }),
      gesture(base + 4, "shortcut", { target: "submit" }),
    ],
  });
}

describe("tokenizeAction", () => {
  it("produces a specific token and a gesture-class token", () => {
    const { specific, class: classToken } = tokenizeAction(gesture(1, "tap", { target: "Submit Button" }));
    expect(specific).toBe("device/tap/submit-button");
    expect(classToken).toBe("device/tap/*");
  });

  it("falls back to the summary when there is no gesture metadata", () => {
    const action: TrajectoryAction = { kind: "action", tool: "shell", summary: "ran build", ts: 1 };
    expect(tokenizeAction(action).specific).toBe("shell/ran-build");
  });
});

describe("buildMovementDataset", () => {
  it("emits a step per action plus an end token, ordered by timestamp", () => {
    const dataset = buildMovementDataset([synthTrajectory("t1", "search-box")], { order: 3 });
    // 4 actions + 1 end-of-sequence target.
    expect(dataset.steps).toHaveLength(5);
    expect(dataset.steps.at(-1)?.token).toBe(MOVEMENT_END_TOKEN);
    expect(dataset.classOf["device/tap/app-icon"]).toBe("device/tap/*");
    expect(dataset.vocabulary).toContain(MOVEMENT_END_TOKEN);
  });

  it("orders actions by timestamp regardless of array order", () => {
    const span = buildTrajectorySpan({
      id: "t-order",
      sessionId: "s",
      actions: [gesture(3, "shortcut", { target: "submit" }), gesture(1, "tap", { target: "field" })],
    });
    const dataset = buildMovementDataset([span], { order: 2 });
    expect(dataset.steps[0]?.token).toBe("device/tap/field");
  });
});

describe("MarkovMovementBackend replay fidelity", () => {
  it("reproduces a recorded movement sequence exactly from a seed", () => {
    const trajectory = synthTrajectory("t1", "search-box");
    const model = trainMovementModel([trajectory], { order: 3 });

    const generated = model.generate(["device/tap/app-icon"]);
    expect(generated).toEqual([
      "device/tap/search-box",
      "device/type/search-box",
      "device/shortcut/submit",
    ]);
  });

  it("scores perfect top-1 accuracy replaying its own seeded contexts", () => {
    const trajectory = synthTrajectory("t1", "search-box");
    const dataset = buildMovementDataset([trajectory], { order: 3 });
    const backend = new MarkovMovementBackend();
    const model = backend.load(backend.train(dataset));
    // Exclude the first step (empty context is unpredictable without a seed).
    const seededSteps = dataset.steps.filter((step) => step.context.length > 0);
    const evaluation = evaluateMovementModel(model, seededSteps);
    expect(evaluation.top1Accuracy).toBe(1);
  });
});

describe("MarkovMovementBackend generalization", () => {
  it("predicts the right gesture class for an unseen-but-related target", () => {
    // Train on tapping several different fields, then tap a brand new one.
    const trajectories = [
      synthTrajectory("t1", "search-box", 0),
      synthTrajectory("t2", "name-field", 10),
      synthTrajectory("t3", "email-field", 20),
    ];
    const model = trainMovementModel(trajectories, { order: 2 });

    // Novel context: tapped an icon we never paired with this exact field.
    const prediction = model.predictNext(["device/tap/app-icon", "device/tap/brand-new-field"]);
    expect(prediction).toBeDefined();
    // The exact specific token is unseen, but the class should be a "type" gesture.
    expect(model.classOf(prediction!.token)).toBe("device/type/*");
    expect(prediction!.source).toBe("generalized");
  });

  it("measures generalization on held-out related trajectories", () => {
    const train = [
      synthTrajectory("t1", "search-box", 0),
      synthTrajectory("t2", "name-field", 10),
    ];
    const heldOut = buildMovementDataset([synthTrajectory("t3", "phone-field", 20)], { order: 2 }).steps;
    const model = trainMovementModel(train, { order: 2 });
    const evaluation = evaluateMovementModel(model, heldOut);
    // Class accuracy should beat exact accuracy because targets differ but the
    // gesture *shape* of the workflow is shared.
    expect(evaluation.classAccuracy).toBeGreaterThanOrEqual(evaluation.top1Accuracy);
    expect(evaluation.classAccuracy).toBeGreaterThan(0.5);
  });
});

describe("artifact round-trip", () => {
  it("serializes and reloads deterministically", () => {
    const dataset = buildMovementDataset([synthTrajectory("t1", "search-box")], { order: 3 });
    const backend = new MarkovMovementBackend();
    const artifact = backend.train(dataset);
    const json = JSON.parse(JSON.stringify(artifact));
    const reloaded = backend.load(json);
    expect(reloaded.generate(["device/tap/app-icon"])).toEqual([
      "device/tap/search-box",
      "device/type/search-box",
      "device/shortcut/submit",
    ]);
    expect(artifact.stats.stepCount).toBe(dataset.steps.length);
    expect(artifact.stats.trajectoryCount).toBe(1);
  });
});

describe("buildMovementDatasetFromReplays", () => {
  it("derives movement steps from exported replay action events", () => {
    const replay: ExportedReplayManifest = {
      sessionId: "s",
      trajectoryIds: ["t1"],
      eventCount: 3,
      events: [
        { kind: "observation", ts: 1, trajectoryId: "t1", source: "device", summary: "app active" },
        { kind: "action", ts: 2, trajectoryId: "t1", tool: "device", summary: "tapped search-box" },
        { kind: "action", ts: 3, trajectoryId: "t1", tool: "device", summary: "typed query" },
      ],
    };
    const dataset = buildMovementDatasetFromReplays([replay], { order: 2 });
    const tokens = dataset.steps.map((step) => step.token);
    expect(tokens).toContain("device/tapped/search-box");
    expect(tokens).toContain("device/typed/query");
    expect(tokens.at(-1)).toBe(MOVEMENT_END_TOKEN);
  });
});
