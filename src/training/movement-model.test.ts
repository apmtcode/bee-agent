import { describe, expect, it } from "vitest";
import {
  DEFAULT_SYNTHETIC_APPS,
  synthesizeMovementTrajectories,
} from "../capture/synthetic.js";
import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  InProcessMovementModelBackend,
  deriveMovementDataset,
  evaluateMovementModel,
  type MovementSample,
} from "./movement-model.js";

function trainOn(trajectories: TrajectorySpan[], threshold?: number) {
  const dataset = deriveMovementDataset(trajectories);
  const backend = new InProcessMovementModelBackend();
  return { dataset, model: backend.train(dataset, threshold === undefined ? {} : { generalizationThreshold: threshold }) };
}

describe("deriveMovementDataset", () => {
  it("pairs each action with the observation that preceded it", async () => {
    const spans = synthesizeMovementTrajectories({ seed: 7, count: 2, stepsPerTrajectory: 2 });
    const dataset = deriveMovementDataset(spans);
    expect(dataset.version).toBe(1);
    // 2 trajectories x 2 steps = 4 samples.
    expect(dataset.samples).toHaveLength(4);
    for (const sample of dataset.samples) {
      expect(sample.context.appId).not.toBe("unknown");
      expect(sample.context.screenTitle).toBeDefined();
      expect(sample.action.tool).toBe("device");
      expect(sample.action.gesture).toBeDefined();
    }
  });

  it("threads recent tools into later contexts", async () => {
    const spans = synthesizeMovementTrajectories({ seed: 2, count: 1, stepsPerTrajectory: 3 });
    const dataset = deriveMovementDataset(spans);
    const withHistory = dataset.samples.filter((sample) => (sample.context.recentTools ?? []).length > 0);
    expect(withHistory.length).toBeGreaterThan(0);
  });
});

describe("InProcessMovementModelBackend", () => {
  it("repeats a recorded movement exactly for a seen context", async () => {
    const spans = synthesizeMovementTrajectories({ seed: 11, count: 6, stepsPerTrajectory: 3 });
    const { dataset, model: modelPromise } = trainOn(spans);
    const model = await modelPromise;

    const seen = dataset.samples[0]!;
    const prediction = model.predict(seen.context);
    expect(prediction).toBeDefined();
    expect(prediction!.source).toBe("exact");
    expect(prediction!.similarity).toBe(1);
    expect(prediction!.action.tool).toBe(seen.action.tool);
    expect(prediction!.action.gesture).toBe(seen.action.gesture);
  });

  it("is deterministic — identical dataset yields identical predictions", async () => {
    const spans = synthesizeMovementTrajectories({ seed: 99, count: 8 });
    const a = await (new InProcessMovementModelBackend()).train(deriveMovementDataset(spans));
    const b = await (new InProcessMovementModelBackend()).train(deriveMovementDataset(spans));
    const context = { appId: "browser", screenTitle: "some-unseen-page" };
    expect(a.predict(context)).toEqual(b.predict(context));
  });

  it("generalizes a movement to a new but related context (same app, unseen screen)", async () => {
    const app = DEFAULT_SYNTHETIC_APPS.find((a) => a.appId === "editor")!;
    // Train only on the editor app so the same-app skill dominates.
    const spans = synthesizeMovementTrajectories({
      seed: 5,
      count: 10,
      stepsPerTrajectory: 4,
      apps: [app],
      holdOutScreenIndexes: [4], // withhold the last screen entirely
    });
    const model = await (new InProcessMovementModelBackend()).train(deriveMovementDataset(spans));

    // A screen the model never saw, but the same app — should transfer the skill.
    const heldOutScreen = app.screens[4]!;
    const prediction = model.predict({ appId: app.appId, screenTitle: heldOutScreen });
    expect(prediction).toBeDefined();
    expect(prediction!.source).toBe("generalized");
    expect(prediction!.action.tool).toBe(app.skill.tool);
    expect(prediction!.action.gesture).toBe(app.skill.gesture);
    expect(prediction!.similarity).toBeGreaterThan(0);
    expect(prediction!.similarity).toBeLessThan(1);
  });

  it("falls back to the global top action when nothing is similar enough", async () => {
    const spans = synthesizeMovementTrajectories({ seed: 3, count: 6, apps: [DEFAULT_SYNTHETIC_APPS[1]!] });
    // High threshold makes any cross-context transfer impossible.
    const model = await (new InProcessMovementModelBackend()).train(deriveMovementDataset(spans), {
      generalizationThreshold: 0.99,
    });
    const prediction = model.predict({ appId: "totally-unknown-app", screenTitle: "nowhere" });
    expect(prediction).toBeDefined();
    expect(prediction!.source).toBe("fallback");
    expect(prediction!.similarity).toBe(0);
  });

  it("returns undefined when trained on an empty dataset", async () => {
    const model = await (new InProcessMovementModelBackend()).train({ version: 1, samples: [] });
    expect(model.predict({ appId: "editor" })).toBeUndefined();
    expect(model.contextCount).toBe(0);
    expect(model.sampleCount).toBe(0);
  });

  it("describe() reports learned contexts", async () => {
    const spans = synthesizeMovementTrajectories({ seed: 8, count: 5 });
    const model = await (new InProcessMovementModelBackend()).train(deriveMovementDataset(spans));
    const descriptor = model.describe();
    expect(descriptor.backendId).toBe("in-process-frequency-v1");
    expect(descriptor.contexts.length).toBe(model.contextCount);
    expect(descriptor.contexts.every((c) => c.features.some((f) => f.startsWith("app:")))).toBe(true);
  });
});

describe("evaluateMovementModel", () => {
  it("achieves high tool/gesture fidelity on held-out related trajectories", async () => {
    const apps = DEFAULT_SYNTHETIC_APPS;
    // Train withholding screen index 4 for every app; eval only on that screen.
    const trainSpans = synthesizeMovementTrajectories({
      seed: 21,
      count: 40,
      stepsPerTrajectory: 4,
      apps,
      holdOutScreenIndexes: [4],
    });
    const model = await (new InProcessMovementModelBackend()).train(deriveMovementDataset(trainSpans));

    // Held-out eval samples: same apps/skills, exclusively the unseen screen.
    const heldOut: MovementSample[] = apps.map((app) => ({
      context: { appId: app.appId, screenTitle: app.screens[4]! },
      action: { tool: app.skill.tool, gesture: app.skill.gesture, target: app.skill.target, summary: "held-out" },
    }));

    const result = evaluateMovementModel(model, heldOut);
    expect(result.total).toBe(apps.length);
    expect(result.predicted).toBe(apps.length);
    // Every held-out context is an unseen screen -> pure generalization.
    expect(result.generalized).toBe(apps.length);
    expect(result.toolFidelity).toBe(1);
    expect(result.gestureFidelity).toBe(1);
  });

  it("reports zeroed fidelity for an empty eval set", async () => {
    const model = await (new InProcessMovementModelBackend()).train({ version: 1, samples: [] });
    const result = evaluateMovementModel(model, []);
    expect(result).toMatchObject({ total: 0, predicted: 0, toolFidelity: 0, gestureFidelity: 0 });
  });
});
