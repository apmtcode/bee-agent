import { describe, expect, it } from "vitest";
import { MarkovMovementBackend } from "./movement-policy.js";
import {
  DEFAULT_MOVEMENT_TEMPLATES,
  evaluateMovementModel,
  generateSyntheticMovementTrajectories,
  splitTrajectories,
} from "./movement-eval.js";

describe("generateSyntheticMovementTrajectories", () => {
  it("is deterministic for a fixed seed", () => {
    const a = generateSyntheticMovementTrajectories({ seed: 42, perTemplate: 4 });
    const b = generateSyntheticMovementTrajectories({ seed: 42, perTemplate: 4 });
    expect(b).toEqual(a);
  });

  it("produces related-but-varied flows sharing gesture shape", () => {
    const trajectories = generateSyntheticMovementTrajectories({ seed: 7, perTemplate: 6 });
    const compose = trajectories.filter((t) => t.id.includes("compose-email"));
    const shapes = new Set(compose.map((t) => t.events.map((e) => e.gesture).join(">")));
    // Same gesture shape across all instances...
    expect(shapes.size).toBe(1);
    // ...but the concrete targets vary across instances.
    const firstTargets = new Set(compose.map((t) => t.events[0]?.target));
    expect(firstTargets.size).toBeGreaterThan(1);
  });
});

describe("splitTrajectories", () => {
  it("partitions without overlap and covers both sides", () => {
    const all = generateSyntheticMovementTrajectories({ seed: 1, perTemplate: 8 });
    const { train, holdout } = splitTrajectories(all, 0.25);
    expect(train.length + holdout.length).toBe(all.length);
    expect(holdout.length).toBeGreaterThan(0);
    const trainIds = new Set(train.map((t) => t.id));
    expect(holdout.every((t) => !trainIds.has(t.id))).toBe(true);
  });
});

describe("evaluateMovementModel", () => {
  it("achieves high replay fidelity on the training set", () => {
    const all = generateSyntheticMovementTrajectories({ seed: 3, perTemplate: 10 });
    const { train } = splitTrajectories(all);
    const model = new MarkovMovementBackend().train(train, { order: 4 });
    const metrics = evaluateMovementModel(model, train);
    expect(metrics.gestureAccuracy).toBeGreaterThan(0.99);
    expect(metrics.tokenAccuracy).toBeGreaterThan(0.5);
  });

  it("generalizes gesture shape to held-out related flows", () => {
    const all = generateSyntheticMovementTrajectories({ seed: 5, perTemplate: 12 });
    const { train, holdout } = splitTrajectories(all, 0.25);
    const model = new MarkovMovementBackend().train(train, { order: 4 });
    const metrics = evaluateMovementModel(model, holdout);
    // Held-out flows reuse the learned gesture structure even with new targets.
    expect(metrics.gestureAccuracy).toBeGreaterThan(0.9);
    expect(metrics.stepCount).toBeGreaterThan(0);
  });

  it("covers every default template", () => {
    const all = generateSyntheticMovementTrajectories({ perTemplate: 3 });
    const apps = new Set(all.map((t) => t.app));
    for (const template of DEFAULT_MOVEMENT_TEMPLATES) {
      expect(apps.has(template.app)).toBe(true);
    }
  });
});
