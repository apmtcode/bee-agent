import { describe, expect, it } from "vitest";
import { buildTrajectorySpan, type TrajectorySpan } from "../capture/trajectory.js";
import {
  DeterministicMovementBackend,
  buildMovementDataset,
  buildMovementDatasetFromReplay,
  defaultMovementBackends,
  evaluateMovementModel,
  resolveMovementBackend,
  rolloutMovements,
  type MovementDataset,
} from "./movement-model.js";

/**
 * Synthetic movement generator — stands in for real OS input capture (unavailable
 * in the cloud). Produces a deterministic "open app → focus field → type → submit"
 * gesture sequence for a given app so the whole capture → dataset → train → infer
 * loop can be validated without a real machine.
 */
function syntheticTrajectory(id: string, app: string, baseTs: number): TrajectorySpan {
  const steps: Array<{ gesture: string; target: string; direction?: string }> = [
    { gesture: "tap", target: `${app}-icon` },
    { gesture: "tap", target: `${app}-search-field` },
    { gesture: "type", target: `${app}-search-field` },
    { gesture: "shortcut", target: `${app}-submit` },
  ];
  return buildTrajectorySpan({
    id,
    sessionId: `session-${app}`,
    captureTier: "full",
    observations: steps.map((step, index) => ({
      kind: "observation" as const,
      source: "device",
      summary: `${app} screen step ${index}`,
      ts: baseTs + index * 10,
      metadata: { appName: app, platform: "macos" },
    })),
    actions: steps.map((step, index) => ({
      kind: "action" as const,
      tool: "device",
      summary: `${step.gesture} on ${step.target}`,
      ts: baseTs + index * 10 + 5,
      metadata: { gesture: step.gesture, target: step.target, ...(step.direction ? { direction: step.direction } : {}) },
    })),
  });
}

describe("buildMovementDataset", () => {
  it("emits one example per action with the running context window", () => {
    const dataset = buildMovementDataset([syntheticTrajectory("mail", "mail", 1000)], { windowSize: 2 });
    expect(dataset.examples).toHaveLength(4);
    expect(dataset.windowSize).toBe(2);

    // First action has no prior actions.
    expect(dataset.examples[0].context.recentActions).toEqual([]);
    expect(dataset.examples[0].context.app).toBe("mail");
    expect(dataset.examples[0].action.gesture).toBe("tap");
    expect(dataset.examples[0].action.target).toBe("mail-icon");

    // Third action sees the two most recent prior actions (window capped at 2).
    expect(dataset.examples[2].context.recentActions).toHaveLength(2);
  });

  it("recovers gesture metadata into structured action labels", () => {
    const dataset = buildMovementDataset([syntheticTrajectory("notes", "notes", 0)]);
    const typed = dataset.examples.find((example) => example.action.gesture === "type");
    expect(typed?.action.target).toBe("notes-search-field");
  });
});

describe("DeterministicMovementBackend", () => {
  it("reproduces every recorded movement exactly (repeat objective)", async () => {
    const dataset = buildMovementDataset([syntheticTrajectory("mail", "mail", 0)], { windowSize: 3 });
    const backend = new DeterministicMovementBackend();
    const model = await backend.train(dataset);

    // Replaying each recorded context recovers the recorded action exactly.
    const recovered = dataset.examples.map((example) => backend.predict(model, example.context));
    expect(recovered.map((prediction) => prediction.action.target)).toEqual([
      "mail-icon",
      "mail-search-field",
      "mail-search-field",
      "mail-submit",
    ]);
    expect(recovered.every((prediction) => prediction.generalized === false)).toBe(true);
    expect(recovered.every((prediction) => prediction.confidence === 1)).toBe(true);
  });

  it("rolls a policy forward from a seed under a stable observation", async () => {
    // A repeated-scroll flow: same observation each step, so the stateless
    // rollout can self-feed and reproduce the movement.
    const trajectory = buildTrajectorySpan({
      id: "scroll",
      sessionId: "session-feed",
      captureTier: "full",
      observations: [{ kind: "observation", source: "device", summary: "feed", ts: 0, metadata: { appName: "feed" } }],
      actions: [0, 1, 2].map((index) => ({
        kind: "action" as const,
        tool: "device",
        summary: "scroll down",
        ts: 10 + index,
        metadata: { gesture: "scroll", direction: "down" },
      })),
    });
    const dataset = buildMovementDataset([trajectory], { windowSize: 0 });
    const backend = new DeterministicMovementBackend();
    const model = await backend.train(dataset);
    const rollout = rolloutMovements(backend, model, { app: "feed", observation: "feed", recentActions: [] }, 3);
    expect(rollout.map((prediction) => prediction.action.direction)).toEqual(["down", "down", "down"]);
  });

  it("is deterministic across repeated training runs", async () => {
    const dataset = buildMovementDataset(
      [syntheticTrajectory("a", "app", 0), syntheticTrajectory("b", "app", 500)],
      { windowSize: 2 },
    );
    const backend = new DeterministicMovementBackend();
    const first = await backend.train(dataset);
    const second = await backend.train(dataset);
    expect(JSON.stringify(first)).toEqual(JSON.stringify(second));
  });

  it("generalizes to a new but related context via feature similarity", async () => {
    // Train on two apps that share the same gesture vocabulary.
    const dataset = buildMovementDataset(
      [syntheticTrajectory("mail", "mail", 0), syntheticTrajectory("chat", "chat", 1000)],
      { windowSize: 3 },
    );
    const backend = new DeterministicMovementBackend();
    const model = await backend.train(dataset);

    // A never-seen app whose observation/actions resemble the "search-field" step.
    const prediction = backend.predict(model, {
      app: "notes",
      observation: "notes screen step 2",
      recentActions: ["device||||tap on notes-search-field"],
    });
    expect(prediction.generalized).toBe(true);
    expect(prediction.confidence).toBeGreaterThan(0);
    // It should propose a device gesture, not the no-op fallback.
    expect(prediction.action.tool).toBe("device");
  });

  it("falls back to a no-op when nothing is similar", async () => {
    const dataset = buildMovementDataset([syntheticTrajectory("mail", "mail", 0)]);
    const backend = new DeterministicMovementBackend();
    const model = await backend.train(dataset);
    const prediction = backend.predict(model, { app: "zzz", observation: "totally unrelated 999", recentActions: [] });
    expect(prediction.action.tool).toBe("noop");
    expect(prediction.confidence).toBe(0);
  });

  it("honours minContextSupport to drop rare contexts", async () => {
    const dataset = buildMovementDataset([syntheticTrajectory("mail", "mail", 0)]);
    const backend = new DeterministicMovementBackend();
    const model = await backend.train(dataset, { minContextSupport: 2 });
    // Every context here is seen exactly once, so all entries are dropped.
    expect(model.entries).toHaveLength(0);
  });
});

describe("evaluateMovementModel", () => {
  it("scores replay fidelity on held-out related trajectories", async () => {
    const train = buildMovementDataset(
      [syntheticTrajectory("mail-1", "mail", 0), syntheticTrajectory("mail-2", "mail", 1000)],
      { windowSize: 3 },
    );
    const heldOut = buildMovementDataset([syntheticTrajectory("mail-3", "mail", 2000)], { windowSize: 3 });
    const backend = new DeterministicMovementBackend();
    const model = await backend.train(train);

    const result = evaluateMovementModel(backend, model, heldOut);
    expect(result.total).toBe(4);
    // Held-out is the same app/flow → the model should reproduce every action.
    expect(result.accuracy).toBe(1);
    expect(result.toolAccuracy).toBe(1);
  });

  it("reports generalized-prediction quality separately", async () => {
    const train = buildMovementDataset([syntheticTrajectory("mail", "mail", 0)], { windowSize: 3 });
    const heldOut = buildMovementDataset([syntheticTrajectory("chat", "chat", 0)], { windowSize: 3 });
    const backend = new DeterministicMovementBackend();
    const model = await backend.train(train);
    const result = evaluateMovementModel(backend, model, heldOut);
    // Different app → all predictions are generalized (nearest-neighbour).
    expect(result.generalizedPredictions).toBe(result.total);
  });
});

describe("buildMovementDatasetFromReplay", () => {
  it("builds examples from a flat replay timeline", () => {
    const dataset: MovementDataset = buildMovementDatasetFromReplay([
      { kind: "observation", ts: 1, trajectoryId: "t1", source: "browser", summary: "docs page" },
      { kind: "action", ts: 2, trajectoryId: "t1", tool: "browser", summary: "click new doc" },
      { kind: "action", ts: 3, trajectoryId: "t1", tool: "browser", summary: "type title" },
    ]);
    expect(dataset.examples).toHaveLength(2);
    expect(dataset.examples[0].context.app).toBe("browser");
    expect(dataset.examples[1].context.recentActions).toHaveLength(1);
  });
});

describe("movement backend registry", () => {
  it("resolves the deterministic backend by id", () => {
    const backend = resolveMovementBackend("deterministic-frequency");
    expect(backend.id).toBe("deterministic-frequency");
  });

  it("exposes the deterministic backend as a default", () => {
    expect(Object.keys(defaultMovementBackends())).toContain("deterministic-frequency");
  });

  it("throws on an unknown backend id", () => {
    expect(() => resolveMovementBackend("mystery-net")).toThrow(/Unknown movement model backend/);
  });
});
