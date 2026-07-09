import { describe, expect, it } from "vitest";
import type { TrajectorySpan } from "../capture/trajectory.js";
import { buildMovementDataset, type MovementContext } from "./movement-dataset.js";
import {
  DeterministicMovementPolicyBackend,
  createDefaultMovementBackendRegistry,
  rolloutMovementPolicy,
} from "./movement-backend.js";

function tapSpan(id: string, target: string): TrajectorySpan {
  return {
    id,
    sessionId: "s",
    createdAt: "2026-01-01T00:00:00.000Z",
    captureTier: "app",
    observations: [
      { kind: "observation", source: "device", summary: `Editor ${target} screen`, ts: 10, metadata: { appName: "Editor" } },
    ],
    actions: [{ kind: "action", tool: "device", summary: `tapped ${target}`, ts: 20, metadata: { gesture: "tap", target } }],
  };
}

describe("DeterministicMovementPolicyBackend", () => {
  it("reproduces recorded movements exactly", async () => {
    const dataset = buildMovementDataset([tapSpan("t1", "Save"), tapSpan("t2", "Open")]);
    const policy = await new DeterministicMovementPolicyBackend().train(dataset);

    const saveContext = dataset.examples[0].context;
    const prediction = policy.predict(saveContext);

    expect(prediction.source).toBe("exact");
    expect(prediction.action).toEqual(dataset.examples[0].action);
    expect(prediction.confidence).toBeGreaterThanOrEqual(0.5);
  });

  it("generalizes to a new-but-related context via backoff", async () => {
    // Trained only on Save/Open taps inside Editor.
    const dataset = buildMovementDataset([tapSpan("t1", "Save"), tapSpan("t2", "Open")]);
    const policy = await new DeterministicMovementPolicyBackend().train(dataset);

    // Novel screen/target never seen exactly, but same app + observation source.
    const novelContext: MovementContext = {
      appName: "Editor",
      observationSource: "device",
      observationSummary: "Editor Close screen",
      stepIndex: 0,
    };
    const prediction = policy.predict(novelContext);

    expect(prediction.source).toBe("generalized");
    expect(prediction.action?.tool).toBe("device");
    expect(prediction.action?.gesture).toBe("tap"); // learned the tap family, not a specific target
    expect(prediction.matchedFeatures).toContain("appName");
  });

  it("falls back to the global most-frequent action when nothing matches", async () => {
    const dataset = buildMovementDataset([tapSpan("t1", "Save"), tapSpan("t2", "Open"), tapSpan("t3", "Save")]);
    const policy = await new DeterministicMovementPolicyBackend().train(dataset);

    const unrelated: MovementContext = { stepIndex: 0 }; // no reusable features at all
    const prediction = policy.predict(unrelated);

    expect(prediction.source).toBe("fallback");
    // "tapped Save" was recorded twice vs "tapped Open" once → most frequent.
    expect(prediction.action?.summary).toBe("tapped Save");
  });

  it("returns an empty prediction when trained on no data", async () => {
    const policy = await new DeterministicMovementPolicyBackend().train({ version: 1, exampleCount: 0, examples: [] });
    const prediction = policy.predict({ stepIndex: 0 });

    expect(prediction.source).toBe("empty");
    expect(prediction.action).toBeUndefined();
    expect(prediction.confidence).toBe(0);
  });

  it("is deterministic regardless of example order", async () => {
    const forward = buildMovementDataset([tapSpan("a", "Save"), tapSpan("b", "Open"), tapSpan("c", "Save")]);
    const reversed = { ...forward, examples: [...forward.examples].reverse() };

    const backend = new DeterministicMovementPolicyBackend();
    const p1 = await backend.train(forward);
    const p2 = await backend.train(reversed);

    expect(p1.serialize()).toEqual(p2.serialize());
  });

  it("round-trips through serialize/load", async () => {
    const dataset = buildMovementDataset([tapSpan("t1", "Save"), tapSpan("t2", "Open")]);
    const backend = new DeterministicMovementPolicyBackend();
    const trained = await backend.train(dataset);

    const restored = backend.load(JSON.parse(JSON.stringify(trained.serialize())));

    const context = dataset.examples[0].context;
    expect(restored.predict(context)).toEqual(trained.predict(context));
    expect(restored.stepCount).toBe(trained.stepCount);
  });

  it("rolls out a policy over held-out related contexts", async () => {
    const dataset = buildMovementDataset([tapSpan("t1", "Save"), tapSpan("t2", "Open")]);
    const policy = await new DeterministicMovementPolicyBackend().train(dataset);

    const predictions = rolloutMovementPolicy(policy, [
      dataset.examples[0].context, // exact
      { appName: "Editor", observationSource: "device", observationSummary: "Editor Delete screen", stepIndex: 0 }, // generalized
    ]);

    expect(predictions.map((p) => p.source)).toEqual(["exact", "generalized"]);
  });
});

describe("createDefaultMovementBackendRegistry", () => {
  it("registers the deterministic backend and looks it up by id", () => {
    const registry = createDefaultMovementBackendRegistry();
    expect(registry.list().map((b) => b.id)).toContain("deterministic-frequency");
    expect(registry.get("deterministic-frequency")).toBeDefined();
    expect(registry.get("nonexistent")).toBeUndefined();
  });
});
