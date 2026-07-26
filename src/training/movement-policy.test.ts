import { describe, expect, it } from "vitest";
import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  MovementPolicyBackendRegistry,
  NEAREST_NEIGHBOR_BACKEND_ID,
  NearestNeighborMovementBackend,
  buildMovementDataset,
  createDefaultMovementPolicyRegistry,
  evaluateMovementPolicy,
  type MovementDataset,
  type MovementPolicyBackend,
  type MovementSample,
} from "./movement-policy.js";

function editorTap(target: string): MovementSample {
  return {
    context: { appId: "editor", platform: "macos", screenTitle: "compose", targetHint: target },
    gesture: { kind: "tap", target },
  };
}

const trainingDataset: MovementDataset = {
  samples: [editorTap("Save"), editorTap("Publish"), editorTap("Bold")],
};

describe("NearestNeighborMovementBackend", () => {
  const backend = new NearestNeighborMovementBackend();

  it("repeats a recorded movement exactly (recall)", () => {
    const model = backend.train(trainingDataset);
    const prediction = model.predict({
      appId: "editor",
      platform: "macos",
      screenTitle: "compose",
      targetHint: "Save",
    });

    expect(prediction.source).toBe("recall");
    expect(prediction.confidence).toBe(1);
    expect(prediction.gesture).toEqual({ kind: "tap", target: "Save" });
  });

  it("generalizes a learned gesture to a new but related target", () => {
    const model = backend.train(trainingDataset);
    const prediction = model.predict({
      appId: "editor",
      platform: "macos",
      screenTitle: "compose",
      targetHint: "Archive", // never seen during training
    });

    expect(prediction.source).toBe("generalized");
    expect(prediction.gesture.kind).toBe("tap");
    // slot-filled from the query hint, not copied from a training target
    expect(prediction.gesture.target).toBe("Archive");
    expect(prediction.confidence).toBeGreaterThan(0);
    expect(prediction.confidence).toBeLessThan(1);
  });

  it("generalizes directional gestures via the direction slot", () => {
    const model = backend.train({
      samples: [
        {
          context: { appId: "gallery", screenTitle: "photos", directionHint: "left" },
          gesture: { kind: "swipe", direction: "left" },
        },
        {
          context: { appId: "gallery", screenTitle: "photos", directionHint: "right" },
          gesture: { kind: "swipe", direction: "right" },
        },
      ],
    });

    const prediction = model.predict({ appId: "gallery", screenTitle: "photos", directionHint: "up" });
    expect(prediction.source).toBe("generalized");
    expect(prediction.gesture.kind).toBe("swipe");
    expect(prediction.gesture.direction).toBe("up");
  });

  it("returns an empty-source prediction for an untrained model", () => {
    const model = backend.train({ samples: [] });
    expect(model.sampleCount).toBe(0);
    const prediction = model.predict({ appId: "editor", targetHint: "Save" });
    expect(prediction.source).toBe("empty");
    expect(prediction.confidence).toBe(0);
    expect(prediction.gesture).toEqual({ kind: "tap", target: "Save" });
  });

  it("round-trips through serialization with identical predictions", () => {
    const model = backend.train(trainingDataset);
    const serialized = model.toJSON();
    expect(serialized.backendId).toBe(NEAREST_NEIGHBOR_BACKEND_ID);

    const reloaded = backend.load(serialized);
    expect(reloaded.sampleCount).toBe(model.sampleCount);

    const context = { appId: "editor", platform: "macos" as const, screenTitle: "compose", targetHint: "Archive" };
    expect(reloaded.predict(context)).toEqual(model.predict(context));
  });

  it("rejects loading a model built by a different backend", () => {
    expect(() => backend.load({ version: 1, backendId: "some-other-backend", samples: [] })).toThrow(
      /Cannot load model/,
    );
  });
});

describe("buildMovementDataset", () => {
  it("reconstructs samples from device-adapter-shaped trajectory metadata", () => {
    const trajectory: TrajectorySpan = {
      id: "traj-1",
      sessionId: "sess-1",
      createdAt: "2026-01-01T00:00:00.000Z",
      captureTier: "app",
      observations: [
        {
          kind: "observation",
          source: "device",
          summary: "Editor on compose",
          ts: 1,
          metadata: { appName: "editor", platform: "macos", screenTitle: "compose" },
        },
      ],
      actions: [
        {
          kind: "action",
          tool: "device",
          summary: "tapped Save",
          ts: 2,
          metadata: { gesture: "tap", target: "Save" },
        },
        {
          kind: "action",
          tool: "device",
          summary: "swiped left",
          ts: 3,
          metadata: { gesture: "swipe", direction: "left" },
        },
      ],
    };

    const dataset = buildMovementDataset([trajectory]);
    expect(dataset.samples).toHaveLength(2);

    expect(dataset.samples[0]).toEqual({
      context: { appId: "editor", platform: "macos", screenTitle: "compose", targetHint: "Save" },
      gesture: { kind: "tap", target: "Save" },
    });

    // The second sample threads the previous gesture kind into the context.
    expect(dataset.samples[1].context).toMatchObject({
      appId: "editor",
      screenTitle: "compose",
      directionHint: "left",
      priorGestureKind: "tap",
    });
    expect(dataset.samples[1].gesture).toEqual({ kind: "swipe", direction: "left" });
  });

  it("ignores actions that carry no gesture metadata", () => {
    const trajectory: TrajectorySpan = {
      id: "traj-2",
      sessionId: "sess-2",
      createdAt: "2026-01-01T00:00:00.000Z",
      captureTier: "operator",
      observations: [],
      actions: [{ kind: "action", tool: "browser", summary: "clicked deploy", ts: 1 }],
    };
    expect(buildMovementDataset([trajectory]).samples).toHaveLength(0);
  });
});

describe("evaluateMovementPolicy", () => {
  it("scores perfect recall on the training samples", () => {
    const model = new NearestNeighborMovementBackend().train(trainingDataset);
    const evaluation = evaluateMovementPolicy(model, trainingDataset.samples);
    expect(evaluation.total).toBe(3);
    expect(evaluation.exactAccuracy).toBe(1);
    expect(evaluation.recallCount).toBe(3);
    expect(evaluation.meanConfidence).toBe(1);
  });

  it("measures generalization on held-out related samples", () => {
    const model = new NearestNeighborMovementBackend().train(trainingDataset);
    const heldOut: MovementSample[] = [editorTap("Archive"), editorTap("Delete")];
    const evaluation = evaluateMovementPolicy(model, heldOut);

    expect(evaluation.kindAccuracy).toBe(1); // gesture kind always generalizes
    expect(evaluation.targetAccuracy).toBe(1); // slot-filled targets match
    expect(evaluation.exactAccuracy).toBe(1);
    expect(evaluation.generalizedCount).toBe(2);
  });
});

describe("MovementPolicyBackendRegistry", () => {
  it("seeds a default nearest-neighbor backend", () => {
    const registry = createDefaultMovementPolicyRegistry();
    expect(registry.list()).toContain(NEAREST_NEIGHBOR_BACKEND_ID);
    expect(registry.get(NEAREST_NEIGHBOR_BACKEND_ID).id).toBe(NEAREST_NEIGHBOR_BACKEND_ID);
  });

  it("supports registering pluggable custom backends", () => {
    const registry = new MovementPolicyBackendRegistry([]);
    const stub: MovementPolicyBackend = {
      id: "stub-backend",
      train: (dataset) => new NearestNeighborMovementBackend().train(dataset),
      load: (serialized) => new NearestNeighborMovementBackend().load({ ...serialized, backendId: NEAREST_NEIGHBOR_BACKEND_ID }),
    };
    registry.register(stub);
    expect(registry.has("stub-backend")).toBe(true);
    expect(registry.get("stub-backend")).toBe(stub);
  });

  it("throws a helpful error for an unknown backend", () => {
    const registry = createDefaultMovementPolicyRegistry();
    expect(() => registry.get("missing")).toThrow(/Unknown movement-policy backend "missing"/);
  });
});
