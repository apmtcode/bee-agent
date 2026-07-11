import { describe, expect, it } from "vitest";
import type { ReplayTimelineEvent } from "../capture/replay.js";
import {
  LocalModelBackendRegistry,
  MockLocalModelBackend,
  createDefaultLocalModelBackendRegistry,
  type MovementDataset,
} from "./model-backend.js";
import { generateSyntheticMovementDataset } from "./synthetic-movements.js";

function dataset(events: ReplayTimelineEvent[], jobId = "job"): MovementDataset {
  return { jobId, events };
}

describe("MockLocalModelBackend", () => {
  it("learns an observation->action policy and predicts an exact match", () => {
    const backend = new MockLocalModelBackend();
    const model = backend.train(generateSyntheticMovementDataset({ repeats: 4 }));

    expect(model.backend).toBe("mock");
    expect(model.metadata.observationCount).toBe(16);
    expect(model.metadata.actionCount).toBe(16);
    expect(model.metadata.transitionCount).toBe(4);

    const prediction = backend.infer(model, { observation: "deploy dashboard is open" });
    expect(prediction).toMatchObject({
      tool: "browser",
      summary: "click the deploy button",
      match: "exact",
    });
    expect(prediction?.confidence).toBe(1);
  });

  it("is deterministic and order-invariant under seeded shuffling", () => {
    const backend = new MockLocalModelBackend();
    const a = backend.train(generateSyntheticMovementDataset({ repeats: 5, seed: 7 }));
    const b = backend.train(generateSyntheticMovementDataset({ repeats: 5, seed: 42 }));

    // Different repetition order, same learned parameters.
    expect(a.parameters).toEqual(b.parameters);

    const predA = backend.infer(a, { observation: "build log is streaming" });
    const predB = backend.infer(b, { observation: "build log is streaming" });
    expect(predA).toEqual(predB);
    expect(predA).toMatchObject({ tool: "terminal", summary: "watch the build log" });
  });

  it("generalizes to a new but related observation via token overlap", () => {
    const backend = new MockLocalModelBackend();
    const model = backend.train(generateSyntheticMovementDataset({ repeats: 3 }));

    // Never seen verbatim, but shares tokens with "deploy dashboard is open".
    const prediction = backend.infer(model, { observation: "the deploy dashboard page is now open" });
    expect(prediction?.match).toBe("generalized");
    expect(prediction).toMatchObject({ tool: "browser", summary: "click the deploy button" });
    expect(prediction!.confidence).toBeGreaterThan(0);
    expect(prediction!.confidence).toBeLessThanOrEqual(1);
  });

  it("infers from event context when no explicit observation is given", () => {
    const backend = new MockLocalModelBackend();
    const model = backend.train(generateSyntheticMovementDataset({ repeats: 2 }));

    const prediction = backend.infer(model, {
      context: [
        { kind: "action", ts: 1, trajectoryId: "t", tool: "browser", summary: "did something" },
        { kind: "observation", ts: 2, trajectoryId: "t", source: "synthetic", summary: "confirmation dialog appeared" },
      ],
    });
    expect(prediction).toMatchObject({ tool: "browser", summary: "confirm the deployment", match: "exact" });
  });

  it("falls back to the most frequent action for an unrelated observation", () => {
    const backend = new MockLocalModelBackend();
    const events: ReplayTimelineEvent[] = [
      { kind: "observation", ts: 1, trajectoryId: "t", source: "s", summary: "alpha view" },
      { kind: "action", ts: 2, trajectoryId: "t", tool: "browser", summary: "click alpha" },
      { kind: "observation", ts: 3, trajectoryId: "t", source: "s", summary: "alpha view" },
      { kind: "action", ts: 4, trajectoryId: "t", tool: "browser", summary: "click alpha" },
      { kind: "observation", ts: 5, trajectoryId: "t", source: "s", summary: "beta view" },
      { kind: "action", ts: 6, trajectoryId: "t", tool: "terminal", summary: "run beta" },
    ];
    const model = backend.train(dataset(events));

    const prediction = backend.infer(model, { observation: "completely different zzz screen" });
    expect(prediction?.match).toBe("fallback");
    expect(prediction).toMatchObject({ tool: "browser", summary: "click alpha" });
    expect(prediction!.confidence).toBeLessThan(0.5);
  });

  it("returns undefined for an empty dataset", () => {
    const backend = new MockLocalModelBackend();
    const model = backend.train(dataset([]));
    expect(backend.infer(model, { observation: "anything" })).toBeUndefined();
  });

  it("survives a JSON serialization round-trip", () => {
    const backend = new MockLocalModelBackend();
    const model = backend.train(generateSyntheticMovementDataset({ repeats: 3 }));
    const restored = JSON.parse(JSON.stringify(model));

    expect(backend.infer(restored, { observation: "deploy dashboard is open" })).toEqual(
      backend.infer(model, { observation: "deploy dashboard is open" }),
    );
  });
});

describe("LocalModelBackendRegistry", () => {
  it("registers, resolves, and lists backends", () => {
    const registry = createDefaultLocalModelBackendRegistry();
    expect(registry.has("mock")).toBe(true);
    expect(registry.list()).toEqual(["mock"]);
    expect(registry.resolve("mock")).toBeInstanceOf(MockLocalModelBackend);
  });

  it("rejects duplicate registration and unknown lookups", () => {
    const registry = new LocalModelBackendRegistry();
    registry.register(new MockLocalModelBackend());
    expect(() => registry.register(new MockLocalModelBackend())).toThrow(/already registered/);
    expect(() => registry.resolve("nope")).toThrow(/Unknown local-model backend/);
  });
});
