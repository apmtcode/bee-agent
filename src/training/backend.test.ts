import { describe, expect, it } from "vitest";
import {
  MockNearestNeighborBackend,
  createDefaultBackendRegistry,
  evaluateMovementModel,
  extractMovementDataset,
  splitMovementDataset,
  tokenizeContext,
} from "./backend.js";
import { synthesizeMovementManifest } from "./synthetic.js";

describe("extractMovementDataset", () => {
  it("emits one sample per action event with the preceding context", () => {
    const manifest = synthesizeMovementManifest({ seed: 7, trajectoryCount: 2, stepsPerTrajectory: 3 });
    const dataset = extractMovementDataset(manifest);

    // 2 trajectories * 3 actions each.
    expect(dataset.sampleCount).toBe(6);
    expect(dataset.samples).toHaveLength(6);

    const first = dataset.samples[0]!;
    expect(first.stepIndex).toBe(0);
    // The goal transcript + the first observation precede the first action.
    expect(first.context.length).toBeGreaterThanOrEqual(2);
    expect(first.context.some((event) => event.kind === "observation")).toBe(true);
    expect(first.action.tool).toMatch(/\./);
  });

  it("respects the context window", () => {
    const manifest = synthesizeMovementManifest({ seed: 3, trajectoryCount: 1, stepsPerTrajectory: 5 });
    const dataset = extractMovementDataset(manifest, { contextWindow: 2 });
    for (const sample of dataset.samples) {
      expect(sample.context.length).toBeLessThanOrEqual(2);
    }
  });

  it("increments step index within a trajectory", () => {
    const manifest = synthesizeMovementManifest({ seed: 9, trajectoryCount: 1, stepsPerTrajectory: 4 });
    const dataset = extractMovementDataset(manifest);
    expect(dataset.samples.map((sample) => sample.stepIndex)).toEqual([0, 1, 2, 3]);
  });
});

describe("MockNearestNeighborBackend", () => {
  it("replays recorded movements exactly for trained contexts", async () => {
    const manifest = synthesizeMovementManifest({ seed: 11, trajectoryCount: 3, stepsPerTrajectory: 4 });
    const dataset = extractMovementDataset(manifest);
    const model = await new MockNearestNeighborBackend().train(dataset);

    const report = evaluateMovementModel(model, dataset.samples);
    // Memorized training set → perfect reproduction.
    expect(report.toolAccuracy).toBe(1);
    expect(report.summaryAccuracy).toBe(1);
    expect(report.exactMatchRate).toBe(1);
    expect(report.meanConfidence).toBe(1);
  });

  it("is deterministic across retrains", async () => {
    const manifest = synthesizeMovementManifest({ seed: 5 });
    const dataset = extractMovementDataset(manifest);
    const backend = new MockNearestNeighborBackend();
    const a = await backend.train(dataset);
    const b = await backend.train(dataset);
    const context = dataset.samples[0]!.context;
    expect(a.predict(context)).toEqual(b.predict(context));
  });

  it("returns undefined for an empty model", async () => {
    const model = await new MockNearestNeighborBackend().train({ version: 1, sampleCount: 0, samples: [] });
    expect(model.predict([{ kind: "observation", text: "anything", source: "ui" }])).toBeUndefined();
  });

  it("round-trips through serialize/restore", async () => {
    const manifest = synthesizeMovementManifest({ seed: 21 });
    const dataset = extractMovementDataset(manifest);
    const backend = new MockNearestNeighborBackend();
    const trained = await backend.train(dataset);
    const restored = backend.restore(trained.serialize());

    for (const sample of dataset.samples) {
      expect(restored.predict(sample.context)).toEqual(trained.predict(sample.context));
    }
  });

  it("generalizes to new-but-related contexts via nearest neighbor", async () => {
    // Train on one scenario; evaluate on a *different seed but same scenario* run.
    const trainManifest = synthesizeMovementManifest({ seed: 100, scenario: "editor", trajectoryCount: 6, stepsPerTrajectory: 5 });
    const holdoutManifest = synthesizeMovementManifest({ seed: 200, scenario: "editor", trajectoryCount: 3, stepsPerTrajectory: 5 });

    const model = await new MockNearestNeighborBackend().train(extractMovementDataset(trainManifest));
    const report = evaluateMovementModel(model, extractMovementDataset(holdoutManifest).samples);

    // Every held-out context gets a neighbor prediction (never undefined) ...
    expect(report.total).toBeGreaterThan(0);
    expect(report.predictions.every((p) => p.predictedTool !== undefined)).toBe(true);
    // ... and the shared vocabulary yields non-trivial confidence.
    expect(report.meanConfidence).toBeGreaterThan(0);
    // The five-tool space means random guessing would sit near 0.2; the model
    // should meaningfully beat that on related movements.
    expect(report.toolAccuracy).toBeGreaterThan(0.2);
  });
});

describe("splitMovementDataset", () => {
  it("deterministically partitions into train and holdout", () => {
    const dataset = extractMovementDataset(
      synthesizeMovementManifest({ seed: 42, trajectoryCount: 4, stepsPerTrajectory: 4 }),
    );
    const { train, holdout } = splitMovementDataset(dataset, 4);
    expect(train.length + holdout.length).toBe(dataset.sampleCount);
    expect(holdout.length).toBe(Math.floor(dataset.sampleCount / 4));
    // Repeated calls are stable.
    const second = splitMovementDataset(dataset, 4);
    expect(second.holdout).toEqual(holdout);
  });
});

describe("createDefaultBackendRegistry", () => {
  it("registers the mock backend and enforces required lookups", () => {
    const registry = createDefaultBackendRegistry();
    expect(registry.get("mock-nearest-neighbor")).toBeInstanceOf(MockNearestNeighborBackend);
    expect(registry.list()).toHaveLength(1);
    expect(() => registry.require("does-not-exist")).toThrow(/no local-model backend/);
  });
});

describe("tokenizeContext", () => {
  it("prefixes kind and source tokens and lowercases text", () => {
    const tokens = tokenizeContext([{ kind: "action", text: "Mouse.Click on Button", source: "mouse.click" }]);
    expect(tokens).toContain("kind:action");
    expect(tokens).toContain("src:mouse");
    expect(tokens).toContain("click");
    expect(tokens).toContain("button");
  });
});
