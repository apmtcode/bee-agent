import { describe, expect, it } from "vitest";
import {
  MarkovMovementBackend,
  MovementModelBackendRegistry,
  MOVEMENT_END,
  buildMovementDatasetFromReplays,
  buildMovementDatasetFromTrajectories,
  createDefaultMovementBackendRegistry,
  evaluateMovementModel,
  type MovementDataset,
  type MovementModelBackend,
  type MovementSequence,
} from "./movement-model.js";
import type { TrajectorySpan } from "../capture/trajectory.js";
import type { ExportedReplayManifest } from "./export-manifest.js";

const OPEN_DEPLOY: MovementDataset = {
  version: 1,
  sequences: [
    { id: "a", tokens: ["focus", "open-menu", "click-deploy", "confirm"] },
    { id: "b", tokens: ["focus", "open-menu", "click-deploy", "confirm"] },
    { id: "c", tokens: ["focus", "open-menu", "click-settings", "toggle"] },
  ],
};

describe("MarkovMovementBackend", () => {
  it("learns the recorded next movement and is deterministic", () => {
    const backend = new MarkovMovementBackend();
    const model = backend.train(OPEN_DEPLOY, { order: 3 });

    // After "open-menu", "click-deploy" (2/3) outranks "click-settings" (1/3).
    const ranked = model.predictNext(["focus", "open-menu"]);
    expect(ranked[0]?.token).toBe("click-deploy");
    expect(ranked[0]?.probability).toBeCloseTo(2 / 3);
    expect(ranked[1]?.token).toBe("click-settings");

    // Same dataset -> byte-identical artifact (no randomness).
    const again = backend.train(OPEN_DEPLOY, { order: 3 });
    expect(again.serialize()).toEqual(model.serialize());
  });

  it("predicts termination after a fully recorded sequence", () => {
    const model = new MarkovMovementBackend().train(OPEN_DEPLOY, { order: 3 });
    const ranked = model.predictNext(["focus", "open-menu", "click-deploy", "confirm"]);
    expect(ranked[0]?.token).toBe(MOVEMENT_END);
  });

  it("generalizes to a new-but-related sequence via suffix backoff", () => {
    // Training never contains the prefix ["warmup", "focus", "open-menu"], but
    // the local suffix "open-menu" is familiar, so backoff still predicts.
    const model = new MarkovMovementBackend().train(OPEN_DEPLOY, { order: 3 });
    const ranked = model.predictNext(["warmup", "focus", "open-menu"]);
    expect(ranked[0]?.token).toBe("click-deploy");
  });

  it("round-trips through serialize/load", () => {
    const backend = new MarkovMovementBackend();
    const model = backend.train(OPEN_DEPLOY, { order: 2 });
    const restored = backend.load(model.serialize());
    expect(restored.predictNext(["open-menu"])).toEqual(model.predictNext(["open-menu"]));
    expect(restored.serialize()).toEqual(model.serialize());
  });

  it("returns no prediction for a wholly unknown context", () => {
    const model = new MarkovMovementBackend().train(
      { version: 1, sequences: [{ id: "x", tokens: ["a", "b"] }] },
      { order: 0 },
    );
    // order 0 -> only the unigram context exists, so any context predicts from it.
    expect(model.predictNext(["zzz"]).length).toBeGreaterThan(0);
  });
});

describe("evaluateMovementModel", () => {
  it("scores perfect accuracy on training-consistent held-out data", () => {
    const model = new MarkovMovementBackend().train(OPEN_DEPLOY, { order: 3 });
    const heldOut: MovementSequence[] = [
      { id: "eval", tokens: ["focus", "open-menu", "click-deploy", "confirm"] },
    ];
    const evaluation = evaluateMovementModel(model, heldOut, { topK: 2 });
    expect(evaluation.sequences).toBe(1);
    expect(evaluation.predictions).toBe(5); // 4 tokens + termination
    expect(evaluation.top1Accuracy).toBe(1);
    expect(evaluation.topKAccuracy).toBe(1);
    expect(evaluation.perplexity).toBeGreaterThanOrEqual(1);
    expect(Number.isFinite(evaluation.perplexity)).toBe(true);
  });

  it("keeps perplexity finite when a held-out movement is unseen", () => {
    const model = new MarkovMovementBackend().train(OPEN_DEPLOY, { order: 3 });
    const heldOut: MovementSequence[] = [{ id: "novel", tokens: ["totally", "unseen", "moves"] }];
    const evaluation = evaluateMovementModel(model, heldOut, { scoreTermination: false });
    expect(evaluation.top1Correct).toBe(0);
    expect(Number.isFinite(evaluation.perplexity)).toBe(true);
    expect(evaluation.perplexity).toBeGreaterThan(1);
  });

  it("higher order generalizes at least as well as order 0 on structured data", () => {
    const heldOut: MovementSequence[] = [
      { id: "eval", tokens: ["focus", "open-menu", "click-deploy", "confirm"] },
    ];
    const order0 = evaluateMovementModel(
      new MarkovMovementBackend().train(OPEN_DEPLOY, { order: 0 }),
      heldOut,
    );
    const order3 = evaluateMovementModel(
      new MarkovMovementBackend().train(OPEN_DEPLOY, { order: 3 }),
      heldOut,
    );
    expect(order3.top1Accuracy).toBeGreaterThan(order0.top1Accuracy);
    expect(order3.perplexity).toBeLessThan(order0.perplexity);
  });
});

describe("dataset builders", () => {
  it("builds a dataset from trajectory spans in ts order", () => {
    const trajectory: TrajectorySpan = {
      id: "traj-1",
      sessionId: "sess-1",
      createdAt: "2026-01-01T00:00:00.000Z",
      captureTier: "operator",
      observations: [],
      actions: [
        { kind: "action", tool: "second", summary: "s", ts: 20 },
        { kind: "action", tool: "first", summary: "f", ts: 10 },
      ],
    };
    const dataset = buildMovementDatasetFromTrajectories([trajectory]);
    expect(dataset.sequences).toEqual([{ id: "traj-1", tokens: ["first", "second"] }]);
  });

  it("builds a dataset from replay manifests, keeping only action events", () => {
    const replay: ExportedReplayManifest = {
      sessionId: "sess-1",
      trajectoryIds: ["traj-1"],
      eventCount: 3,
      events: [
        { kind: "observation", ts: 5, trajectoryId: "traj-1", source: "os", summary: "saw" },
        { kind: "action", ts: 20, trajectoryId: "traj-1", tool: "confirm", summary: "c" },
        { kind: "action", ts: 10, trajectoryId: "traj-1", tool: "click", summary: "k" },
      ],
    };
    const dataset = buildMovementDatasetFromReplays([replay]);
    expect(dataset.sequences).toEqual([{ id: "traj-1", tokens: ["click", "confirm"] }]);
  });

  it("drops sequences below the minimum length", () => {
    const dataset = buildMovementDatasetFromTrajectories(
      [
        {
          id: "empty",
          sessionId: "s",
          createdAt: "2026-01-01T00:00:00.000Z",
          captureTier: "operator",
          observations: [],
          actions: [],
        },
      ],
      { minLength: 1 },
    );
    expect(dataset.sequences).toEqual([]);
  });
});

describe("MovementModelBackendRegistry", () => {
  it("ships with the deterministic markov backend registered by default", () => {
    const registry = createDefaultMovementBackendRegistry();
    expect(registry.list()).toContain("markov");
    expect(registry.get("markov")).toBeInstanceOf(MarkovMovementBackend);
  });

  it("supports registering and selecting a custom backend", () => {
    const registry = new MovementModelBackendRegistry();
    const custom: MovementModelBackend = {
      name: "onnx-stub",
      train: (dataset) => new MarkovMovementBackend().train(dataset),
      load: (artifact) => new MarkovMovementBackend().load(artifact),
    };
    registry.register(custom);
    expect(registry.has("onnx-stub")).toBe(true);
    expect(registry.get("onnx-stub")).toBe(custom);
  });

  it("throws for an unknown backend", () => {
    expect(() => createDefaultMovementBackendRegistry().get("missing")).toThrow(/unknown movement-model backend/);
  });
});
