import { describe, expect, it } from "vitest";
import {
  MOVEMENT_END,
  MarkovMovementBackend,
  buildMovementDatasetFromReplays,
  buildMovementDatasetFromTrajectories,
  evaluateGeneralization,
  evaluateNextTokenAccuracy,
  generateSyntheticMovementSequences,
  getMovementBackend,
  listMovementBackends,
  movementTokenFromAction,
  registerMovementBackend,
  trainMovementModel,
  type MovementDataset,
  type MovementModelBackend,
} from "./movement-model.js";
import type { ReplayManifest } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

describe("movementTokenFromAction", () => {
  it("normalizes tool + summary into a stable slugged token", () => {
    expect(movementTokenFromAction({ tool: "Mouse", summary: "Move Left fast" })).toBe("mouse:move-left-fast");
    // Related summaries collapse to the same token (basis for generalization).
    expect(movementTokenFromAction({ tool: "mouse", summary: "click!!" })).toBe("mouse:click");
    expect(movementTokenFromAction({ tool: "key", summary: "" })).toBe("key");
  });
});

describe("dataset builders", () => {
  it("builds an ordered token sequence from trajectory actions", () => {
    const trajectory: TrajectorySpan = {
      id: "t1",
      sessionId: "s1",
      createdAt: "2026-01-01T00:00:00.000Z",
      captureTier: "operator",
      observations: [],
      actions: [
        { kind: "action", tool: "mouse", summary: "click", ts: 30 },
        { kind: "action", tool: "mouse", summary: "move left", ts: 10 },
        { kind: "action", tool: "mouse", summary: "move down", ts: 20 },
      ],
      outcome: { status: "success", summary: "ok", reward: 1 },
    };
    const dataset = buildMovementDatasetFromTrajectories([trajectory]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]!.tokens).toEqual(["mouse:move-left", "mouse:move-down", "mouse:click"]);
    expect(dataset.sequences[0]!.reward).toBe(1);
  });

  it("drops empty trajectories and reads action events from replays", () => {
    const empty: TrajectorySpan = {
      id: "empty",
      sessionId: "s2",
      createdAt: "2026-01-01T00:00:00.000Z",
      captureTier: "operator",
      observations: [],
      actions: [],
    };
    expect(buildMovementDatasetFromTrajectories([empty]).sequences).toHaveLength(0);

    const replay: ReplayManifest = {
      version: 1,
      sessionId: "s3",
      trajectoryIds: ["t3"],
      eventCount: 3,
      events: [
        { kind: "observation", ts: 1, trajectoryId: "t3", source: "browser", summary: "opened" },
        { kind: "action", ts: 3, trajectoryId: "t3", tool: "mouse", summary: "click" },
        { kind: "action", ts: 2, trajectoryId: "t3", tool: "mouse", summary: "move up" },
      ],
    };
    const dataset = buildMovementDatasetFromReplays([replay]);
    expect(dataset.sequences[0]!.tokens).toEqual(["mouse:move-up", "mouse:click"]);
  });
});

describe("MarkovMovementBackend", () => {
  it("learns transitions and predicts the next movement deterministically", () => {
    const dataset: MovementDataset = {
      version: 1,
      sequences: [
        { id: "a", tokens: ["focus", "mouse:move-right", "mouse:click", "key:confirm"] },
        { id: "b", tokens: ["focus", "mouse:move-right", "mouse:click", "key:confirm"] },
      ],
    };
    const { model, artifact } = trainMovementModel(dataset, { maxOrder: 2 });
    expect(artifact.backendId).toBe("markov-mock");
    expect(artifact.sequenceCount).toBe(2);
    expect(artifact.vocabSize).toBe(4);

    // Deterministic argmax over a context it has seen.
    expect(model.predictNext(["focus"]).token).toBe("mouse:move-right");
    expect(model.predictNext(["focus", "mouse:move-right"]).token).toBe("mouse:click");
    expect(model.predictNext(["mouse:click"]).token).toBe("key:confirm");
    // Empty context resolves against the START-padded start distribution.
    expect(model.predictNext([]).token).toBe("focus");

    const prediction = model.predictNext(["focus", "mouse:move-right"]);
    expect(prediction.probability).toBeCloseTo(1, 5);
    expect(prediction.distribution[0]!.token).toBe("mouse:click");
  });

  it("generalizes to an unseen context via n-gram backoff", () => {
    const dataset: MovementDataset = {
      version: 1,
      sequences: [{ id: "a", tokens: ["focus", "mouse:move-right", "mouse:click", "key:confirm"] }],
    };
    const { model } = trainMovementModel(dataset, { maxOrder: 3 });
    // "window:opened" was never seen, but the suffix "mouse:click" was — backoff
    // still produces the learned continuation instead of giving up.
    const prediction = model.predictNext(["window:opened", "mouse:click"]);
    expect(prediction.token).toBe("key:confirm");
    expect(prediction.order).toBeLessThan(3);
  });

  it("generates a full movement rollout that stops at the natural end", () => {
    const dataset: MovementDataset = {
      version: 1,
      sequences: [{ id: "a", tokens: ["focus", "mouse:move-right", "mouse:click", "key:confirm"] }],
    };
    const { model } = trainMovementModel(dataset, { maxOrder: 3 });
    const rollout = model.generate(["focus"], 20);
    expect(rollout).toEqual(["mouse:move-right", "mouse:click", "key:confirm"]);
    expect(rollout).not.toContain(MOVEMENT_END);
  });

  it("round-trips through a JSON-serialized artifact", () => {
    const dataset: MovementDataset = {
      version: 1,
      sequences: [{ id: "a", tokens: ["focus", "mouse:move-right", "mouse:click"] }],
    };
    const backend = new MarkovMovementBackend();
    const artifact = backend.train(dataset, { maxOrder: 2 });
    const roundTripped = JSON.parse(JSON.stringify(artifact));
    const model = backend.load(roundTripped);
    expect(model.predictNext(["focus"]).token).toBe("mouse:move-right");
  });

  it("returns an empty prediction when there is no signal at all", () => {
    const backend = new MarkovMovementBackend();
    const artifact = backend.train({ version: 1, sequences: [] }, { maxOrder: 2 });
    const model = backend.load(artifact);
    const prediction = model.predictNext(["anything"]);
    expect(prediction.token).toBeUndefined();
    expect(prediction.distribution).toEqual([]);
  });

  it("rejects artifacts from a different backend", () => {
    const backend = new MarkovMovementBackend();
    expect(() =>
      backend.load({
        version: 1,
        backendId: "some-other",
        maxOrder: 1,
        vocabSize: 0,
        sequenceCount: 0,
        tokenCount: 0,
        params: {},
      }),
    ).toThrow(/cannot load artifact/);
  });
});

describe("backend registry", () => {
  it("exposes the built-in mock and supports pluggable registration", () => {
    expect(listMovementBackends()).toContain("markov-mock");
    expect(getMovementBackend("markov-mock")).toBeInstanceOf(MarkovMovementBackend);

    const custom: MovementModelBackend = {
      id: "test-echo-backend",
      train: (dataset) => ({
        version: 1,
        backendId: "test-echo-backend",
        maxOrder: 0,
        vocabSize: 0,
        sequenceCount: dataset.sequences.length,
        tokenCount: 0,
        params: null,
      }),
      load: () => ({
        backendId: "test-echo-backend",
        maxOrder: 0,
        predictNext: () => ({ token: "echo", probability: 1, order: 0, distribution: [{ token: "echo", probability: 1 }] }),
        generate: () => ["echo"],
      }),
    };
    registerMovementBackend(custom);
    expect(listMovementBackends()).toContain("test-echo-backend");
    const { model } = trainMovementModel({ version: 1, sequences: [] }, { backendId: "test-echo-backend" });
    expect(model.predictNext([]).token).toBe("echo");
  });

  it("throws on an unknown backend id", () => {
    expect(() => getMovementBackend("nope")).toThrow(/unknown movement backend/);
  });
});

describe("synthetic generator", () => {
  it("is deterministic for a fixed seed and honors length bounds", () => {
    const a = generateSyntheticMovementSequences({ sequenceCount: 5, seed: 42, minLength: 4, maxLength: 8 });
    const b = generateSyntheticMovementSequences({ sequenceCount: 5, seed: 42, minLength: 4, maxLength: 8 });
    expect(a).toEqual(b);
    expect(a.sequences).toHaveLength(5);
    for (const sequence of a.sequences) {
      expect(sequence.tokens.length).toBeGreaterThanOrEqual(4);
      expect(sequence.tokens.length).toBeLessThanOrEqual(8);
      expect(sequence.tokens).not.toContain(MOVEMENT_END);
      expect(sequence.tokens[0]).toBe("focus");
    }
  });

  it("varies with the seed", () => {
    const a = generateSyntheticMovementSequences({ sequenceCount: 5, seed: 1 });
    const b = generateSyntheticMovementSequences({ sequenceCount: 5, seed: 2 });
    expect(a).not.toEqual(b);
  });
});

describe("generalization eval harness", () => {
  it("measures next-token accuracy on held-out data", () => {
    const dataset = generateSyntheticMovementSequences({ sequenceCount: 40, seed: 7, noise: 0.05 });
    const { model } = trainMovementModel(dataset, { maxOrder: 3 });
    const result = evaluateNextTokenAccuracy(model, dataset.sequences);
    expect(result.total).toBeGreaterThan(0);
    expect(result.accuracy).toBeGreaterThan(0.7);
  });

  it("shows positive lift over a unigram baseline on held-out sequences", () => {
    const dataset = generateSyntheticMovementSequences({ sequenceCount: 60, seed: 11, noise: 0.08 });
    const report = evaluateGeneralization(dataset, { maxOrder: 3, trainRatio: 0.7 });
    expect(report.trainSequences).toBeGreaterThan(0);
    expect(report.evalSequences).toBeGreaterThan(0);
    // The model must beat the unigram baseline on data it never trained on —
    // direct evidence it generalized transition structure.
    expect(report.modelAccuracy).toBeGreaterThan(report.baselineAccuracy);
    expect(report.lift).toBeGreaterThan(0.1);
  });
});
