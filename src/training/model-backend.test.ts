import { describe, expect, it } from "vitest";
import type { ReplayManifest } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  MarkovMovementBackend,
  type MovementDataset,
  datasetFromReplayManifests,
  datasetFromTrajectories,
  evaluateMovementFidelity,
  rolloutMovements,
} from "./model-backend.js";

function dataset(sequences: Array<{ id: string; tokens: string[] }>): MovementDataset {
  return { version: 1, sequences };
}

describe("MarkovMovementBackend", () => {
  it("trains a backend-tagged, serializable artifact with a sorted vocabulary", async () => {
    const backend = new MarkovMovementBackend();
    const artifact = await backend.train(
      dataset([{ id: "a", tokens: ["focus", "click", "type", "submit"] }]),
    );

    expect(artifact.backendId).toBe("markov");
    expect(artifact.order).toBe(1);
    expect(artifact.sequenceCount).toBe(1);
    expect(artifact.vocabulary).toEqual(["click", "focus", "submit", "type"]);
    // Round-trips through JSON (persistable to disk on-device).
    expect(JSON.parse(JSON.stringify(artifact))).toEqual(artifact);
  });

  it("repeats a recorded movement exactly via greedy rollout", async () => {
    const backend = new MarkovMovementBackend();
    const recorded = ["focus", "click", "type", "submit"];
    const artifact = await backend.train(dataset([{ id: "a", tokens: recorded }]), { order: 2 });

    const replayed = rolloutMovements(backend, artifact, { seed: ["focus"], maxSteps: 4 });
    expect(replayed).toEqual(recorded);
  });

  it("is deterministic: identical inputs yield identical predictions", async () => {
    const backend = new MarkovMovementBackend();
    const training = dataset([
      { id: "a", tokens: ["open", "click", "close"] },
      { id: "b", tokens: ["open", "click", "close"] },
    ]);
    const first = await backend.train(training);
    const second = await backend.train(training);
    expect(first).toEqual(second);

    const p1 = backend.predict(first, { history: ["open"] });
    const p2 = backend.predict(second, { history: ["open"] });
    expect(p1).toEqual(p2);
    expect(p1.token).toBe("click");
    expect(p1.confidence).toBe(1);
  });

  it("generalizes to a new-but-related prefix by backing off to shorter contexts", async () => {
    const backend = new MarkovMovementBackend();
    // Every "type" step is followed by "submit"; an order-2 context "menu type"
    // was never seen, so the model must back off to the order-1 context "type".
    const artifact = await backend.train(
      dataset([
        { id: "a", tokens: ["focus", "type", "submit"] },
        { id: "b", tokens: ["click", "type", "submit"] },
      ]),
      { order: 2 },
    );

    const prediction = backend.predict(artifact, { history: ["menu", "type"] });
    expect(prediction.token).toBe("submit");
    expect(prediction.confidence).toBe(1);
  });

  it("falls back to the unigram distribution for a fully unseen context", async () => {
    const backend = new MarkovMovementBackend();
    const artifact = await backend.train(
      dataset([{ id: "a", tokens: ["click", "click", "click", "type"] }]),
    );

    const prediction = backend.predict(artifact, { history: ["nothing-like-this"] });
    // "click" dominates the unigram (3 of 4 occurrences).
    expect(prediction.token).toBe("click");
    expect(prediction.distribution[0]).toEqual({ token: "click", probability: 0.75 });
  });

  it("breaks probability ties deterministically by token order", async () => {
    const backend = new MarkovMovementBackend();
    const artifact = await backend.train(
      dataset([
        { id: "a", tokens: ["start", "zebra"] },
        { id: "b", tokens: ["start", "alpha"] },
      ]),
    );
    const prediction = backend.predict(artifact, { history: ["start"] });
    expect(prediction.token).toBe("alpha");
    expect(prediction.distribution.map((c) => c.token)).toEqual(["alpha", "zebra"]);
  });

  it("rejects artifacts trained by a different backend", async () => {
    const backend = new MarkovMovementBackend();
    const artifact = await backend.train(dataset([{ id: "a", tokens: ["x"] }]));
    const alien = { ...artifact, backendId: "some-other-backend" };
    expect(() => backend.predict(alien, { history: [] })).toThrow(/some-other-backend/);
  });
});

describe("rolloutMovements", () => {
  it("respects maxSteps and the stop token", async () => {
    const backend = new MarkovMovementBackend();
    const artifact = await backend.train(
      dataset([{ id: "a", tokens: ["a", "b", "c", "stop", "d"] }]),
      { order: 1 },
    );

    expect(rolloutMovements(backend, artifact, { seed: ["a"], maxSteps: 2 })).toEqual(["a", "b"]);

    const stopped = rolloutMovements(backend, artifact, { seed: ["a"], maxSteps: 10, stopToken: "stop" });
    expect(stopped).toEqual(["a", "b", "c", "stop"]);
  });

  it("truncates an over-long seed to maxSteps without predicting", async () => {
    const backend = new MarkovMovementBackend();
    const artifact = await backend.train(dataset([{ id: "a", tokens: ["a", "b"] }]));
    expect(rolloutMovements(backend, artifact, { seed: ["a", "b", "c"], maxSteps: 2 })).toEqual(["a", "b"]);
  });
});

describe("dataset builders", () => {
  it("extracts action sequences from replay manifests, ignoring non-actions", () => {
    const manifest: ReplayManifest = {
      version: 1,
      sessionId: "session-1",
      trajectoryIds: ["t1"],
      eventCount: 4,
      events: [
        { kind: "transcript", ts: 1, messageId: "m1", role: "user", content: "go" },
        { kind: "observation", ts: 2, trajectoryId: "t1", source: "screen", summary: "menu open" },
        { kind: "action", ts: 3, trajectoryId: "t1", tool: "click", summary: "click file" },
        { kind: "action", ts: 4, trajectoryId: "t1", tool: "type", summary: "type name" },
      ],
    };

    const ds = datasetFromReplayManifests([manifest]);
    expect(ds.sequences).toEqual([{ id: "session-1", tokens: ["click", "type"] }]);
  });

  it("extracts reviewed actions from trajectories in timestamp order", () => {
    const trajectory: TrajectorySpan = {
      id: "t1",
      sessionId: "s1",
      createdAt: "2026-01-01T00:00:00.000Z",
      captureTier: "full",
      observations: [],
      actions: [
        { kind: "action", tool: "raw-a", summary: "", ts: 1 },
        { kind: "action", tool: "raw-b", summary: "", ts: 2 },
      ],
      review: {
        status: "approved",
        reviewedAt: "2026-01-01T00:01:00.000Z",
        reviewedBy: "reviewer",
        redactedActions: [
          { ts: 20, tool: "submit", summary: "" },
          { ts: 10, tool: "click", summary: "" },
        ],
      },
    };

    const ds = datasetFromTrajectories([trajectory]);
    // Prefers redacted actions and sorts by ts.
    expect(ds.sequences).toEqual([{ id: "t1", tokens: ["click", "submit"] }]);
  });

  it("uses raw actions when a trajectory has no review", () => {
    const trajectory: TrajectorySpan = {
      id: "t2",
      sessionId: "s1",
      createdAt: "2026-01-01T00:00:00.000Z",
      captureTier: "full",
      observations: [],
      actions: [
        { kind: "action", tool: "open", summary: "", ts: 2 },
        { kind: "action", tool: "focus", summary: "", ts: 1 },
      ],
    };
    const ds = datasetFromTrajectories([trajectory]);
    expect(ds.sequences).toEqual([{ id: "t2", tokens: ["focus", "open"] }]);
  });
});

describe("evaluateMovementFidelity", () => {
  it("reports perfect fidelity when repeating training sequences", async () => {
    const backend = new MarkovMovementBackend();
    const sequences = [{ id: "a", tokens: ["focus", "click", "type", "submit"] }];
    const artifact = await backend.train(dataset(sequences), { order: 2 });

    const report = evaluateMovementFidelity(backend, artifact, sequences);
    expect(report.accuracy).toBe(1);
    expect(report.total).toBe(4);
    expect(report.correct).toBe(4);
  });

  it("measures generalization on held-out but related sequences", async () => {
    const backend = new MarkovMovementBackend();
    const train = dataset([
      { id: "a", tokens: ["focus", "click", "type", "submit"] },
      { id: "b", tokens: ["focus", "click", "type", "submit"] },
    ]);
    const artifact = await backend.train(train, { order: 1 });

    // A related-but-unseen trajectory: same click->type->submit tail.
    const heldOut = [{ id: "held", tokens: ["focus", "click", "type", "submit"] }];
    const report = evaluateMovementFidelity(backend, artifact, heldOut);
    expect(report.accuracy).toBeGreaterThan(0.5);
    expect(report.perSequence[0]?.id).toBe("held");
  });

  it("returns zero accuracy for an empty evaluation set", async () => {
    const backend = new MarkovMovementBackend();
    const artifact = await backend.train(dataset([{ id: "a", tokens: ["x"] }]));
    const report = evaluateMovementFidelity(backend, artifact, []);
    expect(report).toEqual({ total: 0, correct: 0, accuracy: 0, perSequence: [] });
  });
});
