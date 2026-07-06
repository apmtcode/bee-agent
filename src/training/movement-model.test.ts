import { describe, expect, it } from "vitest";
import type { ReplayManifest } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  MarkovMovementBackend,
  MovementBackendRegistry,
  buildMovementDataset,
  buildMovementDatasetFromTrajectories,
  deserializeMovementModel,
  evaluateMovementModel,
  movementBackendRegistry,
  tokenizeReplayEvent,
  tokenizeTrajectoryAction,
  type MovementDataset,
  type MovementModelBackend,
  type MovementSequence,
  type TrainedMovementModel,
} from "./movement-model.js";

function dataset(sequences: MovementSequence[]): MovementDataset {
  return { version: 1, sequences };
}

describe("MarkovMovementBackend training + replay (objective c)", () => {
  it("reproduces a single recorded trajectory exactly (replay fidelity)", async () => {
    const backend = new MarkovMovementBackend();
    const model = await backend.train(
      dataset([{ id: "t1", tokens: ["action:open", "action:search", "action:click", "action:submit"] }]),
    );

    expect(model.generate([])).toEqual(["action:open", "action:search", "action:click", "action:submit"]);
  });

  it("routes to the correct continuation given a distinguishing prefix", async () => {
    const backend = new MarkovMovementBackend();
    const model = await backend.train(
      dataset([
        { id: "a", tokens: ["action:open", "action:search", "action:click"] },
        { id: "b", tokens: ["action:type", "action:review", "action:cancel"] },
      ]),
      { order: 2 },
    );

    expect(model.generate(["action:type"])).toEqual(["action:review", "action:cancel"]);
    expect(model.generate(["action:open"])).toEqual(["action:search", "action:click"]);
  });

  it("predicts the most frequent successor deterministically", async () => {
    const backend = new MarkovMovementBackend();
    const model = await backend.train(
      dataset([
        { id: "a", tokens: ["action:open", "action:save"] },
        { id: "b", tokens: ["action:open", "action:save"] },
        { id: "c", tokens: ["action:open", "action:quit"] },
      ]),
      { order: 1 },
    );

    const prediction = model.predictNext(["action:open"]);
    expect(prediction?.token).toBe("action:save");
    expect(prediction?.probability).toBeCloseTo(2 / 3);
    expect(prediction?.order).toBe(1);
  });
});

describe("MarkovMovementBackend generalization (objective d)", () => {
  it("generalizes to a novel prefix via suffix backoff", async () => {
    const backend = new MarkovMovementBackend();
    // Both trajectories end "...search -> click -> submit". A never-seen opening
    // step should still complete through the shared learned suffix.
    const model = await backend.train(
      dataset([
        { id: "a", tokens: ["action:open", "action:search", "action:click", "action:submit"] },
        { id: "b", tokens: ["action:type", "action:search", "action:click", "action:submit"] },
      ]),
      { order: 3 },
    );

    const continuation = model.generate(["action:focus", "action:search"]);
    expect(continuation).toEqual(["action:click", "action:submit"]);
  });

  it("returns undefined for a wholly unseen context", async () => {
    const backend = new MarkovMovementBackend();
    const model = await backend.train(dataset([{ id: "a", tokens: ["action:open", "action:save"] }]), { order: 2 });

    expect(model.predictNext(["action:never-seen-token"])).toBeUndefined();
  });

  it("up-weights higher-reward trajectories during training", async () => {
    const backend = new MarkovMovementBackend();
    const model = await backend.train(
      dataset([
        { id: "good", tokens: ["action:start", "action:win"], weight: 5 },
        { id: "bad", tokens: ["action:start", "action:lose"], weight: 1 },
      ]),
      { order: 1 },
    );

    expect(model.predictNext(["action:start"])?.token).toBe("action:win");
  });

  it("drops zero-weight trajectories", async () => {
    const backend = new MarkovMovementBackend();
    const model = await backend.train(
      dataset([
        { id: "keep", tokens: ["action:a", "action:b"], weight: 1 },
        { id: "drop", tokens: ["action:a", "action:c"], weight: 0 },
      ]),
      { order: 1 },
    );

    expect(model.generate(["action:a"])).toEqual(["action:b"]);
  });
});

describe("serialization", () => {
  it("round-trips a trained model to JSON and back", async () => {
    const backend = new MarkovMovementBackend();
    const model = await backend.train(
      dataset([{ id: "t", tokens: ["action:open", "action:search", "action:submit"] }]),
    );

    const restored = deserializeMovementModel(JSON.parse(JSON.stringify(model.serialize())));

    expect(restored.order).toBe(model.order);
    expect(restored.backendId).toBe(model.backendId);
    expect(restored.generate([])).toEqual(model.generate([]));
  });
});

describe("MovementBackendRegistry", () => {
  it("ships with a default markov backend", () => {
    const registry = new MovementBackendRegistry();
    expect(registry.has("markov")).toBe(true);
    expect(registry.resolve().id).toBe("markov");
    expect(movementBackendRegistry.resolve("markov").id).toBe("markov");
  });

  it("registers and selects a pluggable backend", async () => {
    const registry = new MovementBackendRegistry();
    const calls: string[] = [];
    const fake: MovementModelBackend = {
      id: "on-device-small",
      async train(): Promise<TrainedMovementModel> {
        calls.push("train");
        return new MarkovMovementBackend("on-device-small").train(dataset([{ id: "x", tokens: ["action:noop"] }]));
      },
    };
    registry.register(fake, { makeDefault: true });

    expect(registry.list()).toContain("on-device-small");
    expect(registry.resolve().id).toBe("on-device-small");
    const model = await registry.resolve().train(dataset([]));
    expect(model.backendId).toBe("on-device-small");
    expect(calls).toEqual(["train"]);
  });

  it("throws on an unknown backend id", () => {
    const registry = new MovementBackendRegistry();
    expect(() => registry.resolve("does-not-exist")).toThrow(/Unknown movement backend/);
  });
});

describe("dataset construction", () => {
  it("tokenizes replay manifests into action sequences", () => {
    const replay: ReplayManifest = {
      version: 1,
      sessionId: "s1",
      trajectoryIds: ["traj-1"],
      eventCount: 3,
      events: [
        { kind: "transcript", ts: 1, messageId: "m1", role: "user", content: "do it" },
        { kind: "observation", ts: 2, trajectoryId: "traj-1", source: "Screen Reader", summary: "saw button" },
        { kind: "action", ts: 3, trajectoryId: "traj-1", tool: "Click Button", summary: "clicked" },
      ],
    };

    const built = buildMovementDataset([replay]);
    expect(built.sequences).toHaveLength(1);
    expect(built.sequences[0].tokens).toEqual(["action:click-button"]);
    expect(built.sequences[0].id).toBe("traj-1");
  });

  it("optionally includes observation tokens", () => {
    const replay: ReplayManifest = {
      version: 1,
      sessionId: "s1",
      trajectoryIds: ["traj-1"],
      eventCount: 2,
      events: [
        { kind: "observation", ts: 1, trajectoryId: "traj-1", source: "screen", summary: "x" },
        { kind: "action", ts: 2, trajectoryId: "traj-1", tool: "click", summary: "y" },
      ],
    };

    const built = buildMovementDataset([replay], { includeObservations: true });
    expect(built.sequences[0].tokens).toEqual(["observation:screen", "action:click"]);
  });

  it("builds datasets from trajectory spans sorted by timestamp with reward weighting", () => {
    const trajectory: TrajectorySpan = {
      id: "t1",
      sessionId: "s1",
      createdAt: "2026-01-01T00:00:00.000Z",
      captureTier: "operator",
      observations: [],
      actions: [
        { kind: "action", tool: "second", summary: "", ts: 20 },
        { kind: "action", tool: "first", summary: "", ts: 10 },
      ],
      outcome: { status: "success", summary: "done", reward: 3 },
    };

    const built = buildMovementDatasetFromTrajectories([trajectory]);
    expect(built.sequences[0].tokens).toEqual(["action:first", "action:second"]);
    expect(built.sequences[0].weight).toBe(3);
  });

  it("exposes stable tokenizers", () => {
    expect(tokenizeTrajectoryAction("Open File")).toBe("action:open-file");
    expect(
      tokenizeReplayEvent({ kind: "action", ts: 1, trajectoryId: "t", tool: "Scroll Down", summary: "" }),
    ).toBe("action:scroll-down");
    expect(
      tokenizeReplayEvent({ kind: "transcript", ts: 1, messageId: "m", role: "user", content: "" }),
    ).toBeUndefined();
  });
});

describe("evaluateMovementModel", () => {
  it("scores perfect replay fidelity on training data", async () => {
    const backend = new MarkovMovementBackend();
    const sequences: MovementSequence[] = [
      { id: "a", tokens: ["action:open", "action:search", "action:submit"] },
    ];
    const model = await backend.train(dataset(sequences));

    const report = evaluateMovementModel(model, sequences);
    expect(report.evaluated).toBe(1);
    expect(report.tokenAccuracy).toBe(1);
    expect(report.exactSequenceMatch).toBe(1);
    expect(report.meanLogLikelihood).toBeLessThan(0);
  });

  it("measures generalization on held-out related trajectories", async () => {
    const backend = new MarkovMovementBackend();
    const train: MovementSequence[] = [
      { id: "a", tokens: ["action:open", "action:search", "action:click", "action:submit"] },
      { id: "b", tokens: ["action:type", "action:search", "action:click", "action:submit"] },
    ];
    const model = await backend.train(dataset(train), { order: 3 });

    // Held-out trajectory shares the learned "search -> click -> submit" suffix.
    const heldOut: MovementSequence[] = [
      { id: "c", tokens: ["action:focus", "action:search", "action:click", "action:submit"] },
    ];
    const report = evaluateMovementModel(model, heldOut);
    // The shared "search -> click -> submit" suffix is recovered even though the
    // opening step is novel, so the model generalizes on unseen trajectories.
    expect(report.tokenAccuracy).toBeGreaterThanOrEqual(0.5);
    expect(model.generate(["action:focus", "action:search"])).toEqual(["action:click", "action:submit"]);
  });
});
