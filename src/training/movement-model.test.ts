import { describe, expect, it } from "vitest";
import {
  MarkovMovementBackend,
  buildMovementDatasetFromReplays,
  buildMovementDatasetFromTrajectories,
  evaluateMovementModel,
  movementTokenFor,
  type MovementDataset,
} from "./movement-model.js";
import type { ReplayManifest } from "../capture/replay.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";

function dataset(sequences: { id: string; tokens: string[] }[]): MovementDataset {
  return {
    version: 1,
    sequences: sequences.map((sequence) => ({
      id: sequence.id,
      steps: sequence.tokens.map((token) => ({ token })),
    })),
  };
}

describe("movementTokenFor", () => {
  it("builds a canonical tool.gesture:target token from metadata", () => {
    expect(
      movementTokenFor({ tool: "device", summary: "tapped Submit", metadata: { gesture: "tap", target: "Submit" } }),
    ).toBe("device.tap:submit");
  });

  it("falls back to the summary when no target metadata is present", () => {
    expect(movementTokenFor({ tool: "browser", summary: "Clicked Save Button" })).toBe("browser:clicked-save-button");
  });

  it("uses direction when a target is absent", () => {
    expect(
      movementTokenFor({ tool: "device", summary: "swiped up", metadata: { gesture: "swipe", direction: "up" } }),
    ).toBe("device.swipe:up");
  });
});

describe("MarkovMovementBackend", () => {
  it("reproduces a recorded movement sequence when seeded with its prefix", () => {
    const backend = new MarkovMovementBackend();
    const model = backend.train(dataset([{ id: "t1", tokens: ["open", "type", "submit", "confirm"] }]));
    expect(model.generate(["open"], 10)).toEqual(["type", "submit", "confirm"]);
  });

  it("stops generating at the learned end of a sequence", () => {
    const backend = new MarkovMovementBackend();
    const model = backend.train(dataset([{ id: "t1", tokens: ["a", "b", "c"] }]));
    // Even with a large budget it must not loop past the recorded end.
    expect(model.generate(["a"], 100)).toEqual(["b", "c"]);
  });

  it("predicts the most-likely next movement with a probability and backoff order", () => {
    const backend = new MarkovMovementBackend();
    const model = backend.train(
      dataset([
        { id: "t1", tokens: ["open", "type", "submit"] },
        { id: "t2", tokens: ["open", "type", "cancel"] },
        { id: "t3", tokens: ["open", "type", "submit"] },
      ]),
    );
    const prediction = model.predictNext(["open", "type"]);
    expect(prediction?.token).toBe("submit"); // seen 2x vs cancel 1x
    expect(prediction?.order).toBe(2);
    expect(prediction!.probability).toBeGreaterThan(0.5);
  });

  it("generalises to a new-but-related context via backoff to a recorded suffix", () => {
    const backend = new MarkovMovementBackend();
    const model = backend.train(
      dataset([
        { id: "t1", tokens: ["login", "type", "submit"] },
        { id: "t2", tokens: ["search", "type", "submit"] },
      ]),
    );
    // "checkout type" was never recorded, but "type -> submit" was: backoff generalises.
    const prediction = model.predictNext(["checkout", "type"]);
    expect(prediction?.token).toBe("submit");
    expect(prediction?.order).toBe(1); // backed off from order 2 to order 1
  });

  it("returns no prediction for an empty model", () => {
    const backend = new MarkovMovementBackend();
    const model = backend.train({ version: 1, sequences: [] });
    expect(model.predictNext(["anything"])).toBeUndefined();
    expect(model.generate([], 5)).toEqual([]);
  });

  it("is deterministic across repeated training runs", () => {
    const backend = new MarkovMovementBackend();
    const data = dataset([
      { id: "t1", tokens: ["a", "b", "c"] },
      { id: "t2", tokens: ["a", "b", "d"] },
    ]);
    const first = backend.train(data).toJSON();
    const second = backend.train(data).toJSON();
    expect(first).toEqual(second);
  });

  it("round-trips through a snapshot with identical predictions", () => {
    const backend = new MarkovMovementBackend();
    const model = backend.train(dataset([{ id: "t1", tokens: ["open", "type", "submit"] }]));
    const restored = backend.load(model.toJSON());
    expect(restored.generate(["open"], 10)).toEqual(model.generate(["open"], 10));
    expect(restored.rankNext(["open"])).toEqual(model.rankNext(["open"]));
    expect(restored.vocabulary).toEqual(model.vocabulary);
  });

  it("exposes a sorted vocabulary excluding the end sentinel", () => {
    const backend = new MarkovMovementBackend();
    const model = backend.train(dataset([{ id: "t1", tokens: ["type", "open", "submit"] }]));
    expect(model.vocabulary).toEqual(["open", "submit", "type"]);
  });
});

describe("dataset builders", () => {
  it("builds a movement dataset from trajectory spans, sorted by timestamp", () => {
    const span = buildTrajectorySpan({
      id: "traj-1",
      sessionId: "sess-1",
      actions: [
        { kind: "action", tool: "device", summary: "tapped submit", ts: 20, metadata: { gesture: "tap", target: "submit" } },
        { kind: "action", tool: "device", summary: "typed hello", ts: 10, metadata: { gesture: "type", target: "field" } },
      ],
    });
    const built = buildMovementDatasetFromTrajectories([span]);
    expect(built.sequences).toHaveLength(1);
    expect(built.sequences[0]!.steps.map((step) => step.token)).toEqual(["device.type:field", "device.tap:submit"]);
  });

  it("drops trajectories with no actions", () => {
    const span = buildTrajectorySpan({ id: "empty", sessionId: "s", actions: [] });
    expect(buildMovementDatasetFromTrajectories([span]).sequences).toHaveLength(0);
  });

  it("builds a movement dataset from replay manifests using action events only", () => {
    const replay: ReplayManifest = {
      version: 1,
      sessionId: "sess-1",
      trajectoryIds: ["traj-1"],
      eventCount: 3,
      events: [
        { kind: "observation", ts: 1, trajectoryId: "traj-1", source: "device", summary: "screen" },
        { kind: "action", ts: 2, trajectoryId: "traj-1", tool: "device", summary: "tapped a" },
        { kind: "action", ts: 3, trajectoryId: "traj-1", tool: "device", summary: "tapped b" },
      ],
    };
    const built = buildMovementDatasetFromReplays([replay]);
    expect(built.sequences[0]!.steps.map((step) => step.token)).toEqual(["device:tapped-a", "device:tapped-b"]);
  });
});

describe("evaluateMovementModel", () => {
  it("scores perfect accuracy when held-out sequences match training", () => {
    const backend = new MarkovMovementBackend();
    const data = dataset([{ id: "t1", tokens: ["open", "type", "submit"] }]);
    const model = backend.train(data);
    const result = evaluateMovementModel(model, data.sequences);
    expect(result.predictions).toBe(2);
    expect(result.accuracy).toBe(1);
    expect(result.coverage).toBe(1);
  });

  it("measures generalisation on a held-out related sequence", () => {
    const backend = new MarkovMovementBackend();
    const model = backend.train(
      dataset([
        { id: "t1", tokens: ["login", "type", "submit"] },
        { id: "t2", tokens: ["search", "type", "submit"] },
      ]),
    );
    // Held-out sequence shares the "type -> submit" transition.
    const result = evaluateMovementModel(model, [{ id: "held", steps: ["checkout", "type", "submit"].map((token) => ({ token })) }]);
    expect(result.predictions).toBe(2);
    // "type -> submit" is predicted correctly via backoff; the first step is a cold start.
    expect(result.correct).toBeGreaterThanOrEqual(1);
    expect(result.accuracy).toBeGreaterThan(0);
  });
});
