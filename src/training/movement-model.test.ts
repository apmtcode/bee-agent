import { describe, expect, it } from "vitest";
import {
  NgramMovementBackend,
  MOVEMENT_END_TOKEN,
  buildMovementDatasetFromReplays,
  buildMovementDatasetFromTrajectories,
  decodeMovementToken,
  encodeMovementToken,
  evaluateNextStepPrediction,
  movementStepFromAction,
  type MovementDataset,
} from "./movement-model.js";
import {
  generateSyntheticMovementDataset,
  splitMovementDataset,
} from "./synthetic-movement.js";
import type { TrajectorySpan } from "../capture/trajectory.js";
import type { ReplayManifest } from "../capture/replay.js";

describe("movement token codec", () => {
  it("round-trips steps with and without qualifiers", () => {
    const withQualifier = { channel: "device", verb: "swipe", qualifier: "down" };
    const withoutQualifier = { channel: "os", verb: "focus-changed" };
    expect(decodeMovementToken(encodeMovementToken(withQualifier))).toEqual(withQualifier);
    expect(decodeMovementToken(encodeMovementToken(withoutQualifier))).toEqual(withoutQualifier);
  });

  it("decodes boundary tokens to undefined", () => {
    expect(decodeMovementToken(MOVEMENT_END_TOKEN)).toBeUndefined();
  });
});

describe("NgramMovementBackend", () => {
  const backend = new NgramMovementBackend();

  const dataset: MovementDataset = {
    version: 1,
    sequences: [
      {
        id: "a",
        steps: [
          { channel: "os", verb: "focus-changed", qualifier: "mail" },
          { channel: "device", verb: "tap", qualifier: "compose" },
          { channel: "device", verb: "type", qualifier: "body" },
          { channel: "device", verb: "tap", qualifier: "send" },
        ],
      },
    ],
  };

  it("predicts the next step from a learned prefix", async () => {
    const model = await backend.train(dataset, { order: 3 });
    const prediction = model.predictNext([
      encodeMovementToken({ channel: "os", verb: "focus-changed", qualifier: "mail" }),
    ]);
    expect(prediction.step).toEqual({ channel: "device", verb: "tap", qualifier: "compose" });
    expect(prediction.probability).toBeGreaterThan(0);
  });

  it("greedily regenerates the dominant recorded movement sequence", async () => {
    const model = await backend.train(dataset, { order: 3 });
    const generated = model.generate({ maxSteps: 16 });
    expect(generated).toEqual(dataset.sequences[0]!.steps);
  });

  it("terminates generation at end-of-sequence rather than looping forever", async () => {
    const model = await backend.train(dataset, { order: 3 });
    const generated = model.generate({ maxSteps: 4 });
    expect(generated.length).toBeLessThanOrEqual(4);
  });

  it("produces deterministic output for a fixed random seed", async () => {
    const model = await backend.train(
      generateSyntheticMovementDataset({ sequenceCount: 40, seed: 7 }),
      { order: 2 },
    );
    const first = model.generate({ randomSeed: 123, maxSteps: 12 });
    const second = model.generate({ randomSeed: 123, maxSteps: 12 });
    expect(first).toEqual(second);
    expect(first.length).toBeGreaterThan(0);
  });

  it("survives a snapshot/restore round-trip with identical predictions", async () => {
    const model = await backend.train(dataset, { order: 3 });
    const restored = backend.restore(model.snapshot());
    const context = [encodeMovementToken({ channel: "device", verb: "tap", qualifier: "compose" })];
    expect(restored.predictNext(context)).toEqual(model.predictNext(context));
  });

  it("backs off to shorter contexts for unseen prefixes", async () => {
    const model = await backend.train(dataset, { order: 3 });
    // A context the trigram never saw; should still predict via lower-order backoff.
    const prediction = model.predictNext([
      encodeMovementToken({ channel: "browser", verb: "navigate", qualifier: "docs" }),
      encodeMovementToken({ channel: "device", verb: "tap", qualifier: "compose" }),
    ]);
    expect(prediction.order).toBeLessThan(3);
    expect(prediction.token).not.toBe(MOVEMENT_END_TOKEN);
  });
});

describe("dataset builders", () => {
  it("tokenizes captured actions, ordering by timestamp", () => {
    const trajectory: TrajectorySpan = {
      id: "t1",
      sessionId: "s1",
      createdAt: "2026-01-01T00:00:00.000Z",
      captureTier: "app",
      observations: [],
      actions: [
        { kind: "action", tool: "device", summary: "tapped send", ts: 20, metadata: { gesture: "tap", target: "Send Button" } },
        { kind: "action", tool: "device", summary: "typed body", ts: 10, metadata: { gesture: "type", target: "body" } },
      ],
    };
    const dataset = buildMovementDatasetFromTrajectories([trajectory]);
    expect(dataset.sequences[0]!.steps).toEqual([
      { channel: "device", verb: "type", qualifier: "body" },
      { channel: "device", verb: "tap", qualifier: "send-button" },
    ]);
  });

  it("skips trajectories with no actions", () => {
    const trajectory: TrajectorySpan = {
      id: "empty",
      sessionId: "s1",
      createdAt: "2026-01-01T00:00:00.000Z",
      captureTier: "app",
      observations: [],
      actions: [],
    };
    expect(buildMovementDatasetFromTrajectories([trajectory]).sequences).toHaveLength(0);
  });

  it("builds a dataset from replay manifests", () => {
    const replay: ReplayManifest = {
      version: 1,
      sessionId: "s1",
      trajectoryIds: ["t1"],
      eventCount: 2,
      events: [
        { kind: "observation", ts: 1, trajectoryId: "t1", source: "os", summary: "focused mail" },
        { kind: "action", ts: 2, trajectoryId: "t1", tool: "device", summary: "tapped compose" },
      ],
    };
    const dataset = buildMovementDatasetFromReplays([replay]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]!.steps).toEqual([{ channel: "device", verb: "tapped" }]);
  });

  it("derives a verb from the summary when metadata is absent", () => {
    const step = movementStepFromAction({ kind: "action", tool: "os", summary: "opened settings", ts: 0 });
    expect(step).toEqual({ channel: "os", verb: "opened" });
  });
});

describe("generalization eval harness", () => {
  it("achieves high next-step accuracy on held-out but related sequences", async () => {
    const backend = new NgramMovementBackend();
    const dataset = generateSyntheticMovementDataset({ sequenceCount: 120, seed: 42 });
    const { train, heldOut } = splitMovementDataset(dataset, 0.25);
    expect(heldOut.sequences.length).toBeGreaterThan(0);
    expect(train.sequences.length).toBeGreaterThan(0);

    const model = await backend.train(train, { order: 3 });
    const result = evaluateNextStepPrediction(model, heldOut, { order: 3 });

    // The synthetic flows share stable skeletons, so a trigram model must
    // generalize well to unseen-but-related sequences.
    expect(result.predictionCount).toBeGreaterThan(0);
    expect(result.accuracy).toBeGreaterThan(0.6);
  });

  it("reports zeros for an empty held-out set", async () => {
    const backend = new NgramMovementBackend();
    const model = await backend.train({ version: 1, sequences: [] });
    const result = evaluateNextStepPrediction(model, { version: 1, sequences: [] });
    expect(result).toMatchObject({ predictionCount: 0, correct: 0, accuracy: 0 });
  });
});
