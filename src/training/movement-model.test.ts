import { describe, expect, it } from "vitest";
import { buildReplayManifest } from "../capture/replay.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import {
  BackoffMarkovMovementBackend,
  FrequencyMovementBackend,
  buildMovementDatasetFromReplays,
  buildMovementDatasetFromTrajectories,
  defaultMovementBackend,
  encodeMovementToken,
  evaluateReplayFidelity,
  generateSyntheticMovementSequences,
  movementSimilarity,
  parseMovementToken,
  type MovementDataset,
} from "./movement-model.js";

const X = encodeMovementToken({ tool: "device", gesture: "scroll", direction: "down" });
const X_RELATED = encodeMovementToken({ tool: "device", gesture: "scroll", direction: "up" });
const Y = encodeMovementToken({ tool: "device", gesture: "tap", target: "Item" });

describe("movement token encoding", () => {
  it("round-trips through parse", () => {
    const action = { tool: "device", gesture: "tap", target: "Send", direction: "up" as const };
    const token = encodeMovementToken(action);
    expect(parseMovementToken(token)).toEqual(action);
  });

  it("emits fields in a stable canonical order regardless of input order", () => {
    const a = encodeMovementToken({ target: "Send", tool: "device", gesture: "tap" });
    const b = encodeMovementToken({ gesture: "tap", tool: "device", target: "Send" });
    expect(a).toBe(b);
  });

  it("omits absent optional fields", () => {
    expect(encodeMovementToken({ tool: "device" })).toBe("tool=device");
  });
});

describe("dataset construction", () => {
  it("builds ordered sequences from trajectory actions", () => {
    const span = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [
        { kind: "action", tool: "device", summary: "b", ts: 20, metadata: { gesture: "tap", target: "Item" } },
        { kind: "action", tool: "device", summary: "a", ts: 10, metadata: { gesture: "scroll", direction: "down" } },
      ],
    });
    const dataset = buildMovementDatasetFromTrajectories([span]);
    expect(dataset.sequences).toEqual([[X, Y]]);
  });

  it("drops action-less trajectories", () => {
    const span = buildTrajectorySpan({ id: "t2", sessionId: "s1" });
    expect(buildMovementDatasetFromTrajectories([span]).sequences).toEqual([]);
  });

  it("builds per-trajectory sequences from replay manifests", () => {
    const span = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [
        { kind: "action", tool: "device", summary: "scroll down", ts: 10 },
        { kind: "action", tool: "device", summary: "tap Item", ts: 20 },
      ],
    });
    const manifest = buildReplayManifest({ sessionId: "s1", transcript: [], trajectories: [span] });
    const dataset = buildMovementDatasetFromReplays([manifest]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]).toHaveLength(2);
  });
});

describe("backoff markov backend — repeat recorded movements", () => {
  const dataset: MovementDataset = { version: 1, sequences: [[X, Y], [X, Y], [X, Y]] };

  it("reproduces the recorded sequence with full fidelity", () => {
    const backend = defaultMovementBackend();
    const session = backend.load(backend.train(dataset));
    const report = evaluateReplayFidelity(session, dataset.sequences);
    expect(report.fidelity).toBe(1);
    expect(report.matched).toBe(report.transitions);
  });

  it("greedily generates the learned sequence from an empty seed", () => {
    const backend = new BackoffMarkovMovementBackend();
    const session = backend.load(backend.train(dataset));
    expect(session.generate([], 2)).toEqual([X, Y]);
  });

  it("predicts the recorded next movement with its empirical probability", () => {
    const backend = new BackoffMarkovMovementBackend();
    const session = backend.load(backend.train(dataset));
    const prediction = session.predictNext([X]);
    expect(prediction?.token).toBe(Y);
    expect(prediction?.probability).toBe(1);
    expect(prediction?.generalized).toBe(false);
  });
});

describe("backoff markov backend — generalize to related movements", () => {
  const dataset: MovementDataset = { version: 1, sequences: [[X, Y], [X, Y]] };

  it("maps an unseen-but-similar context to a known movement", () => {
    // X_RELATED (scroll up) was never recorded; only X (scroll down) was.
    expect(movementSimilarity(parseMovementToken(X_RELATED), parseMovementToken(X))).toBeGreaterThan(0);
    const backend = defaultMovementBackend();
    const session = backend.load(backend.train(dataset));
    const prediction = session.predictNext([X_RELATED]);
    expect(prediction?.token).toBe(Y);
    expect(prediction?.generalized).toBe(true);
  });

  it("scores generalized matches on held-out related sequences", () => {
    const backend = defaultMovementBackend();
    const session = backend.load(backend.train(dataset));
    const heldOut = [[X_RELATED, Y]];
    const report = evaluateReplayFidelity(session, heldOut);
    expect(report.generalizedMatches).toBeGreaterThanOrEqual(1);
  });
});

describe("pluggable backends", () => {
  it("frequency baseline satisfies the backend interface and predicts the mode", () => {
    const dataset: MovementDataset = { version: 1, sequences: [[Y, Y, X]] };
    const backend = new FrequencyMovementBackend();
    const model = backend.train(dataset);
    expect(model.backend).toBe("frequency-baseline");
    const session = backend.load(model);
    // Y occurs twice, X once -> always predicts Y regardless of context.
    expect(session.predictNext([X])?.token).toBe(Y);
    expect(session.predictNext([])?.token).toBe(Y);
  });

  it("serialized models are plain JSON (persistable)", () => {
    const backend = defaultMovementBackend();
    const model = backend.train({ version: 1, sequences: [[X, Y]] });
    expect(JSON.parse(JSON.stringify(model))).toEqual(model);
  });
});

describe("synthetic movement stream generator", () => {
  const spec = {
    seed: 42,
    sequenceCount: 5,
    templates: [
      { tool: "device", gesture: "scroll", direction: "down" as const },
      { tool: "device", gesture: "tap" },
    ],
    targets: ["Inbox", "Compose", "Archive"],
  };

  it("is deterministic for a given seed", () => {
    expect(generateSyntheticMovementSequences(spec)).toEqual(generateSyntheticMovementSequences(spec));
  });

  it("varies with the seed", () => {
    const a = generateSyntheticMovementSequences(spec);
    const b = generateSyntheticMovementSequences({ ...spec, seed: 7 });
    expect(a).not.toEqual(b);
  });

  it("round-trips capture -> dataset -> train -> replay with high fidelity", () => {
    const sequences = generateSyntheticMovementSequences({ ...spec, sequenceCount: 20 });
    const backend = defaultMovementBackend();
    const session = backend.load(backend.train({ version: 1, sequences }, { order: 2 }));
    const report = evaluateReplayFidelity(session, sequences);
    // The first tap target is unpredictable (jittered), but the sequence
    // structure is fully learnable, so the model recovers most transitions.
    expect(report.fidelity).toBeGreaterThan(0.5);
  });
});
