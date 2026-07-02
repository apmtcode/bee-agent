import { describe, expect, it } from "vitest";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import {
  buildMovementDataset,
  evaluateReplayFidelity,
  NgramMovementModelBackend,
  tokenizeAction,
  tokenizeTrajectory,
  type MovementDataset,
  type MovementSequence,
} from "./movement-model.js";

function dataset(sequences: MovementSequence[]): MovementDataset {
  return { version: 1, sequences };
}

const TAP_MENU = "A|device|tap|menu";
const TAP_FILE = "A|device|tap|file";
const TAP_SAVE = "A|device|tap|save";
const TAP_TOOLBAR = "A|device|tap|toolbar";
const TAP_NEW = "A|device|tap|new";

describe("tokenizer", () => {
  it("encodes an action's gesture and target from metadata", () => {
    const token = tokenizeAction({
      kind: "action",
      tool: "device",
      summary: "tapped Save button",
      ts: 1,
      metadata: { gesture: "tap", target: "Save Button" },
    });
    expect(token).toBe("A|device|tap|save-button");
  });

  it("orders trajectory observations and actions by timestamp", () => {
    const trajectory = buildTrajectorySpan({
      id: "traj-1",
      sessionId: "sess-1",
      observations: [{ kind: "observation", source: "os", summary: "editor focused", ts: 5 }],
      actions: [
        { kind: "action", tool: "device", summary: "tap", ts: 30, metadata: { gesture: "tap", target: "save" } },
        { kind: "action", tool: "device", summary: "tap", ts: 10, metadata: { gesture: "tap", target: "file" } },
      ],
    });
    const sequence = tokenizeTrajectory(trajectory);
    expect(sequence.tokens).toEqual(["O|os|editor-focused", TAP_FILE, TAP_SAVE]);
  });
});

describe("NgramMovementModelBackend", () => {
  it("repeats a recorded movement from an exact context", async () => {
    const backend = new NgramMovementModelBackend();
    const model = await backend.train(dataset([{ id: "s1", tokens: [TAP_MENU, TAP_FILE, TAP_SAVE] }]), { order: 3 });

    const prediction = model.predict([TAP_MENU, TAP_FILE]);
    expect(prediction.token).toBe(TAP_SAVE);
    expect(prediction.source).toBe("exact");
    expect(prediction.confidence).toBe(1);
  });

  it("generalizes to a new-but-related movement by backing off to a shorter context", async () => {
    const backend = new NgramMovementModelBackend();
    const model = await backend.train(
      dataset([
        { id: "s1", tokens: [TAP_MENU, TAP_FILE, TAP_SAVE] },
        { id: "s2", tokens: [TAP_TOOLBAR, TAP_FILE, TAP_SAVE] },
      ]),
      { order: 3 },
    );

    // The pair (TAP_NEW, TAP_FILE) was never seen, but TAP_FILE -> TAP_SAVE was.
    const prediction = model.predict([TAP_NEW, TAP_FILE]);
    expect(prediction.token).toBe(TAP_SAVE);
    expect(prediction.source).toBe("backoff");
    expect(prediction.contextLength).toBe(1);
  });

  it("autoregressively generates a learned sequence from an empty seed", async () => {
    const backend = new NgramMovementModelBackend();
    const model = await backend.train(
      dataset([
        { id: "s1", tokens: [TAP_MENU, TAP_FILE, TAP_SAVE] },
        { id: "s2", tokens: [TAP_TOOLBAR, TAP_FILE, TAP_SAVE] },
      ]),
      { order: 3 },
    );

    // Empty context -> prior picks the lexicographically-first first token (menu < toolbar).
    expect(model.generate([], 3)).toEqual([TAP_MENU, TAP_FILE, TAP_SAVE]);
  });

  it("returns no prediction when nothing was learned", async () => {
    const backend = new NgramMovementModelBackend();
    const model = await backend.train(dataset([]), { order: 3 });
    const prediction = model.predict([TAP_FILE]);
    expect(prediction.token).toBeUndefined();
    expect(prediction.source).toBe("none");
    expect(model.generate([], 5)).toEqual([]);
  });

  it("round-trips through serialize/load with identical predictions", async () => {
    const backend = new NgramMovementModelBackend();
    const sequences = [
      { id: "s1", tokens: [TAP_MENU, TAP_FILE, TAP_SAVE] },
      { id: "s2", tokens: [TAP_TOOLBAR, TAP_FILE, TAP_SAVE] },
    ];
    const model = await backend.train(dataset(sequences), { order: 3 });
    const restored = backend.load(model.serialize());

    expect(restored.order).toBe(model.order);
    expect(restored.vocabularySize).toBe(model.vocabularySize);
    expect(restored.sequenceCount).toBe(model.sequenceCount);
    expect(restored.predict([TAP_NEW, TAP_FILE])).toEqual(model.predict([TAP_NEW, TAP_FILE]));
    expect(restored.generate([], 3)).toEqual(model.generate([], 3));
  });

  it("trains deterministically (same dataset -> identical serialization)", async () => {
    const backend = new NgramMovementModelBackend();
    const sequences = [
      { id: "s1", tokens: [TAP_MENU, TAP_FILE, TAP_SAVE] },
      { id: "s2", tokens: [TAP_TOOLBAR, TAP_FILE, TAP_SAVE] },
    ];
    const a = await backend.train(dataset(sequences), { order: 3 });
    const b = await backend.train(dataset(sequences), { order: 3 });
    expect(a.serialize()).toBe(b.serialize());
  });
});

describe("evaluateReplayFidelity", () => {
  it("reproduces an unambiguous training sequence with perfect fidelity", async () => {
    const backend = new NgramMovementModelBackend();
    // A single deterministic sequence has no branching, so every step is recoverable.
    const sequences = [{ id: "s1", tokens: [TAP_MENU, TAP_FILE, TAP_SAVE] }];
    const model = await backend.train(dataset(sequences), { order: 3 });
    const report = evaluateReplayFidelity(model, sequences);
    expect(report.accuracy).toBe(1);
    expect(report.correct).toBe(report.steps);
    expect(report.perSequence).toHaveLength(1);
  });

  it("measures generalization on a held-out related sequence above chance", async () => {
    const backend = new NgramMovementModelBackend();
    const train = [
      { id: "s1", tokens: [TAP_MENU, TAP_FILE, TAP_SAVE] },
      { id: "s2", tokens: [TAP_TOOLBAR, TAP_FILE, TAP_SAVE] },
    ];
    const model = await backend.train(dataset(train), { order: 3 });
    // Held-out: novel opener, but the file->save tail should still be predicted.
    const heldOut = [{ id: "s3", tokens: [TAP_NEW, TAP_FILE, TAP_SAVE] }];
    const report = evaluateReplayFidelity(model, heldOut);
    // 2 of 3 tokens (file, save) are recoverable from backoff.
    expect(report.correct).toBeGreaterThanOrEqual(2);
  });
});

describe("buildMovementDataset", () => {
  it("builds a dataset from trajectories and drops empty sequences", () => {
    const withActions = buildTrajectorySpan({
      id: "traj-1",
      sessionId: "sess-1",
      actions: [{ kind: "action", tool: "device", summary: "tap", ts: 10, metadata: { gesture: "tap", target: "file" } }],
    });
    const empty = buildTrajectorySpan({ id: "traj-2", sessionId: "sess-1" });

    const built = buildMovementDataset([withActions, empty]);
    expect(built.sequences).toHaveLength(1);
    expect(built.sequences[0]?.id).toBe("traj-1");
    expect(built.sequences[0]?.tokens).toEqual([TAP_FILE]);
  });
});
