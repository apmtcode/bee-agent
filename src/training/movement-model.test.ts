import { describe, expect, it } from "vitest";
import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";
import {
  MarkovMovementBackend,
  evaluateNextTokenAccuracy,
  generateSyntheticMovementSequences,
  movementSequencesFromTrajectories,
  movementTokenFromAction,
  type MovementSequence,
} from "./movement-model.js";

function action(partial: Partial<TrajectoryAction>): TrajectoryAction {
  return {
    kind: "action",
    tool: "device",
    summary: "did a thing",
    ts: 0,
    ...partial,
  };
}

describe("movementTokenFromAction", () => {
  it("prefers structured gesture/direction metadata", () => {
    const token = movementTokenFromAction(
      action({ tool: "device", metadata: { gesture: "scroll", direction: "down" } }),
    );
    expect(token).toBe("device/scroll:down");
  });

  it("uses gesture + target when no direction is present", () => {
    const token = movementTokenFromAction(
      action({ tool: "device", metadata: { gesture: "tap", target: "Save Button" } }),
    );
    expect(token).toBe("device/tap:save-button");
  });

  it("falls back to a slugified summary when metadata is absent", () => {
    const token = movementTokenFromAction(action({ tool: "browser", summary: "Clicked Submit" }));
    expect(token).toBe("browser/clicked-submit");
  });
});

describe("movementSequencesFromTrajectories", () => {
  it("orders actions by timestamp and drops empty trajectories", () => {
    const trajectories: TrajectorySpan[] = [
      {
        id: "t1",
        sessionId: "s1",
        createdAt: "2026-07-01T00:00:00.000Z",
        captureTier: "full",
        observations: [],
        actions: [
          action({ ts: 20, metadata: { gesture: "tap", target: "b" } }),
          action({ ts: 10, metadata: { gesture: "tap", target: "a" } }),
        ],
      },
      {
        id: "t2",
        sessionId: "s1",
        createdAt: "2026-07-01T00:00:00.000Z",
        captureTier: "full",
        observations: [],
        actions: [],
      },
    ];
    const sequences = movementSequencesFromTrajectories(trajectories);
    expect(sequences).toHaveLength(1);
    expect(sequences[0]).toEqual({ id: "t1", tokens: ["device/tap:a", "device/tap:b"] });
  });
});

describe("MarkovMovementBackend", () => {
  const backend = new MarkovMovementBackend();

  it("repeats a recorded movement exactly (highest-order match)", () => {
    const sequences: MovementSequence[] = [
      { id: "s", tokens: ["a", "b", "c", "d"] },
    ];
    const model = backend.train(sequences, { maxOrder: 3 });

    const prediction = model.predictNext(["a", "b", "c"]);
    expect(prediction?.token).toBe("d");
    expect(prediction?.backedOff).toBe(false);
    expect(prediction?.orderUsed).toBe(3);

    // Replaying the recorded length from a seed reproduces the recorded tail.
    expect(model.generate(["a"], { maxSteps: 3 })).toEqual(["b", "c", "d"]);
  });

  it("generalizes to an unseen context via backoff", () => {
    // "open -> save" is the dominant bigram; an unseen prefix should still
    // predict "save" after "open" by backing off to the bigram context.
    const sequences: MovementSequence[] = [
      { id: "1", tokens: ["launch", "open", "save"] },
      { id: "2", tokens: ["focus", "open", "save"] },
    ];
    const model = backend.train(sequences, { maxOrder: 3 });

    const prediction = model.predictNext(["never", "seen", "open"]);
    expect(prediction?.token).toBe("save");
    expect(prediction?.backedOff).toBe(true);
    expect(prediction?.orderUsed).toBe(1);
  });

  it("returns a deterministic ranked distribution with tie-break by token", () => {
    const model = backend.train([{ id: "1", tokens: ["x", "a", "x", "b"] }], { maxOrder: 1 });
    const prediction = model.predictNext(["x"]);
    // "a" and "b" each follow "x" once → equal probability, tie-broken a < b.
    expect(prediction?.candidates.map((c) => c.token)).toEqual(["a", "b"]);
    expect(prediction?.token).toBe("a");
    expect(prediction?.candidates[0]?.probability).toBeCloseTo(0.5);
  });

  it("falls back to the unigram prior for a wholly unknown context", () => {
    const model = backend.train([{ id: "1", tokens: ["a", "a", "a", "b"] }], { maxOrder: 2 });
    const prediction = model.predictNext(["totally-unknown"]);
    expect(prediction?.orderUsed).toBe(0);
    expect(prediction?.token).toBe("a"); // most frequent unigram
  });

  it("returns undefined when trained on nothing", () => {
    const model = backend.train([], { maxOrder: 2 });
    expect(model.predictNext(["a"])).toBeUndefined();
    expect(model.generate(["a"])).toEqual([]);
  });

  it("stops generating under the repetition guard", () => {
    const model = backend.train([{ id: "1", tokens: ["loop", "loop", "loop", "loop"] }], {
      maxOrder: 1,
    });
    const generated = model.generate(["loop"], { maxSteps: 100, repetitionGuard: 3 });
    expect(generated.length).toBeLessThanOrEqual(3);
    expect(generated.every((token) => token === "loop")).toBe(true);
  });

  it("round-trips through a snapshot", () => {
    const model = backend.train([{ id: "1", tokens: ["a", "b", "c"] }], { maxOrder: 2 });
    const snapshot = model.snapshot();
    const restored = backend.load(JSON.parse(JSON.stringify(snapshot)));
    expect(restored.predictNext(["a", "b"])?.token).toBe("c");
    expect(restored.maxOrder).toBe(2);
    expect(restored.backend).toBe("markov-ngram");
  });

  it("clamps out-of-range maxOrder deterministically", () => {
    const model = backend.train([{ id: "1", tokens: ["a", "b"] }], { maxOrder: 999 });
    expect(model.maxOrder).toBe(8);
    const zero = backend.train([{ id: "1", tokens: ["a", "b"] }], { maxOrder: -5 });
    expect(zero.maxOrder).toBe(0);
  });
});

describe("synthetic generation + generalization eval", () => {
  it("is fully deterministic for a given seed", () => {
    const vocab = ["m/1", "m/2", "m/3", "m/4"];
    const a = generateSyntheticMovementSequences({ seed: 42, sequenceCount: 5, vocabulary: vocab });
    const b = generateSyntheticMovementSequences({ seed: 42, sequenceCount: 5, vocabulary: vocab });
    expect(a).toEqual(b);
    const c = generateSyntheticMovementSequences({ seed: 43, sequenceCount: 5, vocabulary: vocab });
    expect(c).not.toEqual(a);
  });

  it("throws on a degenerate vocabulary", () => {
    expect(() =>
      generateSyntheticMovementSequences({ seed: 1, sequenceCount: 1, vocabulary: ["only"] }),
    ).toThrow(/at least 2/);
  });

  it("learns the synthetic motif and predicts held-out sequences well", () => {
    const vocab = ["m/1", "m/2", "m/3", "m/4", "m/5", "m/6"];
    const train = generateSyntheticMovementSequences({
      seed: 7,
      sequenceCount: 200,
      vocabulary: vocab,
      motifStrength: 0.9,
      minLength: 6,
      maxLength: 10,
    });
    const test = generateSyntheticMovementSequences({
      seed: 99,
      sequenceCount: 40,
      vocabulary: vocab,
      motifStrength: 0.9,
      minLength: 6,
      maxLength: 10,
    });
    const model = new MarkovMovementBackend().train(train, { maxOrder: 2 });
    const result = evaluateNextTokenAccuracy(model, test);

    expect(result.predictions).toBeGreaterThan(0);
    // The motif (token i -> token i+1) holds 90% of the time, so a model that
    // learned it should beat random (1/6 ≈ 0.17) by a wide margin.
    expect(result.accuracy).toBeGreaterThan(0.7);
  });

  it("reports zero accuracy for an empty eval set without dividing by zero", () => {
    const model = new MarkovMovementBackend().train([{ id: "1", tokens: ["a", "b"] }]);
    const result = evaluateNextTokenAccuracy(model, [{ id: "e", tokens: [] }]);
    expect(result.predictions).toBe(0);
    expect(result.accuracy).toBe(0);
    expect(result.generalizedShare).toBe(0);
  });
});
