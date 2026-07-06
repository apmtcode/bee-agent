import { describe, expect, it } from "vitest";
import type { ReplayTimelineEvent } from "../capture/replay.js";
import {
  MarkovMovementBackend,
  MOVEMENT_END,
  MOVEMENT_START,
  defaultActionToken,
  evaluateNextTokenAccuracy,
  measureReplayFidelity,
  movementDatasetFromReplays,
  movementSequenceFromReplay,
  type MovementDataset,
} from "./movement-policy.js";

function dataset(sequences: Array<{ id: string; tokens: string[] }>): MovementDataset {
  return { version: 1, sequences };
}

describe("MarkovMovementBackend", () => {
  it("rejects a non-positive order", () => {
    expect(() => new MarkovMovementBackend({ order: 0 })).toThrow(/positive integer/);
    expect(() => new MarkovMovementBackend({ order: 1.5 })).toThrow(/positive integer/);
  });

  it("reproduces a recorded movement sequence deterministically", () => {
    const backend = new MarkovMovementBackend({ order: 3 });
    const model = backend.train(
      dataset([{ id: "s1", tokens: ["focus:app", "tap:menu", "tap:new", "type:title", "tap:save"] }]),
    );

    // Full replay from an empty prefix.
    expect(model.generate()).toEqual(["focus:app", "tap:menu", "tap:new", "type:title", "tap:save"]);
    // Fidelity is perfect on the training sequence.
    expect(
      measureReplayFidelity(model, {
        id: "s1",
        tokens: ["focus:app", "tap:menu", "tap:new", "type:title", "tap:save"],
      }),
    ).toBe(1);
  });

  it("predicts the most frequent continuation with a probability", () => {
    const backend = new MarkovMovementBackend({ order: 2 });
    const model = backend.train(
      dataset([
        { id: "a", tokens: ["open", "save"] },
        { id: "b", tokens: ["open", "save"] },
        { id: "c", tokens: ["open", "close"] },
      ]),
    );

    const prediction = model.predict(["open"]);
    expect(prediction.token).toBe("save");
    expect(prediction.probability).toBeCloseTo(2 / 3);
    expect(prediction.order).toBe(1);
    expect(prediction.backedOff).toBe(false);
    expect(prediction.candidates.map((candidate) => candidate.token)).toEqual(["save", "close"]);
  });

  it("returns an empty prediction for an untrained model", () => {
    const model = new MarkovMovementBackend().train(dataset([]));
    const prediction = model.predict(["anything"]);
    expect(prediction.token).toBeUndefined();
    expect(prediction.candidates).toEqual([]);
  });

  it("generalizes to novel-but-related sequences via backoff", () => {
    // Two apps share the same "menu -> new -> save" tail after a distinct open.
    const backend = new MarkovMovementBackend({ order: 2 });
    const model = backend.train(
      dataset([
        { id: "mail", tokens: ["open:mail", "tap:menu", "tap:new", "tap:save"] },
        { id: "notes", tokens: ["open:notes", "tap:menu", "tap:new", "tap:save"] },
      ]),
    );

    // A never-seen opener still resolves the shared movement via backoff.
    const prediction = model.predict(["open:calendar", "tap:menu"]);
    expect(prediction.token).toBe("tap:new");
    expect(prediction.backedOff).toBe(true);
    expect(prediction.order).toBeLessThan(2);
  });

  it("caps generation and terminates on the END sentinel", () => {
    const backend = new MarkovMovementBackend({ order: 1 });
    // A self-loop token would generate forever without the cap.
    const model = backend.train(dataset([{ id: "loop", tokens: ["a", "a", "a", "a"] }]));
    const generated = model.generate([], { maxSteps: 3 });
    expect(generated).toHaveLength(3);
    expect(generated).not.toContain(MOVEMENT_START);
    expect(generated).not.toContain(MOVEMENT_END);
  });

  it("round-trips through serialize/load with identical predictions", () => {
    const backend = new MarkovMovementBackend({ order: 3 });
    const trained = backend.train(
      dataset([
        { id: "a", tokens: ["open", "edit", "save"] },
        { id: "b", tokens: ["open", "edit", "close"] },
      ]),
    );
    const serialized = trained.serialize();
    const restored = backend.load(serialized);

    expect(restored.order).toBe(trained.order);
    expect(restored.vocabulary).toEqual(trained.vocabulary);
    expect(restored.predict(["open", "edit"])).toEqual(trained.predict(["open", "edit"]));
    expect(restored.generate()).toEqual(trained.generate());
    // Serialization is stable (deterministic key ordering).
    expect(restored.serialize()).toBe(serialized);
  });

  it("refuses to load a model from a different backend", () => {
    const backend = new MarkovMovementBackend();
    expect(() => backend.load(JSON.stringify({ backend: "other", order: 1, vocabulary: [], counts: [] }))).toThrow(
      /cannot load backend/,
    );
  });
});

describe("evaluateNextTokenAccuracy", () => {
  it("scores perfect accuracy on the training set", () => {
    const backend = new MarkovMovementBackend({ order: 3 });
    const training = dataset([{ id: "s", tokens: ["a", "b", "c", "d"] }]);
    const model = backend.train(training);
    const result = evaluateNextTokenAccuracy(model, training);
    expect(result.accuracy).toBe(1);
    expect(result.sequenceCount).toBe(1);
    expect(result.perSequence[0]?.accuracy).toBe(1);
  });

  it("measures partial generalization on held-out related sequences", () => {
    const backend = new MarkovMovementBackend({ order: 2 });
    const model = backend.train(
      dataset([
        { id: "t1", tokens: ["open", "menu", "new", "save"] },
        { id: "t2", tokens: ["open", "menu", "new", "save"] },
      ]),
    );
    // Held-out sequence shares the tail but has a novel opener.
    const heldOut = dataset([{ id: "h", tokens: ["launch", "menu", "new", "save"] }]);
    const result = evaluateNextTokenAccuracy(model, heldOut);
    expect(result.accuracy).toBeGreaterThan(0);
    expect(result.accuracy).toBeLessThan(1);
  });
});

describe("capture -> dataset bridge", () => {
  function action(ts: number, tool: string, summary: string): ReplayTimelineEvent {
    return { kind: "action", ts, trajectoryId: "traj", tool, summary };
  }

  it("tokenizes action events in timestamp order", () => {
    const sequence = movementSequenceFromReplay({
      sessionId: "sess",
      trajectoryIds: ["traj"],
      events: [action(30, "device", "Tapped Save"), action(10, "device", "Tapped  Menu"), action(20, "device", "typed title")],
    });
    expect(sequence.id).toBe("traj");
    expect(sequence.tokens).toEqual(["device:tapped menu", "device:typed title", "device:tapped save"]);
  });

  it("optionally interleaves observations and skips transcript events", () => {
    const events: ReplayTimelineEvent[] = [
      { kind: "transcript", ts: 5, messageId: "m1", role: "user", content: "do it" },
      { kind: "observation", ts: 10, trajectoryId: "t", source: "device", summary: "Mail active" },
      action(20, "device", "tapped compose"),
    ];
    const withObs = movementSequenceFromReplay({ events }, { includeObservations: true });
    expect(withObs.tokens).toEqual(["obs:device:mail active", "device:tapped compose"]);

    const withoutObs = movementSequenceFromReplay({ events });
    expect(withoutObs.tokens).toEqual(["device:tapped compose"]);
  });

  it("feeds a trained backend end to end from replay manifests", () => {
    const replays = [
      {
        sessionId: "s1",
        trajectoryIds: ["a"],
        events: [action(1, "device", "tapped menu"), action(2, "device", "tapped new")],
      },
      {
        sessionId: "s2",
        trajectoryIds: ["b"],
        events: [action(1, "device", "tapped menu"), action(2, "device", "tapped new")],
      },
    ];
    const built = movementDatasetFromReplays(replays);
    const model = new MarkovMovementBackend({ order: 2 }).train(built);
    const prediction = model.predict([defaultActionToken({ tool: "device", summary: "tapped menu" })]);
    expect(prediction.token).toBe("device:tapped new");
  });
});
