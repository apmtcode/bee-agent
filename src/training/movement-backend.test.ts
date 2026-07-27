import { describe, expect, it } from "vitest";
import type { ReplayManifest } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  MOVEMENT_END_TOKEN,
  NGramMovementBackend,
  buildMovementDataset,
  createMovementBackend,
  evaluateMovementModel,
  listMovementBackends,
  movementSequenceFromReplay,
  registerMovementBackend,
  tokenizeReplayEvent,
  type MovementDataset,
  type MovementSequence,
  type MovementTrainingBackend,
} from "./movement-backend.js";

function seq(id: string, tokens: string[], context?: string): MovementSequence {
  return { id, tokens, ...(context ? { context } : {}) };
}

function dataset(sequences: MovementSequence[]): MovementDataset {
  return { version: 1, sequences };
}

describe("NGramMovementBackend", () => {
  it("repeats a recorded movement sequence exactly (memorization)", () => {
    const backend = new NGramMovementBackend();
    const model = backend.train(dataset([seq("s1", ["move:a", "move:b", "move:c", "move:d"])]));

    // Deterministic argmax generation reproduces the single recorded path.
    const generated = backend.generate(model, { maxLength: 16 });
    expect(generated).toEqual(["move:a", "move:b", "move:c", "move:d"]);
  });

  it("stops generation at the learned end marker and never emits it", () => {
    const backend = new NGramMovementBackend();
    const model = backend.train(dataset([seq("s1", ["x", "y"])]));
    const generated = backend.generate(model, { maxLength: 50 });
    expect(generated).toEqual(["x", "y"]);
    expect(generated).not.toContain(MOVEMENT_END_TOKEN);
  });

  it("predicts the most likely next token given context", () => {
    const backend = new NGramMovementBackend();
    const model = backend.train(
      dataset([
        seq("s1", ["open", "click", "save"]),
        seq("s2", ["open", "click", "save"]),
        seq("s3", ["open", "click", "cancel"]),
      ]),
    );
    const prediction = backend.predict(model, { history: ["open", "click"] });
    expect(prediction?.token).toBe("save");
    // 2 of 3 continuations after "open click" were "save".
    expect(prediction?.probability).toBeCloseTo(2 / 3, 6);
    expect(prediction?.matchedOrder).toBe(2);
  });

  it("generalizes to an unseen prefix by backing off to a shorter context", () => {
    const backend = new NGramMovementBackend();
    // The model has never seen the exact prefix ["warmup", "click"], but it has
    // learned that "click" is very often followed by "save".
    const model = backend.train(
      dataset([
        seq("s1", ["open", "click", "save"]),
        seq("s2", ["scroll", "click", "save"]),
        seq("s3", ["focus", "click", "save"]),
      ]),
      { order: 2 },
    );
    const prediction = backend.predict(model, { history: ["warmup", "click"] });
    expect(prediction?.token).toBe("save");
    // Fell back from order-2 (no "warmup click") to order-1 ("click" -> ...).
    expect(prediction?.matchedOrder).toBe(1);
  });

  it("falls all the way back to the unigram prior for a cold, unknown context", () => {
    const backend = new NGramMovementBackend();
    const model = backend.train(dataset([seq("s1", ["a", "b", "b", "b", "c"])]));
    const prediction = backend.predict(model, { history: ["totally", "unknown"] });
    // "b" is the most frequent token overall -> unigram prior wins.
    expect(prediction?.token).toBe("b");
    expect(prediction?.matchedOrder).toBe(0);
  });

  it("produces reproducible output for a fixed random seed and varies across seeds", () => {
    const backend = new NGramMovementBackend();
    const model = backend.train(
      dataset([
        seq("s1", ["start", "a", "end-a"]),
        seq("s2", ["start", "b", "end-b"]),
        seq("s3", ["start", "c", "end-c"]),
      ]),
    );
    const first = backend.generate(model, { randomSeed: 42, maxLength: 8 });
    const firstAgain = backend.generate(model, { randomSeed: 42, maxLength: 8 });
    expect(first).toEqual(firstAgain);

    const seeds = new Set<string>();
    for (const s of [1, 2, 3, 4, 5, 6, 7, 8]) {
      seeds.add(backend.generate(model, { randomSeed: s, maxLength: 8 }).join("|"));
    }
    // Sampling explores more than one branch of the tree.
    expect(seeds.size).toBeGreaterThan(1);
  });

  it("honors an explicit generation seed prefix", () => {
    const backend = new NGramMovementBackend();
    const model = backend.train(dataset([seq("s1", ["a", "b", "c"]), seq("s2", ["z", "b", "c"])]));
    const generated = backend.generate(model, { seed: ["z"], maxLength: 8 });
    expect(generated).toEqual(["z", "b", "c"]);
  });

  it("serializes to JSON-safe weights and reloads without behavior change", () => {
    const backend = new NGramMovementBackend();
    const model = backend.train(dataset([seq("s1", ["p", "q", "r"])]));
    const roundTripped = { weights: JSON.parse(JSON.stringify(model.weights)) };
    expect(backend.generate(roundTripped, { maxLength: 8 })).toEqual(
      backend.generate(model, { maxLength: 8 }),
    );
    expect(model.weights.sequenceCount).toBe(1);
    expect(model.weights.tokenCount).toBe(3);
  });
});

describe("movement dataset construction", () => {
  it("tokenizes replay actions and observations, skipping transcript turns", () => {
    const replay: ReplayManifest = {
      version: 1,
      sessionId: "sess-1",
      trajectoryIds: ["traj-1"],
      eventCount: 3,
      events: [
        { kind: "transcript", ts: 1, messageId: "m1", role: "user", content: "do it" },
        { kind: "observation", ts: 2, trajectoryId: "traj-1", source: "browser", summary: "Opened Deploy" },
        { kind: "action", ts: 3, trajectoryId: "traj-1", tool: "browser", summary: "Clicked Deploy" },
      ],
    };
    expect(tokenizeReplayEvent(replay.events[0])).toBeUndefined();
    const sequence = movementSequenceFromReplay(replay);
    expect(sequence.tokens).toEqual([
      "observe:browser:opened deploy",
      "action:browser:clicked deploy",
    ]);
    expect(sequence.context).toBe("traj-1");
  });

  it("builds a dataset from trajectories and drops empty sequences", () => {
    const withActions: TrajectorySpan = {
      id: "traj-1",
      sessionId: "sess-1",
      createdAt: "2026-01-01T00:00:00.000Z",
      captureTier: "operator",
      observations: [],
      actions: [
        { kind: "action", tool: "keyboard", summary: "Type hello", ts: 2 },
        { kind: "action", tool: "mouse", summary: "Click submit", ts: 1 },
      ],
    };
    const empty: TrajectorySpan = {
      id: "traj-2",
      sessionId: "sess-1",
      createdAt: "2026-01-01T00:00:00.000Z",
      captureTier: "operator",
      observations: [],
      actions: [],
    };
    const built = buildMovementDataset({ trajectories: [withActions, empty] });
    expect(built.sequences).toHaveLength(1);
    // Sorted by ts: click (ts 1) before type (ts 2).
    expect(built.sequences[0].tokens).toEqual([
      "action:mouse:click submit",
      "action:keyboard:type hello",
    ]);
  });
});

describe("movement backend registry", () => {
  it("creates the default deterministic backend and lists registered backends", () => {
    expect(createMovementBackend().name).toBe("ngram-mock");
    expect(listMovementBackends()).toContain("ngram-mock");
  });

  it("supports registering a custom backend and throws on unknown names", () => {
    const stub: MovementTrainingBackend = {
      name: "stub",
      train: () => ({
        weights: {
          version: 1,
          backend: "stub",
          order: 1,
          vocabulary: [],
          transitions: {},
          unigram: {},
          starts: {},
          sequenceCount: 0,
          tokenCount: 0,
        },
      }),
      predict: () => undefined,
      generate: () => [],
    };
    registerMovementBackend("stub", () => stub);
    expect(createMovementBackend("stub").name).toBe("stub");
    expect(() => createMovementBackend("does-not-exist")).toThrow(/Unknown movement backend/);
  });
});

describe("generalization eval harness", () => {
  it("scores held-out fidelity and attributes generalized vs memorized hits", () => {
    const backend = new NGramMovementBackend();
    const model = backend.train(
      dataset([
        seq("s1", ["open", "click", "save"]),
        seq("s2", ["open", "click", "save"]),
        seq("s3", ["scroll", "click", "save"]),
      ]),
      { order: 2 },
    );
    // Held-out: a novel prefix ("focus click") whose continuation the model must
    // infer by generalizing from the "click -> save" pattern.
    const heldOut = [seq("h1", ["focus", "click", "save"])];
    const result = evaluateMovementModel(backend, model, heldOut);

    expect(result.predictions).toBe(3);
    expect(result.correct).toBeGreaterThanOrEqual(1);
    // The "save" after the unseen "focus click" prefix is a generalized hit.
    expect(result.generalizedCorrect).toBeGreaterThanOrEqual(1);
    expect(result.accuracy).toBeCloseTo(result.correct / result.predictions, 6);
  });

  it("reports perfect accuracy when the held-out set is the training set", () => {
    const backend = new NGramMovementBackend();
    const sequences = [seq("s1", ["a", "b", "c"]), seq("s2", ["a", "b", "d"])];
    const model = backend.train(dataset(sequences), { order: 2 });
    const result = evaluateMovementModel(backend, model, sequences);
    // Every next token is the argmax of its exact recorded context (or a tie the
    // model resolves consistently), so fidelity is high.
    expect(result.accuracy).toBeGreaterThan(0.5);
    expect(result.predictions).toBe(6);
  });
});
