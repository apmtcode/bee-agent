import { describe, expect, it } from "vitest";
import type { ReplayManifest } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  NGramMovementBackend,
  buildMovementDataset,
  buildMovementDatasetFromTrajectories,
  evaluateMovementModel,
  movementTokenFromAction,
  type MovementDataset,
  type MovementSequence,
} from "./movement-model.js";

function seq(id: string, ...tokens: string[]): MovementSequence {
  return { id, tokens };
}

function datasetOf(...sequences: MovementSequence[]): MovementDataset {
  return { version: 1, sequences };
}

describe("NGramMovementBackend", () => {
  it("repeats a recorded movement sequence exactly via greedy generation", () => {
    const backend = new NGramMovementBackend();
    const model = backend.train(datasetOf(seq("s1", "open", "click", "type", "submit")));

    expect(model.generate(["open"])).toEqual(["click", "type", "submit"]);
    // Full roll-out from the START boundary reproduces the whole sequence.
    expect(model.generate()).toEqual(["open", "click", "type", "submit"]);
  });

  it("is deterministic: identical training yields identical predictions", () => {
    const backend = new NGramMovementBackend();
    const dataset = datasetOf(seq("a", "x", "y", "z"), seq("b", "x", "y", "w"));
    const first = backend.train(dataset).predictNext(["x", "y"]);
    const second = backend.train(dataset).predictNext(["x", "y"]);
    expect(first).toEqual(second);
  });

  it("ranks candidates by observed frequency with a stable tie-break", () => {
    const backend = new NGramMovementBackend();
    // After "x": "y" appears twice, "z" once -> y outranks z.
    const model = backend.train(datasetOf(seq("a", "x", "y"), seq("b", "x", "y"), seq("c", "x", "z")), {
      order: 2,
    });
    const ranked = model.predictNext(["x"]);
    expect(ranked[0]?.token).toBe("y");
    expect(ranked[0]?.probability).toBeCloseTo(2 / 3);
    expect(ranked.map((prediction) => prediction.token)).toEqual(["y", "z"]);
  });

  it("generalizes to an unseen prefix by backing off to a shared suffix", () => {
    const backend = new NGramMovementBackend();
    // Train only "login -> password -> submit". A novel prefix ending in the same
    // "password" context should still predict "submit" via stupid-backoff.
    const model = backend.train(datasetOf(seq("s", "login", "password", "submit")), { order: 3 });

    const prediction = model.predictNext(["totallyNew", "password"]);
    expect(prediction[0]?.token).toBe("submit");
    // The full 2-token context was unseen, so it matched a shorter one.
    expect(prediction[0]?.matchedContext).toBeLessThan(2);
  });

  it("stops generation at the END boundary and respects maxSteps", () => {
    const backend = new NGramMovementBackend();
    // A self-loop that would never terminate without a step cap.
    const model = backend.train(datasetOf(seq("loop", "a", "a", "a", "a")), { order: 2 });
    const generated = model.generate(["a"], { maxSteps: 3 });
    expect(generated.length).toBeLessThanOrEqual(3);
  });

  it("round-trips through serialize / fromSerialized", () => {
    const backend = new NGramMovementBackend();
    const model = backend.train(datasetOf(seq("s1", "open", "click", "submit")));
    const restored = NGramMovementBackend.fromSerialized(model.serialize());

    expect(restored.order).toBe(model.order);
    expect(restored.generate(["open"])).toEqual(model.generate(["open"]));
    expect(restored.serialize()).toEqual(model.serialize());
  });
});

describe("evaluateMovementModel", () => {
  it("reports perfect fidelity on the training sequence", () => {
    const backend = new NGramMovementBackend();
    const training = seq("s1", "open", "click", "type", "submit");
    const model = backend.train(datasetOf(training));

    const evaluation = evaluateMovementModel(model, [training]);
    expect(evaluation.accuracy).toBe(1);
    expect(evaluation.exactSequenceMatches).toBe(1);
    expect(evaluation.correct).toBe(evaluation.predictions);
  });

  it("scores partial credit on a held-out but related sequence", () => {
    const backend = new NGramMovementBackend();
    const model = backend.train(
      datasetOf(seq("a", "open", "click", "submit"), seq("b", "open", "click", "confirm")),
      { order: 2 },
    );

    // Held-out shares the "open click" prefix; the model should predict several
    // steps correctly even though this exact sequence was never seen.
    const evaluation = evaluateMovementModel(model, [seq("held", "open", "click", "submit")]);
    expect(evaluation.accuracy).toBeGreaterThan(0);
    expect(evaluation.predictions).toBeGreaterThan(0);
  });
});

describe("dataset extraction", () => {
  it("builds a movement dataset from replay action events, ignoring observations", () => {
    const replay: ReplayManifest = {
      version: 1,
      sessionId: "sess-1",
      trajectoryIds: ["t1"],
      eventCount: 3,
      events: [
        { kind: "observation", ts: 1, trajectoryId: "t1", source: "device", summary: "screen" },
        { kind: "action", ts: 2, trajectoryId: "t1", tool: "device", summary: "tapped submit" },
        { kind: "action", ts: 3, trajectoryId: "t1", tool: "device", summary: "typed name" },
      ],
    };

    const dataset = buildMovementDataset([replay]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]?.tokens).toEqual(["device:tapped submit", "device:typed name"]);
  });

  it("builds a movement dataset from trajectory spans sorted by timestamp", () => {
    const trajectory = {
      id: "t1",
      actions: [
        { kind: "action", tool: "device", summary: "second", ts: 20 },
        { kind: "action", tool: "device", summary: "first", ts: 10 },
      ],
    } as unknown as TrajectorySpan;

    const dataset = buildMovementDatasetFromTrajectories([trajectory]);
    expect(dataset.sequences[0]?.tokens).toEqual(["device:first", "device:second"]);
  });

  it("normalizes whitespace when tokenizing actions", () => {
    expect(movementTokenFromAction({ tool: " device ", summary: "tapped   submit" })).toBe(
      "device:tapped submit",
    );
  });
});
