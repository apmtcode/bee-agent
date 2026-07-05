import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { buildReplayManifest } from "../capture/replay.js";
import { buildTrajectorySpan, type TrajectoryAction, type TrajectorySpan } from "../capture/trajectory.js";
import {
  MarkovMovementBackend,
  MOVEMENT_END_TOKEN,
  datasetFromReplayManifest,
  datasetFromTrajectories,
  evaluateNextTokenAccuracy,
  loadMovementModel,
  movementTokenFromAction,
  saveMovementModel,
  type MovementDataset,
} from "./movement-model.js";

/**
 * Synthetic movement-stream generator. Simulates a deterministic "workflow"
 * (e.g. open-app -> focus-field -> type -> submit) so we can validate the
 * capture -> dataset -> train -> replay loop with zero real OS input.
 */
function syntheticSequence(id: string, steps: string[]): { id: string; tokens: string[] } {
  return { id, tokens: steps };
}

function actionsToTrajectory(id: string, tools: Array<[string, string]>): TrajectorySpan {
  const actions: TrajectoryAction[] = tools.map(([tool, summary], index) => ({
    kind: "action",
    tool,
    summary,
    ts: index + 1,
  }));
  return buildTrajectorySpan({ id, sessionId: `session-${id}`, actions });
}

describe("MarkovMovementBackend", () => {
  const backend = new MarkovMovementBackend();

  it("reproduces a recorded movement verbatim (replay)", () => {
    const dataset: MovementDataset = {
      version: 1,
      sequences: [syntheticSequence("s1", ["open-app", "focus-field", "type-text", "submit"])],
    };
    const model = backend.train(dataset, { order: 2 });
    const rolled = backend.generate(model, ["open-app"], 10);
    expect(rolled).toEqual(["focus-field", "type-text", "submit"]);
  });

  it("terminates generation at the END sentinel instead of looping", () => {
    const dataset: MovementDataset = {
      version: 1,
      sequences: [syntheticSequence("s1", ["a", "b", "c"])],
    };
    const model = backend.train(dataset, { order: 2 });
    const rolled = backend.generate(model, ["a"], 100);
    expect(rolled).toEqual(["b", "c"]);
    // The next-token prediction after the full sequence is the END sentinel.
    expect(backend.predict(model, ["a", "b", "c"]).token).toBe(MOVEMENT_END_TOKEN);
  });

  it("generalizes to a new but related context via back-off", () => {
    // Two workflows that share the middle step "focus-field" -> "type-text".
    // A never-before-seen prefix ending in "focus-field" should still predict
    // "type-text" by backing off from the (unseen) order-2 context to order-1.
    const dataset: MovementDataset = {
      version: 1,
      sequences: [
        syntheticSequence("s1", ["open-app", "focus-field", "type-text", "submit"]),
        syntheticSequence("s2", ["open-menu", "focus-field", "type-text", "confirm"]),
      ],
    };
    const model = backend.train(dataset, { order: 2 });
    const prediction = backend.predict(model, ["never-seen-step", "focus-field"]);
    expect(prediction.token).toBe("type-text");
    expect(prediction.fallback).toBe(true);
    expect(prediction.order).toBe(1);
  });

  it("is deterministic and breaks ties reproducibly", () => {
    const dataset: MovementDataset = {
      version: 1,
      sequences: [
        syntheticSequence("s1", ["start", "beta"]),
        syntheticSequence("s2", ["start", "alpha"]),
      ],
    };
    const modelA = backend.train(dataset, { order: 1 });
    const modelB = backend.train(dataset, { order: 1 });
    expect(modelA).toEqual(modelB);
    // Equal counts (1 each) -> lexicographically smaller token wins.
    expect(backend.predict(modelA, ["start"]).token).toBe("alpha");
  });

  it("falls back to the unigram distribution for a wholly unknown context", () => {
    const dataset: MovementDataset = {
      version: 1,
      sequences: [syntheticSequence("s1", ["x", "y", "y", "z"])],
    };
    const model = backend.train(dataset, { order: 2 });
    const prediction = backend.predict(model, ["completely-unknown"]);
    expect(prediction.order).toBe(0);
    expect(prediction.token).toBe("y"); // most frequent next token overall
  });

  it("returns a null prediction from an empty model", () => {
    const model = backend.train({ version: 1, sequences: [] }, { order: 2 });
    expect(backend.predict(model, ["anything"]).token).toBeNull();
    expect(backend.generate(model, ["anything"], 5)).toEqual([]);
  });
});

describe("dataset builders", () => {
  it("derives a movement dataset from trajectory action streams (sorted by ts)", () => {
    const trajectory = actionsToTrajectory("t1", [
      ["Click", "Open App"],
      ["Type", "Hello World"],
    ]);
    const dataset = datasetFromTrajectories([trajectory]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]!.tokens).toEqual(["click:open-app", "type:hello-world"]);
  });

  it("drops trajectories with no actions", () => {
    const empty = buildTrajectorySpan({ id: "empty", sessionId: "s" });
    expect(datasetFromTrajectories([empty]).sequences).toHaveLength(0);
  });

  it("derives a single-sequence dataset from a replay manifest", () => {
    const trajectory = actionsToTrajectory("t1", [
      ["Click", "Open"],
      ["Submit", "Form"],
    ]);
    const manifest = buildReplayManifest({
      sessionId: "session-t1",
      transcript: [],
      trajectories: [trajectory],
    });
    const dataset = datasetFromReplayManifest(manifest);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]!.tokens).toEqual(["click:open", "submit:form"]);
  });

  it("normalizes tokens consistently", () => {
    expect(movementTokenFromAction("  Mouse Move ", " to (10, 20) ")).toBe("mouse-move:to-(10,-20)");
    expect(movementTokenFromAction("Key", "")).toBe("key");
  });
});

describe("evaluateNextTokenAccuracy", () => {
  const backend = new MarkovMovementBackend();

  it("scores memorized sequences perfectly", () => {
    const dataset: MovementDataset = {
      version: 1,
      sequences: [syntheticSequence("s1", ["open-app", "focus-field", "type-text", "submit"])],
    };
    const model = backend.train(dataset, { order: 3 });
    const result = evaluateNextTokenAccuracy(backend, model, dataset);
    expect(result.accuracy).toBe(1);
    expect(result.predictions).toBe(5); // 4 tokens + END
  });

  it("scores above chance on held-out related sequences", () => {
    const train: MovementDataset = {
      version: 1,
      sequences: [
        syntheticSequence("s1", ["open-app", "focus-field", "type-text", "submit"]),
        syntheticSequence("s2", ["open-app", "focus-field", "type-text", "confirm"]),
      ],
    };
    const heldOut: MovementDataset = {
      version: 1,
      sequences: [syntheticSequence("s3", ["open-app", "focus-field", "type-text", "submit"])],
    };
    const model = backend.train(train, { order: 2 });
    const result = evaluateNextTokenAccuracy(backend, model, heldOut);
    expect(result.accuracy).toBeGreaterThan(0.5);
  });
});

describe("model persistence", () => {
  const backend = new MarkovMovementBackend();
  let dir: string;

  beforeEach(async () => {
    dir = await mkdtemp(path.join(tmpdir(), "movement-model-"));
  });

  afterEach(async () => {
    await rm(dir, { recursive: true, force: true });
  });

  it("round-trips a trained model through disk", async () => {
    const dataset: MovementDataset = {
      version: 1,
      sequences: [syntheticSequence("s1", ["a", "b", "c"])],
    };
    const model = backend.train(dataset, { order: 2 });
    const file = path.join(dir, "model.json");
    await saveMovementModel(file, model);
    const loaded = await loadMovementModel(file);
    expect(loaded).toEqual(model);
    expect(backend.generate(loaded!, ["a"], 10)).toEqual(["b", "c"]);
  });

  it("returns undefined when no model exists", async () => {
    expect(await loadMovementModel(path.join(dir, "missing.json"))).toBeUndefined();
  });
});
