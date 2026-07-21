import { describe, expect, it } from "vitest";
import type { ReplayManifest } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  MovementBackendRegistry,
  NgramMovementBackend,
  actionToken,
  defaultMovementRegistry,
  evaluateMovementModel,
  movementDatasetFromReplays,
  movementDatasetFromTrajectories,
  type MovementDataset,
  type MovementModelBackend,
  type MovementSequence,
} from "./movement-model.js";

/** Deterministic synthetic movement stream (no real OS input needed). */
function syntheticDataset(): MovementDataset {
  return {
    sequences: [
      { id: "open-doc", tokens: ["focus:: window", "click:: file", "click:: open", "type:: name", "press:: enter"] },
      { id: "save-doc", tokens: ["focus:: window", "click:: file", "click:: save", "type:: name", "press:: enter"] },
      { id: "close-doc", tokens: ["focus:: window", "click:: file", "click:: close"] },
    ],
  };
}

describe("NgramMovementBackend", () => {
  it("reproduces a recorded movement sequence exactly (objective 2c)", () => {
    const dataset: MovementDataset = {
      sequences: [{ id: "s", tokens: ["a", "b", "c", "d", "e"] }],
    };
    const model = new NgramMovementBackend().train(dataset, { order: 3 });
    // With a single training trajectory, unseeded generation replays it verbatim.
    expect(model.generate()).toEqual(["a", "b", "c", "d", "e"]);
  });

  it("predicts the next movement from a seen context", () => {
    const model = new NgramMovementBackend().train(syntheticDataset(), { order: 3 });
    const prediction = model.predictNext(["focus:: window", "click:: file"]);
    // "click:: file" is followed by open/save/close; deterministic tie-break -> smallest token.
    expect(prediction?.token).toBe("click:: close");
    expect(prediction?.contextLength).toBe(2);
    expect(prediction?.probability).toBeGreaterThan(0);
  });

  it("generalizes to an unseen-but-related prefix via back-off (objective 2d)", () => {
    const dataset: MovementDataset = {
      sequences: [
        { id: "1", tokens: ["start", "move", "grab", "drop", "end"] },
        { id: "2", tokens: ["begin", "move", "grab", "lift", "end"] },
      ],
    };
    const model = new NgramMovementBackend().train(dataset, { order: 3 });
    // "warmup move" was never recorded, but "move" -> "grab" always was.
    const prediction = model.predictNext(["warmup", "move"]);
    expect(prediction?.token).toBe("grab");
    // Back-off used a shorter context than the full order.
    expect(prediction?.contextLength).toBeLessThanOrEqual(2);
  });

  it("stops generation at end-of-sequence rather than looping forever", () => {
    const dataset: MovementDataset = { sequences: [{ id: "s", tokens: ["x", "y"] }] };
    const model = new NgramMovementBackend().train(dataset, { order: 2 });
    const generated = model.generate([], 100);
    expect(generated).toEqual(["x", "y"]);
  });

  it("is deterministic across repeated training and inference", () => {
    const backend = new NgramMovementBackend();
    const a = backend.train(syntheticDataset(), { order: 3 });
    const b = backend.train(syntheticDataset(), { order: 3 });
    expect(a.generate()).toEqual(b.generate());
    expect(a.serialize()).toEqual(b.serialize());
  });

  it("serializes and restores to an identical model", () => {
    const backend = new NgramMovementBackend();
    const model = backend.train(syntheticDataset(), { order: 3 });
    const snapshot = model.serialize();
    // Snapshot must survive a JSON round-trip (persistence).
    const restored = backend.restore(JSON.parse(JSON.stringify(snapshot)));
    expect(restored.serialize()).toEqual(snapshot);
    expect(restored.generate(["focus:: window", "click:: file"])).toEqual(
      model.generate(["focus:: window", "click:: file"]),
    );
  });

  it("assigns higher log-probability to seen than to unseen movements", () => {
    const model = new NgramMovementBackend().train(syntheticDataset(), { order: 3 });
    const seen = model.scoreSequence(["focus:: window", "click:: file", "click:: open"]);
    const unseen = model.scoreSequence(["nonsense:: a", "nonsense:: b", "nonsense:: c"]);
    expect(seen).toBeGreaterThan(unseen);
  });
});

describe("MovementBackendRegistry", () => {
  it("registers the deterministic n-gram backend by default", () => {
    const registry = defaultMovementRegistry();
    expect(registry.list()).toContain("ngram");
  });

  it("supports pluggable custom backends", () => {
    const custom: MovementModelBackend = {
      name: "always-halt",
      train: () => ({
        backend: "always-halt",
        order: 1,
        vocabulary: [],
        predictNext: () => undefined,
        generate: () => [],
        scoreSequence: () => 0,
        serialize: () => ({ version: 1, backend: "always-halt", order: 1, grams: [], vocabulary: [], sequenceCount: 0 }),
      }),
      restore: () => {
        throw new Error("not implemented");
      },
    };
    const registry = new MovementBackendRegistry([new NgramMovementBackend(), custom]);
    expect(registry.list()).toEqual(["always-halt", "ngram"]);
    expect(registry.train("always-halt", syntheticDataset()).generate()).toEqual([]);
  });

  it("throws a helpful error for an unknown backend", () => {
    const registry = defaultMovementRegistry();
    expect(() => registry.get("nope")).toThrow(/unknown movement backend: nope/);
  });

  it("restores a model by dispatching on the snapshot backend name", () => {
    const registry = defaultMovementRegistry();
    const snapshot = registry.train("ngram", syntheticDataset()).serialize();
    const restored = registry.restore(snapshot);
    expect(restored.backend).toBe("ngram");
    expect(restored.serialize()).toEqual(snapshot);
  });
});

describe("dataset builders", () => {
  it("builds a dataset from replay manifests, keeping action order", () => {
    const replay: ReplayManifest = {
      version: 1,
      sessionId: "sess-1",
      trajectoryIds: ["traj-1"],
      eventCount: 4,
      events: [
        { kind: "observation", ts: 1, trajectoryId: "traj-1", source: "screen", summary: "menu" },
        { kind: "action", ts: 2, trajectoryId: "traj-1", tool: "click", summary: "file" },
        { kind: "action", ts: 4, trajectoryId: "traj-1", tool: "press", summary: "enter" },
        { kind: "action", ts: 3, trajectoryId: "traj-1", tool: "type", summary: "name" },
      ],
    };
    const dataset = movementDatasetFromReplays([replay]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]!.tokens).toEqual([
      actionToken("click", "file"),
      actionToken("type", "name"),
      actionToken("press", "enter"),
    ]);
  });

  it("drops replays that contain no actions", () => {
    const replay: ReplayManifest = {
      version: 1,
      sessionId: "sess-empty",
      trajectoryIds: ["traj-empty"],
      eventCount: 1,
      events: [{ kind: "observation", ts: 1, trajectoryId: "traj-empty", source: "screen", summary: "idle" }],
    };
    expect(movementDatasetFromReplays([replay]).sequences).toHaveLength(0);
  });

  it("builds a dataset from trajectory spans in timestamp order", () => {
    const trajectory: TrajectorySpan = {
      id: "traj-2",
      sessionId: "sess-2",
      createdAt: "2026-07-21T00:00:00.000Z",
      captureTier: "app",
      observations: [],
      actions: [
        { kind: "action", tool: "press", summary: "enter", ts: 30 },
        { kind: "action", tool: "click", summary: "file", ts: 10 },
        { kind: "action", tool: "type", summary: "name", ts: 20 },
      ],
    };
    const dataset = movementDatasetFromTrajectories([trajectory]);
    expect(dataset.sequences[0]!.tokens).toEqual([
      actionToken("click", "file"),
      actionToken("type", "name"),
      actionToken("press", "enter"),
    ]);
  });
});

describe("evaluateMovementModel", () => {
  it("scores perfect fidelity on a single memorized sequence", () => {
    // One unambiguous trajectory => the eval harness reports exact recall.
    const dataset: MovementDataset = {
      sequences: [{ id: "a", tokens: ["a1", "a2", "a3", "a4"] }],
    };
    const model = new NgramMovementBackend().train(dataset, { order: 3 });
    const result = evaluateMovementModel(model, dataset.sequences);
    expect(result.nextTokenAccuracy).toBe(1);
    expect(result.predictions).toBe(4);
    expect(result.meanLogProbability).toBeLessThanOrEqual(0);
  });

  it("cannot perfectly satisfy ambiguous shared prefixes (honest metric)", () => {
    const dataset = syntheticDataset();
    const model = new NgramMovementBackend().train(dataset, { order: 3 });
    const result = evaluateMovementModel(model, dataset.sequences);
    // Three continuations after "click:: file" => argmax must miss some.
    expect(result.nextTokenAccuracy).toBeLessThan(1);
    expect(result.nextTokenAccuracy).toBeGreaterThan(0.5);
  });

  it("reports partial accuracy on related-but-novel held-out sequences", () => {
    const train: MovementDataset = {
      sequences: [
        { id: "1", tokens: ["home", "search", "click", "result", "read"] },
        { id: "2", tokens: ["home", "search", "click", "result", "share"] },
      ],
    };
    const model = new NgramMovementBackend().train(train, { order: 3 });
    const heldOut: MovementSequence[] = [{ id: "3", tokens: ["home", "search", "click", "result", "read"] }];
    const result = evaluateMovementModel(model, heldOut);
    expect(result.nextTokenAccuracy).toBeGreaterThan(0);
    expect(result.nextTokenAccuracy).toBeLessThanOrEqual(1);
  });
});
