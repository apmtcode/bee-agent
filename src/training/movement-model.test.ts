import { describe, expect, it } from "vitest";
import type { ReplayManifest } from "../capture/replay.js";
import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";
import {
  DEFAULT_MOVEMENT_TRAINING_CONFIG,
  MOVEMENT_END_TOKEN,
  MOVEMENT_START_TOKEN,
  NGramMovementBackend,
  buildMovementDatasetFromReplays,
  buildMovementDatasetFromTrajectories,
  buildMovementSequenceFromTrajectory,
  evaluateMovementModel,
  getMovementBackend,
  listMovementBackends,
  normalizeMovementToken,
  type MovementDataset,
} from "./movement-model.js";

function action(tool: string, summary: string, ts: number): TrajectoryAction {
  return { kind: "action", tool, summary, ts };
}

function trajectory(id: string, actions: TrajectoryAction[]): TrajectorySpan {
  return {
    id,
    sessionId: `session-${id}`,
    createdAt: "2026-07-21T00:00:00.000Z",
    captureTier: "full",
    observations: [],
    actions,
  };
}

/**
 * Deterministic synthetic "open-and-save" movement macro: move → click →
 * type → key. `seed` shifts timestamps and a coordinate metadata field so we
 * get distinct-but-related trajectories without any real OS input.
 */
function syntheticMacro(id: string, seed: number): TrajectorySpan {
  const base = seed * 1000;
  return trajectory(id, [
    action("mouse.move", "move to menu", base + 1),
    action("mouse.click", "click file", base + 2),
    action("keyboard.type", "type filename", base + 3),
    action("keyboard.key", "press enter", base + 4),
  ]);
}

describe("movement token normalization", () => {
  it("keys on tool plus a coarse verb so related movements share tokens", () => {
    expect(normalizeMovementToken({ tool: "Mouse.Move", summary: "Move to menu" })).toBe("mouse.move:move");
    expect(normalizeMovementToken({ tool: "keyboard.type", summary: "type filename" })).toBe("keyboard.type:type");
  });

  it("falls back to the bare tool when the summary has no simple verb", () => {
    expect(normalizeMovementToken({ tool: "mouse.click", summary: "42px offset" })).toBe("mouse.click");
    expect(normalizeMovementToken({ tool: "mouse.click" })).toBe("mouse.click");
  });
});

describe("dataset construction", () => {
  it("orders trajectory actions by timestamp and drops empty sequences", () => {
    const out = buildMovementSequenceFromTrajectory(
      trajectory("t1", [action("b", "second", 20), action("a", "first", 10)]),
    );
    expect(out.tokens).toEqual(["a:first", "b:second"]);

    const dataset = buildMovementDatasetFromTrajectories([trajectory("empty", [])]);
    expect(dataset.sequences).toHaveLength(0);
  });

  it("extracts action events from replay manifests", () => {
    const manifest: ReplayManifest = {
      version: 1,
      sessionId: "s1",
      trajectoryIds: ["t1"],
      eventCount: 3,
      events: [
        { kind: "observation", ts: 1, trajectoryId: "t1", source: "os", summary: "focus" },
        { kind: "action", ts: 2, trajectoryId: "t1", tool: "mouse.move", summary: "move up" },
        { kind: "action", ts: 3, trajectoryId: "t1", tool: "mouse.click", summary: "click ok" },
      ],
    };
    const dataset = buildMovementDatasetFromReplays([manifest]);
    expect(dataset.sequences[0].tokens).toEqual(["mouse.move:move", "mouse.click:click"]);
  });
});

describe("NGramMovementBackend", () => {
  const backend = new NGramMovementBackend();

  it("is registered as a pluggable backend", () => {
    expect(listMovementBackends()).toContain("ngram");
    expect(getMovementBackend("ngram")?.name).toBe("ngram");
  });

  it("repeats a recorded movement exactly (objective 2c)", () => {
    const dataset = buildMovementDatasetFromTrajectories([syntheticMacro("m1", 1)]);
    const model = backend.train(dataset);
    const rollout = model.generate([], 10);
    expect(rollout).toEqual(dataset.sequences[0].tokens);
  });

  it("generalizes to a new-but-related movement via back-off (objective 2d)", () => {
    // Train on macros that all start move→click→type→key, plus a variant that
    // diverges after "type" to a different key. The model should still predict a
    // plausible continuation for an unseen prefix that ends in a familiar suffix.
    const training = buildMovementDatasetFromTrajectories([
      syntheticMacro("m1", 1),
      syntheticMacro("m2", 2),
      trajectory("m3", [
        action("mouse.move", "move to toolbar", 1),
        action("mouse.click", "click save", 2),
        action("keyboard.type", "type note", 3),
        action("keyboard.key", "press escape", 4),
      ]),
    ]);
    const model = backend.train(training, { order: 2 });

    // Unseen prefix (starts with a click, no leading move) — back-off from the
    // "keyboard.type:type" context still yields a "keyboard.key" continuation.
    const prediction = model.predictNext([MOVEMENT_START_TOKEN, "mouse.click:click", "keyboard.type:type"]);
    expect(prediction?.token).toBe("keyboard.key:press");
    expect(prediction?.backoffOrder).toBeGreaterThan(0);
  });

  it("backs off to a lower order for unseen high-order contexts", () => {
    const dataset: MovementDataset = { sequences: [{ id: "s", tokens: ["a", "b", "c"] }] };
    const model = backend.train(dataset, { order: 3 });
    // Context "x b" was never seen at order 2, but "b" -> "c" was at order 1.
    const prediction = model.predictNext(["x", "b"]);
    expect(prediction?.token).toBe("c");
    expect(prediction?.backoffOrder).toBe(1);
  });

  it("produces a deterministic ranked distribution with lexical tie-breaks", () => {
    const dataset: MovementDataset = {
      sequences: [
        { id: "s1", tokens: ["a", "z"] },
        { id: "s2", tokens: ["a", "m"] },
      ],
    };
    const model = backend.train(dataset, { order: 1 });
    const dist = model.distribution(["a"]);
    // Both "m" and "z" follow "a" once; ties break lexically so "m" ranks first.
    expect(dist.map((entry) => entry.token)).toEqual(["m", "z"]);
    expect(dist[0].probability).toBeCloseTo(0.5);
  });

  it("round-trips through serialize/restore with identical predictions", () => {
    const dataset = buildMovementDatasetFromTrajectories([syntheticMacro("m1", 1), syntheticMacro("m2", 2)]);
    const model = backend.train(dataset, { order: 2 });
    const serialized = model.serialize();
    expect(serialized.backend).toBe("ngram");
    expect(serialized.order).toBe(2);
    expect(serialized.vocabulary).toContain(MOVEMENT_END_TOKEN);

    const restored = backend.restore(serialized);
    expect(restored.generate([], 10)).toEqual(model.generate([], 10));
    expect(restored.serialize()).toEqual(serialized);
  });

  it("returns no prediction from an empty model", () => {
    const model = backend.train({ sequences: [] });
    expect(model.predictNext(["anything"])).toBeUndefined();
    expect(model.generate([], 5)).toEqual([]);
  });

  it("uses the default order when none is supplied", () => {
    const model = backend.train({ sequences: [{ id: "s", tokens: ["a", "b"] }] });
    expect(model.order).toBe(DEFAULT_MOVEMENT_TRAINING_CONFIG.order);
  });
});

describe("evaluateMovementModel (generalization harness)", () => {
  const backend = new NGramMovementBackend();

  it("scores perfect fidelity on the training sequences", () => {
    const dataset = buildMovementDatasetFromTrajectories([syntheticMacro("m1", 1)]);
    const model = backend.train(dataset, { order: 3 });
    const result = evaluateMovementModel(model, dataset);
    expect(result.nextTokenAccuracy).toBe(1);
    expect(result.exactReplayRate).toBe(1);
    expect(result.replayFidelity).toBe(1);
    expect(result.evaluatedSequences).toBe(1);
  });

  it("generalizes to a held-out related trajectory above chance", () => {
    const train = buildMovementDatasetFromTrajectories([
      syntheticMacro("m1", 1),
      syntheticMacro("m2", 2),
      syntheticMacro("m3", 3),
    ]);
    const heldOut = buildMovementDatasetFromTrajectories([syntheticMacro("m4", 4)]);
    const model = backend.train(train, { order: 2 });
    const result = evaluateMovementModel(model, heldOut);
    // The held-out macro shares its whole token vocabulary/order with training,
    // so the model should reproduce it with high next-token accuracy.
    expect(result.nextTokenAccuracy).toBeGreaterThanOrEqual(0.8);
    expect(result.replayFidelity).toBeGreaterThan(0.5);
  });

  it("reports zero on an empty reference set", () => {
    const model = backend.train({ sequences: [{ id: "s", tokens: ["a"] }] });
    const result = evaluateMovementModel(model, { sequences: [] });
    expect(result.nextTokenAccuracy).toBe(0);
    expect(result.evaluatedSequences).toBe(0);
  });
});
