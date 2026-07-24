import { describe, expect, it } from "vitest";
import { buildReplayManifest } from "../capture/replay.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import {
  NgramMovementModel,
  evaluateNextTokenAccuracy,
  evaluateSequenceFidelity,
  movementTokenKey,
  tokenizeReplayManifest,
  tokenizeTrajectory,
  type MovementSequence,
  type MovementToken,
} from "./movement-model.js";

function seq(id: string, ...kinds: string[]): MovementSequence {
  return { id, tokens: kinds.map((kind) => ({ kind })) };
}

describe("movementTokenKey", () => {
  it("is stable and distinguishes target and direction", () => {
    expect(movementTokenKey({ kind: "swipe", direction: "up" })).toBe(
      movementTokenKey({ kind: "swipe", direction: "up" }),
    );
    expect(movementTokenKey({ kind: "swipe", direction: "up" })).not.toBe(
      movementTokenKey({ kind: "swipe", direction: "down" }),
    );
    expect(movementTokenKey({ kind: "tap", target: "ok" })).not.toBe(
      movementTokenKey({ kind: "tap", target: "cancel" }),
    );
  });
});

describe("NgramMovementModel training", () => {
  it("rejects a non-positive order", () => {
    expect(() => new NgramMovementModel(0)).toThrow(/positive integer/);
  });

  it("records vocabulary, starts, and transition counts", () => {
    const model = new NgramMovementModel(2);
    const state = model.train([seq("a", "tap", "type", "shortcut"), seq("b", "tap", "type", "shortcut")]);
    expect(state.sequenceCount).toBe(2);
    expect(state.tokenCount).toBe(6);
    expect(state.vocabulary).toEqual([...state.vocabulary].sort());
    // "tap" started both sequences.
    expect(state.starts[movementTokenKey({ kind: "tap" })]).toBe(2);
    // "type" follows "tap" twice at the unigram-plus-context level.
    const afterTap = state.transitions[movementTokenKey({ kind: "tap" })];
    expect(afterTap?.[movementTokenKey({ kind: "type" })]).toBe(2);
  });

  it("is deterministic — identical inputs produce identical state", () => {
    const model = new NgramMovementModel(3);
    const data = [seq("a", "tap", "type", "scroll"), seq("b", "tap", "scroll", "type")];
    expect(model.train(data)).toEqual(model.train(data));
  });
});

describe("repeat recorded movements", () => {
  it("reproduces the dominant recorded path from a seed", () => {
    const model = new NgramMovementModel(2);
    // A dominant path tap -> type -> shortcut -> tap, recorded 3x, plus noise.
    const state = model.train([
      seq("a", "tap", "type", "shortcut", "tap"),
      seq("b", "tap", "type", "shortcut", "tap"),
      seq("c", "tap", "type", "shortcut", "tap"),
      seq("noise", "tap", "scroll"),
    ]);
    const generated = model.generate(state, {
      seed: [{ kind: "tap" }],
      maxLength: 4,
    });
    expect(generated.map((token) => token.kind)).toEqual(["tap", "type", "shortcut", "tap"]);
  });

  it("uses the most common recorded start when no seed is given", () => {
    const model = new NgramMovementModel(1);
    const state = model.train([
      seq("a", "tap", "type"),
      seq("b", "tap", "type"),
      seq("c", "scroll", "type"),
    ]);
    const generated = model.generate(state, { maxLength: 2 });
    expect(generated[0]?.kind).toBe("tap");
  });
});

describe("generalize to new but related movements", () => {
  it("backs off to a shared suffix for an unseen prefix", () => {
    const model = new NgramMovementModel(2);
    // Training always has "type" following "shortcut", regardless of what came before.
    const state = model.train([
      seq("a", "tap", "shortcut", "type"),
      seq("b", "tap", "shortcut", "type"),
      seq("c", "swipe", "shortcut", "type"),
    ]);
    // Novel prefix: scroll -> shortcut (the (scroll, shortcut) bigram was never seen).
    const prediction = model.predictNext(state, [{ kind: "scroll" }, { kind: "shortcut" }]);
    expect(prediction).toBeDefined();
    expect(prediction?.token.kind).toBe("type");
    // It generalized by backing off from order-2 context to the order-1 "shortcut".
    expect(prediction?.contextOrder).toBe(1);
  });

  it("returns undefined when the model is empty", () => {
    const model = new NgramMovementModel(2);
    const state = model.train([]);
    expect(model.predictNext(state, [{ kind: "tap" }])).toBeUndefined();
    expect(model.generate(state)).toEqual([]);
  });

  it("reports a conditional probability for the prediction", () => {
    const model = new NgramMovementModel(1);
    const state = model.train([seq("a", "tap", "type"), seq("b", "tap", "type"), seq("c", "tap", "scroll")]);
    const prediction = model.predictNext(state, [{ kind: "tap" }]);
    expect(prediction?.token.kind).toBe("type");
    expect(prediction?.observationCount).toBe(2);
    // type:2, scroll:1 -> 2/3.
    expect(prediction?.probability).toBeCloseTo(2 / 3, 10);
  });
});

describe("generation guards", () => {
  it("respects a stop token and the max length", () => {
    const model = new NgramMovementModel(1);
    const state = model.train([seq("a", "tap", "type", "shortcut")]);
    const stopped = model.generate(state, { seed: [{ kind: "tap" }], stopToken: { kind: "type" } });
    expect(stopped.map((t) => t.kind)).toEqual(["tap", "type"]);

    // A self-cycle would loop forever without the length cap.
    const cyclic = model.train([seq("loop", "tap", "tap", "tap")]);
    const capped = model.generate(cyclic, { seed: [{ kind: "tap" }], maxLength: 5 });
    expect(capped).toHaveLength(5);
  });
});

describe("tokenizers bridge the capture schema", () => {
  it("tokenizes a trajectory's actions in timestamp order using gesture metadata", () => {
    const span = buildTrajectorySpan({
      id: "traj-1",
      sessionId: "sess-1",
      actions: [
        { kind: "action", tool: "device", summary: "swiped up", ts: 20, metadata: { gesture: "swipe", direction: "up" } },
        { kind: "action", tool: "device", summary: "tapped ok", ts: 10, metadata: { gesture: "tap", target: "ok" } },
      ],
    });
    const sequence = tokenizeTrajectory(span);
    expect(sequence.id).toBe("traj-1");
    expect(sequence.tokens).toEqual([
      { kind: "tap", target: "ok" },
      { kind: "swipe", direction: "up" },
    ]);
  });

  it("tokenizes a replay manifest's action timeline", () => {
    const span = buildTrajectorySpan({
      id: "traj-1",
      sessionId: "sess-1",
      actions: [{ kind: "action", tool: "device", summary: "tapped ok", ts: 10, metadata: { gesture: "tap" } }],
    });
    const manifest = buildReplayManifest({ sessionId: "sess-1", transcript: [], trajectories: [span] });
    const sequence = tokenizeReplayManifest(manifest);
    expect(sequence.tokens).toEqual([{ kind: "device", target: "tapped ok" }]);
  });

  it("round-trips: train on a tokenized trajectory and repeat it", () => {
    const span = buildTrajectorySpan({
      id: "traj-1",
      sessionId: "sess-1",
      actions: [
        { kind: "action", tool: "device", summary: "tap", ts: 1, metadata: { gesture: "tap", target: "menu" } },
        { kind: "action", tool: "device", summary: "type", ts: 2, metadata: { gesture: "type", target: "search" } },
      ],
    });
    const model = new NgramMovementModel(2);
    const state = model.train([tokenizeTrajectory(span)]);
    const generated = model.generate(state, { maxLength: 2 });
    expect(generated).toEqual([
      { kind: "tap", target: "menu" },
      { kind: "type", target: "search" },
    ]);
  });
});

describe("evaluation harnesses", () => {
  it("scores sequence fidelity position-wise", () => {
    const expected: MovementToken[] = [{ kind: "tap" }, { kind: "type" }, { kind: "scroll" }];
    const predicted: MovementToken[] = [{ kind: "tap" }, { kind: "type" }, { kind: "shortcut" }];
    expect(evaluateSequenceFidelity(predicted, expected)).toEqual({ matched: 2, total: 3, accuracy: 2 / 3 });
    expect(evaluateSequenceFidelity([], [])).toEqual({ matched: 0, total: 0, accuracy: 1 });
  });

  it("measures held-out next-token accuracy and generalization backoff", () => {
    const model = new NgramMovementModel(2);
    const state = model.train([
      seq("a", "tap", "type", "shortcut"),
      seq("b", "tap", "type", "shortcut"),
    ]);
    // Held-out sequence with a novel start that shares the "type -> shortcut" suffix.
    const result = evaluateNextTokenAccuracy(model, state, [seq("h", "scroll", "type", "shortcut")]);
    expect(result.evaluated).toBe(2);
    // "type" -> "shortcut" is learned; the (scroll -> ?) and (scroll,type -> ?) both back off.
    expect(result.correct).toBeGreaterThanOrEqual(1);
    expect(result.backedOff).toBeGreaterThanOrEqual(1);
    expect(result.accuracy).toBeCloseTo(result.correct / result.evaluated, 10);
  });
});
