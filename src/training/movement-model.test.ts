import { describe, expect, it } from "vitest";
import { buildTrajectorySpan, type TrajectoryAction, type TrajectorySpan } from "../capture/trajectory.js";
import {
  buildMovementDataset,
  createDefaultMovementModelRegistry,
  MOVEMENT_START_TOKEN,
  MovementModelRegistry,
  NgramMovementBackend,
  replayFidelity,
  rolloutMovements,
  tokenizeAction,
  type MovementModelBackend,
  type MovementToken,
} from "./movement-model.js";

function action(tool: string, summary: string, ts: number, metadata?: Record<string, unknown>): TrajectoryAction {
  return { kind: "action", tool, summary, ts, ...(metadata ? { metadata } : {}) };
}

function span(id: string, actions: TrajectoryAction[]): TrajectorySpan {
  return buildTrajectorySpan({ id, sessionId: `session-${id}`, actions });
}

/** Synthetic gesture stream: swipe up, tap "row-N", type. */
function gestureSequence(id: string, rowIndex: number): TrajectorySpan {
  return span(id, [
    action("device", "swiped up", 10, { gesture: "swipe", direction: "up" }),
    action("device", `tapped row-${rowIndex}`, 20, { gesture: "tap", target: `row-${rowIndex}` }),
    action("device", "typed into field", 30, { gesture: "type", target: "field" }),
  ]);
}

describe("tokenizeAction", () => {
  it("derives a canonical token from gesture metadata", () => {
    expect(tokenizeAction(action("device", "swiped up", 1, { gesture: "swipe", direction: "up" }))).toBe(
      "device:swipe:up",
    );
  });

  it("buckets numeric targets so related movements share a token", () => {
    const a = tokenizeAction(action("device", "tapped row-3", 1, { gesture: "tap", target: "row-3" }));
    const b = tokenizeAction(action("device", "tapped row-7", 1, { gesture: "tap", target: "row-7" }));
    expect(a).toBe(b);
    expect(a).toBe("device:tap:row");
  });

  it("falls back to a slug of the summary when there is no gesture", () => {
    expect(tokenizeAction(action("shell", "Ran Build Step", 1))).toBe("shell:ran-build-step");
  });
});

describe("buildMovementDataset", () => {
  it("orders actions by timestamp and collects a sorted vocabulary", () => {
    const dataset = buildMovementDataset([
      span("t1", [
        action("device", "typed", 30, { gesture: "type", target: "field" }),
        action("device", "swiped up", 10, { gesture: "swipe", direction: "up" }),
      ]),
    ]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]!.tokens).toEqual(["device:swipe:up", "device:type:field"]);
    expect(dataset.vocabulary).toEqual(["device:swipe:up", "device:type:field"]);
  });

  it("skips trajectories with no actions", () => {
    const dataset = buildMovementDataset([span("empty", [])]);
    expect(dataset.sequences).toHaveLength(0);
  });
});

describe("NgramMovementBackend", () => {
  const backend = new NgramMovementBackend();

  it("replays a recorded movement sequence verbatim (objective 2c)", () => {
    const dataset = buildMovementDataset([gestureSequence("t1", 3)]);
    const model = backend.train(dataset);
    const produced = rolloutMovements(backend, model);
    expect(produced).toEqual(["device:swipe:up", "device:tap:row", "device:type:field"]);
    expect(replayFidelity(dataset.sequences[0]!.tokens, produced)).toBe(1);
  });

  it("is deterministic across repeated training + rollout", () => {
    const dataset = buildMovementDataset([gestureSequence("t1", 3), gestureSequence("t2", 5)]);
    const first = rolloutMovements(backend, backend.train(dataset));
    const second = rolloutMovements(backend, backend.train(dataset));
    expect(first).toEqual(second);
  });

  it("generalizes to a novel-but-related prefix via backoff (objective 2d)", () => {
    // Two recorded patterns share the tail (tap -> type) but differ in how they
    // start. A never-seen exact prefix should still back off to the shared
    // lower-order statistics and continue sensibly.
    const dataset = buildMovementDataset([
      span("open", [
        action("device", "shortcut", 10, { gesture: "shortcut", target: "spotlight" }),
        action("device", "tapped item", 20, { gesture: "tap", target: "item-1" }),
        action("device", "typed", 30, { gesture: "type", target: "field" }),
      ]),
      span("scroll", [
        action("device", "scrolled down", 10, { gesture: "scroll", direction: "down" }),
        action("device", "tapped item", 20, { gesture: "tap", target: "item-2" }),
        action("device", "typed", 30, { gesture: "type", target: "field" }),
      ]),
    ]);
    const model = backend.train(dataset, { order: 2 });

    // Novel context: a "tap" that was never preceded by this exact 2-gram in a
    // way the top-order context covers — the model must still know a tap tends
    // to be followed by a type.
    const prediction = backend.predict(model, ["device:tap:item"]);
    expect(prediction.token).toBe("device:type:field");
    expect(prediction.backoffOrder).toBeLessThanOrEqual(1);
    expect(prediction.probability).toBeGreaterThan(0);
    expect(prediction.candidates[0]!.token).toBe("device:type:field");
  });

  it("respects the rollout token cap", () => {
    // A self-looping pattern would run forever without the cap.
    const dataset = buildMovementDataset([
      span("loop", [
        action("device", "swiped up", 10, { gesture: "swipe", direction: "up" }),
        action("device", "swiped up", 20, { gesture: "swipe", direction: "up" }),
        action("device", "swiped up", 30, { gesture: "swipe", direction: "up" }),
      ]),
    ]);
    const model = backend.train(dataset, { order: 1 });
    const produced = rolloutMovements(backend, model, { maxTokens: 5 });
    expect(produced.length).toBeLessThanOrEqual(5);
  });

  it("emits the end sentinel for an empty model", () => {
    const model = backend.train(buildMovementDataset([]));
    const produced = rolloutMovements(backend, model);
    expect(produced).toEqual([]);
    const prediction = backend.predict(model, [MOVEMENT_START_TOKEN]);
    expect(prediction.token).toBe("<end>");
  });

  it("can continue from a mid-sequence context (new movement from a seed)", () => {
    const dataset = buildMovementDataset([gestureSequence("t1", 3)]);
    const model = backend.train(dataset);
    const produced = rolloutMovements(backend, model, { context: ["device:swipe:up"] });
    expect(produced).toEqual(["device:tap:row", "device:type:field"]);
  });
});

describe("replayFidelity", () => {
  it("scores partial reproduction by longest common subsequence", () => {
    const reference: MovementToken[] = ["a", "b", "c", "d"];
    expect(replayFidelity(reference, ["a", "b", "c", "d"])).toBe(1);
    expect(replayFidelity(reference, ["a", "x", "c", "y"])).toBe(0.5);
    expect(replayFidelity([], [])).toBe(1);
    expect(replayFidelity(["a"], [])).toBe(0);
  });
});

describe("MovementModelRegistry", () => {
  it("registers and requires backends by name", () => {
    const registry = new MovementModelRegistry().register(new NgramMovementBackend());
    expect(registry.list()).toEqual(["ngram-mock"]);
    expect(registry.get("ngram-mock")).toBeInstanceOf(NgramMovementBackend);
    expect(registry.get("missing")).toBeUndefined();
    expect(() => registry.require("missing")).toThrow(/unknown movement model backend/);
  });

  it("default registry is pre-seeded with the mock backend and satisfies the interface", () => {
    const registry = createDefaultMovementModelRegistry();
    const backend: MovementModelBackend = registry.require("ngram-mock");
    const model = backend.train(buildMovementDataset([gestureSequence("t1", 1)]));
    expect(rolloutMovements(backend, model)).toEqual([
      "device:swipe:up",
      "device:tap:row",
      "device:type:field",
    ]);
  });
});
