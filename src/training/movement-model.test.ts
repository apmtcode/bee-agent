import { describe, expect, it } from "vitest";
import { buildTrajectorySpan, type TrajectoryAction } from "../capture/trajectory.js";
import {
  buildMovementDataset,
  createDefaultMovementBackendRegistry,
  deriveMovementToken,
  evaluateNextTokenAccuracy,
  MarkovMovementBackend,
  MovementBackendRegistry,
  tokenizeTrajectory,
  type MovementModelBackend,
} from "./movement-model.js";
import {
  DEFAULT_SYNTHETIC_WORKFLOWS,
  generateSyntheticMovementTrajectories,
} from "./movement-synthetic.js";

function action(tool: string, summary: string, ts: number, metadata?: Record<string, unknown>): TrajectoryAction {
  return { kind: "action", tool, summary, ts, ...(metadata ? { metadata } : {}) };
}

function span(id: string, actions: TrajectoryAction[]) {
  return buildTrajectorySpan({ id, sessionId: `session-${id}`, actions });
}

describe("deriveMovementToken", () => {
  it("prefers structured metadata over free text and normalizes segments", () => {
    expect(
      deriveMovementToken(action("device", "tapped the Submit Button", 1, { gesture: "tap", target: "Submit Button" })),
    ).toBe("device:tap:submit-button");
  });

  it("falls back to the leading verb and trailing target of the summary", () => {
    expect(deriveMovementToken(action("os", "focused Editor Window", 1))).toBe("os:focused:editor-window");
  });

  it("emits a tool:verb token when there is no target", () => {
    expect(deriveMovementToken(action("device", "scrolled", 1, { gesture: "scroll" }))).toBe("device:scroll");
  });
});

describe("tokenizeTrajectory / buildMovementDataset", () => {
  it("orders actions by timestamp and collects a sorted vocabulary", () => {
    const trajectory = span("t1", [
      action("device", "type query", 30, { gesture: "type", target: "query" }),
      action("os", "focus-changed browser", 10, { event: "focus-changed", target: "browser" }),
      action("device", "tap address-bar", 20, { gesture: "tap", target: "address-bar" }),
    ]);

    expect(tokenizeTrajectory(trajectory)).toEqual([
      "os:focus-changed:browser",
      "device:tap:address-bar",
      "device:type:query",
    ]);

    const dataset = buildMovementDataset([trajectory]);
    expect(dataset.samples).toHaveLength(1);
    expect(dataset.vocabulary).toEqual([
      "device:tap:address-bar",
      "device:type:query",
      "os:focus-changed:browser",
    ]);
  });

  it("drops trajectories with no actions", () => {
    expect(buildMovementDataset([span("empty", [])]).samples).toHaveLength(0);
  });
});

describe("MarkovMovementBackend", () => {
  const backend = new MarkovMovementBackend();

  it("repeats a recorded movement sequence exactly (objective 2c)", () => {
    const recorded = span("rec", [
      action("os", "focus-changed editor", 1, { event: "focus-changed", target: "editor" }),
      action("device", "tap open-file", 2, { gesture: "tap", target: "open-file" }),
      action("device", "type code", 3, { gesture: "type", target: "code" }),
      action("device", "shortcut save", 4, { gesture: "shortcut", target: "save" }),
    ]);
    const tokens = tokenizeTrajectory(recorded);
    const model = backend.train(buildMovementDataset([recorded]), { order: 3 });

    // From a clean start the model should regenerate the recorded sequence.
    expect(model.generate([], 10)).toEqual(tokens);
  });

  it("generalizes to a new-but-related context via backoff (objective 2d)", () => {
    // Two workflows that share a common suffix step: ... -> type:code -> save.
    const a = span("a", [
      action("os", "focus-changed editor", 1, { event: "focus-changed", target: "editor" }),
      action("device", "type code", 2, { gesture: "type", target: "code" }),
      action("device", "shortcut save", 3, { gesture: "shortcut", target: "save" }),
    ]);
    const b = span("b", [
      action("os", "focus-changed notes", 1, { event: "focus-changed", target: "notes" }),
      action("device", "type code", 2, { gesture: "type", target: "code" }),
      action("device", "shortcut save", 3, { gesture: "shortcut", target: "save" }),
    ]);
    const model = backend.train(buildMovementDataset([a, b]), { order: 3 });

    // A novel COMBINATION of seen tokens — focusing "gallery" (never preceded
    // type:code) then typing code — was never recorded as a pair, yet the model
    // should still predict the shared "save" continuation by backing off to the
    // seen "type:code" suffix. That backoff is the generalization mechanism.
    const prediction = model.predict(["os:focus-changed:gallery", "device:type:code"]);
    expect(prediction.token).toBe("device:shortcut:save");
    expect(prediction.source).toBe("backoff");
    expect(prediction.matchedOrder).toBe(1);
  });

  it("returns a deterministic prediction with ranked alternatives", () => {
    const model = backend.train(
      buildMovementDataset([
        span("x1", [action("device", "tap a", 1, { gesture: "tap", target: "a" }), action("device", "tap b", 2, { gesture: "tap", target: "b" })]),
        span("x2", [action("device", "tap a", 1, { gesture: "tap", target: "a" }), action("device", "tap c", 2, { gesture: "tap", target: "c" })]),
      ]),
      { order: 2 },
    );

    const first = model.predict(["device:tap:a"]);
    const second = model.predict(["device:tap:a"]);
    expect(first).toEqual(second);
    expect(first.alternatives.map((entry) => entry.token)).toEqual(["device:tap:b", "device:tap:c"]);
  });

  it("round-trips through a snapshot", () => {
    const trajectory = span("snap", [
      action("device", "tap a", 1, { gesture: "tap", target: "a" }),
      action("device", "tap b", 2, { gesture: "tap", target: "b" }),
    ]);
    const model = backend.train(buildMovementDataset([trajectory]), { order: 2 });
    const restored = backend.restore(model.snapshot());

    expect(restored.snapshot()).toEqual(model.snapshot());
    expect(restored.predict(["device:tap:a"])).toEqual(model.predict(["device:tap:a"]));
  });

  it("predicts nothing for an empty model", () => {
    const model = backend.train(buildMovementDataset([]), { order: 2 });
    expect(model.predict([]).source).toBe("none");
    expect(model.generate([], 5)).toEqual([]);
  });
});

describe("MovementBackendRegistry", () => {
  it("registers, looks up and lists pluggable backends", () => {
    const registry = createDefaultMovementBackendRegistry();
    expect(registry.list()).toContain("markov-backoff");
    expect(registry.get("markov-backoff")).toBeInstanceOf(MarkovMovementBackend);
    expect(() => registry.get("nope")).toThrow(/Unknown movement backend/);
  });

  it("accepts a custom backend implementing the interface", () => {
    const custom: MovementModelBackend = {
      id: "custom-stub",
      train: () => ({
        backendId: "custom-stub",
        order: 1,
        predict: () => ({ token: null, probability: 0, matchedOrder: 0, source: "none", alternatives: [] }),
        generate: () => [],
        snapshot: () => ({ version: 1, backendId: "custom-stub", order: 1, vocabulary: [], transitions: {} }),
      }),
      restore: () => {
        throw new Error("not implemented");
      },
    };
    const registry = new MovementBackendRegistry().register(custom);
    expect(registry.has("custom-stub")).toBe(true);
  });
});

describe("synthetic movement generator + eval harness", () => {
  it("is deterministic for a given seed", () => {
    const a = generateSyntheticMovementTrajectories({ seed: 7, sessions: 5 });
    const b = generateSyntheticMovementTrajectories({ seed: 7, sessions: 5 });
    expect(a.map((t) => tokenizeTrajectory(t))).toEqual(b.map((t) => tokenizeTrajectory(t)));
    expect(a).toHaveLength(5);
  });

  it("trains a model that recalls its training movements with high accuracy", () => {
    const trajectories = generateSyntheticMovementTrajectories({ seed: 11, sessions: 40 });
    const dataset = buildMovementDataset(trajectories);
    const model = new MarkovMovementBackend().train(dataset, { order: 3 });

    const recall = evaluateNextTokenAccuracy(model, dataset.samples);
    expect(recall.total).toBeGreaterThan(0);
    expect(recall.accuracy).toBeGreaterThan(0.7);
  });

  it("generalizes to held-out related trajectories better than chance", () => {
    const workflows = DEFAULT_SYNTHETIC_WORKFLOWS;
    const train = generateSyntheticMovementTrajectories({ seed: 3, sessions: 60, workflows });
    const heldOut = generateSyntheticMovementTrajectories({ seed: 999, sessions: 20, workflows });

    const model = new MarkovMovementBackend().train(buildMovementDataset(train), { order: 3 });
    const evalResult = evaluateNextTokenAccuracy(model, buildMovementDataset(heldOut).samples);

    // Held-out sequences are drawn from the same workflow library, so a model
    // that has learned the structure should predict most next-movements.
    expect(evalResult.accuracy).toBeGreaterThan(0.6);
  });
});
