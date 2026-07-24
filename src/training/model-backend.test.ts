import { describe, expect, it } from "vitest";
import type { TrajectorySpan, TrajectoryAction } from "../capture/trajectory.js";
import {
  DeterministicMarkovMovementBackend,
  MovementModelBackendRegistry,
  buildMovementDataset,
  evaluateMovementModel,
  movementTokenKey,
  rolloutMovements,
  type MovementActionToken,
} from "./model-backend.js";

function action(tool: string, summary: string, ts: number): TrajectoryAction {
  return { kind: "action", tool, summary, ts };
}

function span(id: string, actions: TrajectoryAction[], reward?: number): TrajectorySpan {
  return {
    id,
    sessionId: `session-${id}`,
    createdAt: "2026-07-24T00:00:00.000Z",
    captureTier: "operator",
    observations: [],
    actions,
    ...(reward === undefined ? {} : { outcome: { status: "success", summary: "ok", reward } }),
  };
}

/** A synthetic "open → focus → type → save" movement macro. */
function macro(id: string, target: string, reward?: number): TrajectorySpan {
  return span(
    id,
    [
      action("mouse.click", `open ${target}`, 10),
      action("window.focus", `focus ${target}`, 20),
      action("keyboard.type", `type into ${target}`, 30),
      action("keyboard.shortcut", `save ${target}`, 40),
    ],
    reward,
  );
}

describe("buildMovementDataset", () => {
  it("slides a window over time-ordered actions", () => {
    const dataset = buildMovementDataset([macro("t1", "editor")], { contextWindow: 2 });
    expect(dataset.contextWindow).toBe(2);
    // 4 actions → 3 (context → next) examples.
    expect(dataset.examples).toHaveLength(3);
    expect(dataset.examples[0]).toMatchObject({
      context: [{ tool: "mouse.click", summary: "open editor" }],
      next: { tool: "window.focus", summary: "focus editor" },
    });
    // Window caps context length at 2.
    expect(dataset.examples[2]!.context).toHaveLength(2);
  });

  it("sorts unordered actions by timestamp before windowing", () => {
    const unordered = span("t1", [
      action("c", "third", 30),
      action("a", "first", 10),
      action("b", "second", 20),
    ]);
    const dataset = buildMovementDataset([unordered], { contextWindow: 3 });
    expect(dataset.examples[0]!.context.map((t) => t.tool)).toEqual(["a"]);
    expect(dataset.examples[1]!.next.tool).toBe("c");
  });

  it("skips trajectories with fewer than two actions", () => {
    const dataset = buildMovementDataset([span("t1", [action("only", "one", 10)])]);
    expect(dataset.examples).toHaveLength(0);
  });

  it("prefers reviewed/redacted actions and derives positive weights from reward", () => {
    const reviewed: TrajectorySpan = {
      ...macro("t1", "editor"),
      review: {
        status: "approved",
        reviewedAt: "2026-07-24T00:00:00.000Z",
        reviewedBy: "reviewer",
        redactedActions: [
          { tool: "safe.a", summary: "one", ts: 1 },
          { tool: "safe.b", summary: "two", ts: 2 },
        ],
      },
      outcome: { status: "failure", summary: "bad", reward: -5 },
    };
    const dataset = buildMovementDataset([reviewed], { contextWindow: 2 });
    expect(dataset.examples).toHaveLength(1);
    expect(dataset.examples[0]!.next.tool).toBe("safe.b");
    // reward -5 clamps to a small strictly-positive weight, never 0/negative.
    expect(dataset.examples[0]!.weight).toBeGreaterThan(0);
  });
});

describe("DeterministicMarkovMovementBackend", () => {
  const backend = new DeterministicMarkovMovementBackend();

  it("replays a recorded movement sequence via exact suffix match", () => {
    const dataset = buildMovementDataset([macro("t1", "editor")], { contextWindow: 3 });
    const model = backend.train(dataset, { trainedAt: "2026-07-24T00:00:00.000Z" });

    const prediction = backend.predict(model, [
      { tool: "mouse.click", summary: "open editor" },
      { tool: "window.focus", summary: "focus editor" },
    ]);
    expect(prediction.source).toBe("exact");
    expect(prediction.action).toEqual({ tool: "keyboard.type", summary: "type into editor" });
    expect(prediction.confidence).toBeGreaterThan(0.9);
  });

  it("is deterministic across repeated training/prediction", () => {
    const dataset = buildMovementDataset([macro("t1", "editor"), macro("t2", "browser")], { contextWindow: 3 });
    const a = backend.train(dataset);
    const b = backend.train(dataset);
    const context: MovementActionToken[] = [{ tool: "window.focus", summary: "focus editor" }];
    expect(backend.predict(a, context)).toEqual(backend.predict(b, context));
  });

  it("generalizes to a new-but-related context via tool-level back-off", () => {
    // Train on two targets so tool transitions are established.
    const dataset = buildMovementDataset([macro("t1", "editor"), macro("t2", "browser")], { contextWindow: 3 });
    const model = backend.train(dataset);

    // Unseen summary ("focus spreadsheet") → no exact suffix, but the tool
    // window.focus was always followed by keyboard.type.
    const prediction = backend.predict(model, [{ tool: "window.focus", summary: "focus spreadsheet" }]);
    expect(prediction.source).toBe("generalized");
    expect(prediction.action?.tool).toBe("keyboard.type");
    expect(prediction.confidence).toBeGreaterThan(0);
    expect(prediction.confidence).toBeLessThan(1);
  });

  it("falls back to the global prior for an empty context", () => {
    const dataset = buildMovementDataset([macro("t1", "editor")], { contextWindow: 3 });
    const model = backend.train(dataset);
    const prediction = backend.predict(model, []);
    expect(prediction.source).toBe("prior");
    expect(prediction.action).toBeDefined();
  });

  it("returns a none prediction when the model has no data", () => {
    const empty = backend.train({ version: 1, contextWindow: 2, examples: [] });
    const prediction = backend.predict(empty, [{ tool: "x", summary: "y" }]);
    expect(prediction.source).toBe("none");
    expect(prediction.action).toBeUndefined();
    expect(prediction.candidates).toHaveLength(0);
  });

  it("weights higher-reward continuations above lower-reward ones", () => {
    // Same context leads to two different next moves; the high-reward one wins.
    const good = span("good", [action("start", "s", 1), action("finish", "good-path", 2)], 5);
    const bad = span("bad", [action("start", "s", 1), action("finish", "bad-path", 2)], -0.5);
    const dataset = buildMovementDataset([good, bad], { contextWindow: 1 });
    const model = backend.train(dataset);
    const prediction = backend.predict(model, [{ tool: "start", summary: "s" }]);
    expect(prediction.action?.summary).toBe("good-path");
  });
});

describe("rolloutMovements", () => {
  it("regenerates the full recorded macro from a seed", () => {
    const backend = new DeterministicMarkovMovementBackend();
    const dataset = buildMovementDataset([macro("t1", "editor")], { contextWindow: 3 });
    const model = backend.train(dataset);

    const produced = rolloutMovements(backend, model, [{ tool: "mouse.click", summary: "open editor" }], {
      maxSteps: 5,
      minConfidence: 0.5,
    });
    expect(produced.map((t) => t.tool)).toEqual(["window.focus", "keyboard.type", "keyboard.shortcut"]);
    expect(produced.every((t) => t.source === "exact")).toBe(true);
  });

  it("stops when stopWhen matches", () => {
    const backend = new DeterministicMarkovMovementBackend();
    const dataset = buildMovementDataset([macro("t1", "editor")], { contextWindow: 3 });
    const model = backend.train(dataset);
    const produced = rolloutMovements(backend, model, [{ tool: "mouse.click", summary: "open editor" }], {
      maxSteps: 5,
      stopWhen: (token) => token.tool === "keyboard.type",
    });
    expect(produced.map((t) => t.tool)).toEqual(["window.focus", "keyboard.type"]);
  });
});

describe("evaluateMovementModel", () => {
  it("reports high fidelity on a held-out related trajectory", () => {
    const backend = new DeterministicMarkovMovementBackend();
    const train = buildMovementDataset([macro("t1", "editor"), macro("t2", "browser")], { contextWindow: 3 });
    const model = backend.train(train);

    // Held-out macro over a brand-new target: same tool structure, new summaries.
    const evaluation = evaluateMovementModel(backend, model, [macro("t3", "terminal")]);
    expect(evaluation.total).toBe(3);
    expect(evaluation.coverage).toBe(1);
    // Every move is a known tool, so tool accuracy is perfect even though the
    // summaries ("... terminal") were never seen — that is the generalization.
    expect(evaluation.toolAccuracy).toBe(1);
    expect(evaluation.exactAccuracy).toBeLessThan(1);
    expect(evaluation.sourceCounts.generalized).toBeGreaterThan(0);
  });

  it("handles empty held-out sets without dividing by zero", () => {
    const backend = new DeterministicMarkovMovementBackend();
    const model = backend.train({ version: 1, contextWindow: 2, examples: [] });
    const evaluation = evaluateMovementModel(backend, model, []);
    expect(evaluation).toMatchObject({ total: 0, exactAccuracy: 0, toolAccuracy: 0, coverage: 0 });
  });
});

describe("MovementModelBackendRegistry", () => {
  it("provides the deterministic backend by default and resolves by id", () => {
    const registry = new MovementModelBackendRegistry();
    expect(registry.list()).toContain("deterministic-markov");
    expect(registry.require("deterministic-markov").id).toBe("deterministic-markov");
  });

  it("throws for an unknown backend and accepts custom registrations", () => {
    const registry = new MovementModelBackendRegistry([]);
    expect(() => registry.require("missing")).toThrow(/unknown movement-model backend/);
    registry.register(new DeterministicMarkovMovementBackend());
    expect(registry.get("deterministic-markov")).toBeDefined();
  });
});

describe("movementTokenKey", () => {
  it("is stable and round-trips distinct tokens to distinct keys", () => {
    const a = movementTokenKey({ tool: "mouse.click", summary: "open editor" });
    const b = movementTokenKey({ tool: "mouse.click", summary: "open browser" });
    expect(a).not.toBe(b);
    expect(movementTokenKey({ tool: "mouse.click", summary: "open editor" })).toBe(a);
  });
});
