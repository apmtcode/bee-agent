import { describe, expect, it } from "vitest";
import type { ReplayManifest } from "../capture/replay.js";
import { buildTrajectorySpan, type TrajectoryAction } from "../capture/trajectory.js";
import {
  DeterministicMarkovBackend,
  MovementBackendRegistry,
  buildMovementDataset,
  evaluateGeneralization,
  movementSequenceFromReplay,
  movementSequenceFromTrajectory,
  tokenizeMovementAction,
  type LocalModelBackend,
  type MovementDataset,
} from "./movement-model.js";

function action(tool: string, summary: string, ts: number, metadata?: Record<string, unknown>): TrajectoryAction {
  return { kind: "action", tool, summary, ts, ...(metadata ? { metadata } : {}) };
}

function datasetOf(...sequences: string[][]): MovementDataset {
  return {
    version: 1,
    sequences: sequences.map((tokens, index) => ({ id: `seq-${index}`, tokens })),
  };
}

describe("tokenizeMovementAction", () => {
  it("prefers gesture + target metadata and normalizes it", () => {
    const token = tokenizeMovementAction(
      action("device", "tapped Submit Button", 1, { gesture: "tap", target: "Submit Button" }),
    );
    expect(token).toBe("tap:submit-button");
  });

  it("falls back to direction, then to the first summary word", () => {
    expect(tokenizeMovementAction(action("device", "scrolled", 1, { gesture: "scroll", direction: "down" }))).toBe(
      "scroll:down",
    );
    expect(tokenizeMovementAction(action("browser", "clicked link", 1))).toBe("browser:clicked");
  });
});

describe("DeterministicMarkovBackend", () => {
  it("replays a memorized full-order sequence without generalizing", () => {
    const backend = new DeterministicMarkovBackend();
    const model = backend.train(datasetOf(["a", "b", "c"]), { order: 2 });

    const prediction = backend.predict(model, ["a", "b"]);
    expect(prediction.action).toBe("c");
    expect(prediction.confidence).toBe(1);
    expect(prediction.order).toBe(2);
    expect(prediction.generalized).toBe(false);
  });

  it("generalizes via backoff when the exact context was never seen", () => {
    const backend = new DeterministicMarkovBackend();
    // "open" is always followed by "menu"; the full 2-gram "x open" is unseen.
    const model = backend.train(datasetOf(["home", "open", "menu"], ["settings", "open", "menu"]), { order: 2 });

    const prediction = backend.predict(model, ["unseen-app", "open"]);
    expect(prediction.action).toBe("menu");
    expect(prediction.generalized).toBe(true);
    expect(prediction.order).toBe(1);
  });

  it("is deterministic and breaks probability ties lexically", () => {
    const backend = new DeterministicMarkovBackend();
    const model = backend.train(datasetOf(["x", "b"], ["x", "a"]), { order: 1 });

    const first = backend.predict(model, ["x"]);
    const second = backend.predict(model, ["x"]);
    expect(first).toEqual(second);
    expect(first.action).toBe("a");
    expect(first.candidates.map((candidate) => candidate.action)).toEqual(["a", "b"]);
  });

  it("returns an abstaining prediction for an empty model", () => {
    const backend = new DeterministicMarkovBackend();
    const model = backend.train(datasetOf(), { order: 2 });
    const prediction = backend.predict(model, ["a"]);
    expect(prediction.action).toBeUndefined();
    expect(prediction.candidates).toEqual([]);
  });

  it("produces a JSON-serializable model artifact", () => {
    const backend = new DeterministicMarkovBackend();
    const model = backend.train(datasetOf(["a", "b", "c"]), { order: 2 });
    const roundTripped = JSON.parse(JSON.stringify(model));
    expect(roundTripped.vocabulary).toEqual(["a", "b", "c"]);
    expect(backend.predict(roundTripped, ["a", "b"]).action).toBe("c");
  });
});

describe("dataset builders", () => {
  it("derives an ordered token sequence from a trajectory span", () => {
    const span = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [
        action("device", "tapped Menu", 20, { gesture: "tap", target: "Menu" }),
        action("device", "swiped left", 10, { gesture: "swipe", direction: "left" }),
      ],
    });
    const sequence = movementSequenceFromTrajectory(span);
    expect(sequence.tokens).toEqual(["swipe:left", "tap:menu"]);
  });

  it("drops empty trajectories when assembling the dataset", () => {
    const withActions = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [action("device", "tapped Menu", 1, { gesture: "tap", target: "Menu" })],
    });
    const empty = buildTrajectorySpan({ id: "t2", sessionId: "s1" });
    const dataset = buildMovementDataset([withActions, empty]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]!.id).toBe("t1");
  });

  it("extracts action tokens from a replay manifest", () => {
    const manifest: ReplayManifest = {
      version: 1,
      sessionId: "s1",
      trajectoryIds: ["t1"],
      eventCount: 2,
      events: [
        { kind: "observation", ts: 1, trajectoryId: "t1", source: "device", summary: "app active" },
        { kind: "action", ts: 2, trajectoryId: "t1", tool: "device", summary: "tapped Submit" },
      ],
    };
    expect(movementSequenceFromReplay(manifest).tokens).toEqual(["device:tapped"]);
  });
});

describe("MovementBackendRegistry", () => {
  it("registers the deterministic backend by default and resolves it by id", () => {
    const registry = new MovementBackendRegistry();
    expect(registry.list()).toEqual(["deterministic-markov"]);
    expect(registry.get("deterministic-markov").id).toBe("deterministic-markov");
  });

  it("allows a custom backend to be swapped in and throws on unknown ids", () => {
    const stub: LocalModelBackend = {
      id: "stub",
      train: () => ({
        backend: "stub",
        version: 1,
        order: 0,
        vocabulary: [],
        transitions: {},
        sequenceCount: 0,
        exampleCount: 0,
      }),
      predict: () => ({ action: undefined, confidence: 0, order: 0, generalized: false, candidates: [] }),
    };
    const registry = new MovementBackendRegistry([stub]);
    expect(registry.has("stub")).toBe(true);
    expect(() => registry.get("deterministic-markov")).toThrow(/unknown movement-model backend/);
  });
});

describe("evaluateGeneralization", () => {
  it("scores replay accuracy and credits generalized (backed-off) hits", () => {
    const backend = new DeterministicMarkovBackend();
    const model = backend.train(datasetOf(["home", "open", "menu"], ["settings", "open", "menu"]), { order: 2 });

    // Held-out but related: a fresh prefix leading into the learned "open -> menu".
    const report = evaluateGeneralization(backend, model, [{ id: "held", tokens: ["profile", "open", "menu"] }]);
    expect(report.total).toBe(3);
    // "open" (from unigram prior) and "menu" (backed off to "open -> menu") are correct.
    expect(report.correct).toBeGreaterThanOrEqual(1);
    expect(report.generalizedCorrect).toBeGreaterThanOrEqual(1);
    expect(report.accuracy).toBeCloseTo(report.correct / report.total);
    expect(report.generalizationRate).toBeCloseTo(report.generalizedCorrect / report.correct);
  });

  it("reports zeroed metrics for an empty held-out set", () => {
    const backend = new DeterministicMarkovBackend();
    const model = backend.train(datasetOf(["a", "b"]), { order: 1 });
    expect(evaluateGeneralization(backend, model, [])).toEqual({
      total: 0,
      correct: 0,
      generalizedCorrect: 0,
      abstained: 0,
      accuracy: 0,
      generalizationRate: 0,
    });
  });
});
