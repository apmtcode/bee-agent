import { describe, expect, it } from "vitest";
import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";
import {
  BackoffNgramMovementBackend,
  buildMovementDataset,
  evaluateNextActionAccuracy,
  MovementModel,
  movementSequenceTokens,
  tokenizeMovementAction,
  type MovementModelBackend,
  type MovementModelState,
  type MovementPrediction,
  type MovementSequence,
} from "./movement-model.js";

function action(tool: string, summary: string, ts: number): TrajectoryAction {
  return { kind: "action", tool, summary, ts };
}

function span(id: string, actions: TrajectoryAction[], overrides: Partial<TrajectorySpan> = {}): TrajectorySpan {
  return {
    id,
    sessionId: `session-${id}`,
    createdAt: "2026-07-18T00:00:00.000Z",
    captureTier: "full",
    observations: [],
    actions,
    ...overrides,
  };
}

function seq(id: string, tokens: string[]): MovementSequence {
  return {
    trajectoryId: id,
    sessionId: `session-${id}`,
    steps: tokens.map((token) => ({ token, tool: token, summary: token })),
  };
}

describe("tokenizeMovementAction", () => {
  it("folds the summary intent into the token by default", () => {
    expect(tokenizeMovementAction({ tool: "type", summary: "Email subject line" }).token).toBe("type:email");
    expect(tokenizeMovementAction({ tool: "Tap", summary: "  Send button " }).token).toBe("tap:send");
  });

  it("omits the intent when disabled and falls back for empty tools", () => {
    expect(tokenizeMovementAction({ tool: "swipe", summary: "up" }, { includeSummaryIntent: false }).token).toBe("swipe");
    expect(tokenizeMovementAction({ tool: "  ", summary: "" }).token).toBe("action");
  });
});

describe("buildMovementDataset", () => {
  it("sorts steps by timestamp and preserves outcome", () => {
    const dataset = buildMovementDataset([
      span("t1", [action("tap", "b", 20), action("tap", "a", 10)], {
        outcome: { status: "success", summary: "ok" },
      }),
    ]);
    expect(dataset).toHaveLength(1);
    expect(movementSequenceTokens(dataset[0]!)).toEqual(["tap:a", "tap:b"]);
    expect(dataset[0]!.outcome).toBe("success");
  });

  it("honours requireApproved and minSteps filters", () => {
    const trajectories = [
      span("approved", [action("tap", "a", 1), action("tap", "b", 2)], {
        review: { status: "approved", reviewedAt: "x", reviewedBy: "y" },
      }),
      span("pending", [action("tap", "a", 1), action("tap", "b", 2)]),
      span("tiny", [action("tap", "a", 1)], {
        review: { status: "approved", reviewedAt: "x", reviewedBy: "y" },
      }),
    ];
    const dataset = buildMovementDataset(trajectories, { requireApproved: true, minSteps: 2 });
    expect(dataset.map((s) => s.trajectoryId)).toEqual(["approved"]);
  });
});

describe("BackoffNgramMovementBackend training + prediction", () => {
  it("predicts the majority next movement deterministically", () => {
    const model = MovementModel.train([
      seq("a", ["open", "type", "send"]),
      seq("b", ["open", "type", "send"]),
      seq("c", ["open", "type", "discard"]),
    ]);
    const prediction = model.predictNext(["open", "type"]);
    expect(prediction.token).toBe("send");
    expect(prediction.order).toBe(2);
    expect(prediction.probability).toBeCloseTo(2 / 3);
    // Deterministic across repeated instantiations.
    const again = MovementModel.train([
      seq("a", ["open", "type", "send"]),
      seq("b", ["open", "type", "send"]),
      seq("c", ["open", "type", "discard"]),
    ]);
    expect(again.predictNext(["open", "type"]).token).toBe("send");
  });

  it("backs off to a shorter observed suffix for an unseen prefix (generalization)", () => {
    const model = MovementModel.train([
      seq("a", ["focus", "select", "copy"]),
      seq("b", ["scroll", "select", "copy"]),
    ]);
    // "click select" was never seen, but "select -> copy" was: back off to order 1.
    const prediction = model.predictNext(["click", "select"]);
    expect(prediction.token).toBe("copy");
    expect(prediction.order).toBe(1);
  });

  it("returns a null token only for an empty model", () => {
    const empty = MovementModel.train([]);
    expect(empty.predictNext(["anything"]).token).toBeNull();
    expect(empty.vocabulary).toEqual([]);
  });

  it("exposes a sorted vocabulary and step counts in the state", () => {
    const model = MovementModel.train([seq("a", ["b", "a", "c"])]);
    expect(model.vocabulary).toEqual(["a", "b", "c"]);
    expect(model.modelState.stepCount).toBe(3);
    expect(model.modelState.sequenceCount).toBe(1);
    expect(model.modelState.backend).toBe("backoff-ngram");
  });
});

describe("MovementModel.generate", () => {
  it("rolls a recorded pattern forward autoregressively", () => {
    const model = MovementModel.train([
      seq("a", ["open", "type", "send", "close"]),
      seq("b", ["open", "type", "send", "close"]),
    ]);
    expect(model.generate(["open"], 3)).toEqual(["type", "send", "close"]);
  });

  it("stops early when a prediction repeats past maxRepeat", () => {
    const model = MovementModel.train([seq("loop", ["tick", "tick", "tick", "tick"])]);
    const generated = model.generate(["tick"], 10, { maxRepeat: 2 });
    expect(generated.length).toBeLessThan(10);
    expect(generated.every((token) => token === "tick")).toBe(true);
  });
});

describe("evaluateNextActionAccuracy", () => {
  it("scores generalization on held-out related sequences", () => {
    const model = MovementModel.train([
      seq("train1", ["open", "search", "select", "open-item"]),
      seq("train2", ["open", "search", "select", "open-item"]),
    ]);
    // Held-out sequence shares the learned transitions but is a fresh trajectory.
    const result = evaluateNextActionAccuracy(model, [seq("holdout", ["open", "search", "select", "open-item"])]);
    expect(result.predictions).toBe(3);
    expect(result.correct).toBe(3);
    expect(result.accuracy).toBe(1);
    expect(result.averageBackoffOrder).toBeGreaterThan(0);
  });

  it("reports zeroed metrics for an empty held-out set", () => {
    const model = MovementModel.train([seq("a", ["x", "y"])]);
    expect(evaluateNextActionAccuracy(model, [])).toEqual({
      sequences: 0,
      predictions: 0,
      correct: 0,
      accuracy: 0,
      averageBackoffOrder: 0,
    });
  });
});

describe("pluggable backend", () => {
  it("accepts a custom MovementModelBackend implementation", () => {
    const constantBackend: MovementModelBackend = {
      name: "constant",
      order: 1,
      train(dataset): MovementModelState {
        return {
          backend: "constant",
          order: 1,
          vocabulary: ["always"],
          sequenceCount: dataset.length,
          stepCount: 0,
          payload: null,
        };
      },
      predict(): MovementPrediction {
        return { token: "always", probability: 1, order: 0, candidates: [{ token: "always", probability: 1, count: 1 }] };
      },
    };
    const model = MovementModel.train([seq("a", ["ignored"])], constantBackend);
    expect(model.predictNext(["anything", "here"]).token).toBe("always");
    expect(model.generate([], 3)).toEqual(["always", "always", "always"]);
  });

  it("respects a configurable n-gram order", () => {
    const backend = new BackoffNgramMovementBackend(2);
    const model = MovementModel.train([seq("a", ["p", "q", "r", "s"])], backend);
    expect(model.modelState.order).toBe(2);
    // Order 2 only conditions on the single previous token.
    expect(model.predictNext(["p", "q"]).order).toBeLessThanOrEqual(1);
  });
});
