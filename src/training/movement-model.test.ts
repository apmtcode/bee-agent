import { describe, expect, it } from "vitest";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import {
  generateSyntheticTrajectories,
  DEFAULT_SYNTHETIC_WORKFLOWS,
} from "../capture/synthetic.js";
import {
  createMovementBackend,
  deserializeMovementModel,
  evaluateMovementModel,
  MOVEMENT_END_TOKEN,
  NGramMovementBackend,
  registerMovementBackend,
  tokenizeAction,
  tokenizeTrajectory,
  type MovementModelBackend,
  type MovementSequence,
  type TrainedMovementModel,
} from "./movement-model.js";

function seq(...tokens: string[]): MovementSequence {
  return { tokens };
}

describe("tokenizeAction / tokenizeTrajectory", () => {
  it("prefers structured gesture metadata and slugifies parts", () => {
    expect(
      tokenizeAction({
        kind: "action",
        tool: "device",
        summary: "tapped Send Button",
        ts: 1,
        metadata: { gesture: "tap", target: "Send Button" },
      }),
    ).toBe("device:tap:send-button");
  });

  it("falls back to the summary when no gesture metadata is present", () => {
    expect(
      tokenizeAction({ kind: "action", tool: "browser", summary: "Clicked Deploy!", ts: 1 }),
    ).toBe("browser:clicked-deploy");
  });

  it("orders trajectory actions by timestamp", () => {
    const span = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [
        { kind: "action", tool: "device", summary: "b", ts: 20, metadata: { gesture: "tap", target: "b" } },
        { kind: "action", tool: "device", summary: "a", ts: 10, metadata: { gesture: "tap", target: "a" } },
      ],
    });
    expect(tokenizeTrajectory(span)).toEqual({
      sourceTrajectoryId: "t1",
      tokens: ["device:tap:a", "device:tap:b"],
    });
  });
});

describe("NGramMovementBackend", () => {
  it("repeats a recorded movement sequence exactly (objective 2c)", () => {
    const model = createMovementBackend("ngram").train([seq("open", "type", "save", "close")]);
    // Given the recorded first move, it reproduces the recorded continuation.
    expect(model.generate(["open"], 10)).toEqual(["type", "save", "close"]);
  });

  it("terminates generation at the learned end sentinel", () => {
    const model = new NGramMovementBackend().train([seq("a", "b")]);
    const prediction = model.predictNext(["a", "b"]);
    expect(prediction?.token).toBe(MOVEMENT_END_TOKEN);
    expect(model.generate(["a"], 10)).toEqual(["b"]);
  });

  it("is deterministic under ties (lexical tie-break)", () => {
    // "start" is followed once by "zebra" and once by "alpha" -> tie -> "alpha".
    const model = new NGramMovementBackend().train([seq("start", "zebra"), seq("start", "alpha")]);
    expect(model.predictNext(["start"])?.token).toBe("alpha");
  });

  it("generalizes to an unseen context by backing off to a seen suffix (objective 2d)", () => {
    const model = new NGramMovementBackend().train(
      [seq("home", "search", "result", "open"), seq("menu", "search", "result", "open")],
      { order: 3 },
    );
    // The full trigram context ["a","b","search"] was never seen, but the
    // unigram/bigram suffix "search" -> "result" was; back-off predicts it.
    const prediction = model.predictNext(["a", "b", "search"]);
    expect(prediction?.token).toBe("result");
    expect(prediction?.order).toBe(1);
  });

  it("returns undefined for an empty, untrained model", () => {
    const model = new NGramMovementBackend().train([]);
    expect(model.predictNext(["anything"])).toBeUndefined();
    expect(model.generate([], 5)).toEqual([]);
  });

  it("round-trips through serialization", () => {
    const original = new NGramMovementBackend().train([seq("a", "b", "c")], { order: 2 });
    const restored = deserializeMovementModel(original.serialize());
    expect(restored.generate(["a"], 5)).toEqual(original.generate(["a"], 5));
    expect(restored.order).toBe(2);
    expect(restored.serialize()).toEqual(original.serialize());
  });
});

describe("backend registry", () => {
  it("resolves the default ngram backend", () => {
    expect(createMovementBackend().name).toBe("ngram");
  });

  it("throws for an unknown backend", () => {
    expect(() => createMovementBackend("does-not-exist")).toThrow(/unknown movement backend/);
  });

  it("allows a custom backend to be registered and resolved (pluggable seam)", () => {
    const stub: MovementModelBackend = {
      name: "stub",
      train(): TrainedMovementModel {
        return {
          backend: "stub",
          order: 0,
          predictNext: () => ({ token: "noop", order: 0, probability: 1 }),
          generate: () => ["noop"],
          serialize: () => ({ version: 1, backend: "stub", order: 0, vocabulary: [], transitions: {} }),
        };
      },
    };
    registerMovementBackend(stub);
    expect(createMovementBackend("stub")).toBe(stub);
  });
});

describe("evaluateMovementModel", () => {
  it("reproduces every recorded continuation given a real prefix", () => {
    const training = [seq("a", "b", "c"), seq("a", "b", "d")];
    const model = new NGramMovementBackend().train(training, { order: 3 });
    // Position 0 is a cold start (empty-context unigram), but every subsequent
    // recorded next token is predicted from its true prefix.
    const tokens = ["a", "b", "c", MOVEMENT_END_TOKEN];
    for (let i = 1; i < tokens.length; i += 1) {
      expect(model.predictNext(tokens.slice(0, i))?.token).toBe(tokens[i]);
    }
    const result = evaluateMovementModel(model, [seq("a", "b", "c")]);
    // Only the cold-start position can miss, so fidelity is high.
    expect(result.accuracy).toBeGreaterThanOrEqual(0.75);
  });

  it("generalizes above chance on held-out synthetic variants", () => {
    const train = generateSyntheticTrajectories({ seed: 7, count: 40 }).map(tokenizeTrajectory);
    const heldOut = generateSyntheticTrajectories({ seed: 99, count: 12, variationRate: 0.2 }).map(
      tokenizeTrajectory,
    );
    const model = new NGramMovementBackend().train(train, { order: 3 });
    const result = evaluateMovementModel(model, heldOut);
    // Random next-token accuracy would be ~1/vocab; the model should far exceed it.
    expect(result.predictions).toBeGreaterThan(0);
    expect(result.accuracy).toBeGreaterThan(0.6);
  });
});

describe("synthetic generator", () => {
  it("is deterministic for a fixed seed and count", () => {
    const a = generateSyntheticTrajectories({ seed: 3, count: 5 });
    const b = generateSyntheticTrajectories({ seed: 3, count: 5 });
    expect(a.map(tokenizeTrajectory)).toEqual(b.map(tokenizeTrajectory));
  });

  it("produces different streams for different seeds", () => {
    const a = generateSyntheticTrajectories({ seed: 1, count: 8 }).map(tokenizeTrajectory);
    const b = generateSyntheticTrajectories({ seed: 2, count: 8 }).map(tokenizeTrajectory);
    expect(a).not.toEqual(b);
  });

  it("always yields non-empty spans even under aggressive variation", () => {
    const spans = generateSyntheticTrajectories({ seed: 5, count: 20, variationRate: 0.9 });
    expect(spans).toHaveLength(20);
    for (const span of spans) {
      expect(span.actions.length).toBeGreaterThan(0);
    }
  });

  it("only emits tokens drawn from the workflow vocabulary", () => {
    const validTokens = new Set(
      DEFAULT_SYNTHETIC_WORKFLOWS.flatMap((workflow) =>
        workflow.steps.map((step) =>
          ["device", step.gesture, step.target ?? step.direction]
            .filter(Boolean)
            .map((part) => String(part).toLowerCase().replace(/[^a-z0-9]+/g, "-"))
            .join(":"),
        ),
      ),
    );
    const sequences = generateSyntheticTrajectories({ seed: 4, count: 15 }).map(tokenizeTrajectory);
    for (const sequence of sequences) {
      for (const token of sequence.tokens) {
        expect(validTokens.has(token)).toBe(true);
      }
    }
  });
});
