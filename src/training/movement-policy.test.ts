import { describe, expect, it } from "vitest";
import type { ExportedReplayManifest } from "./export-manifest.js";
import {
  MovementPolicyEngine,
  NearestNeighborMovementBackend,
  contextTokens,
  extractTransitions,
  generalizeSummary,
  jaccardSimilarity,
  tokenize,
  type MovementModelBackend,
  type MovementContext,
  type MovementTransition,
  type PredictedMovement,
} from "./movement-policy.js";

function replay(events: ExportedReplayManifest["events"]): ExportedReplayManifest {
  return {
    sessionId: "s1",
    trajectoryIds: [...new Set(events.filter((e) => e.kind !== "transcript").map((e) => (e as { trajectoryId: string }).trajectoryId))],
    eventCount: events.length,
    events,
  };
}

const CHECKOUT_REPLAY = replay([
  { kind: "observation", ts: 1, trajectoryId: "t1", source: "device", summary: "Checkout on Cart" },
  { kind: "action", ts: 2, trajectoryId: "t1", tool: "device", summary: "tapped Checkout button" },
  { kind: "observation", ts: 3, trajectoryId: "t1", source: "device", summary: "Checkout on Payment" },
  { kind: "action", ts: 4, trajectoryId: "t1", tool: "device", summary: "typed into card field" },
]);

describe("tokenize", () => {
  it("lowercases, splits, drops stop-words and short tokens", () => {
    expect(tokenize("CheckoutApp on Cart screen")).toEqual(["checkoutapp", "cart", "screen"]);
  });
});

describe("jaccardSimilarity", () => {
  it("is 1 for identical token sets and 0 for disjoint", () => {
    expect(jaccardSimilarity(["a", "b"], ["a", "b"])).toBe(1);
    expect(jaccardSimilarity(["a"], ["b"])).toBe(0);
  });

  it("is order-independent and deduplicates", () => {
    expect(jaccardSimilarity(["a", "b", "a"], ["b", "a"])).toBe(1);
  });
});

describe("contextTokens", () => {
  it("merges recent summaries, goal and focus into a deduped token set", () => {
    const tokens = contextTokens({ recentSummaries: ["tapped Checkout button"], goal: "buy item", focus: "CheckoutApp" });
    expect(tokens).toContain("checkout");
    expect(tokens).toContain("buy");
    expect(tokens).toContain("checkoutapp");
    expect(new Set(tokens).size).toBe(tokens.length);
  });
});

describe("extractTransitions", () => {
  it("mines one transition per action with its preceding-window context and focus", () => {
    const transitions = extractTransitions({ replays: [CHECKOUT_REPLAY] });
    expect(transitions).toHaveLength(2);
    expect(transitions[0].action).toEqual({ tool: "device", summary: "tapped Checkout button" });
    expect(transitions[0].focus).toBe("Checkout on Cart");
    expect(transitions[0].contextTokens).toContain("checkout");
    expect(transitions[1].action.summary).toBe("typed into card field");
  });

  it("respects the window size", () => {
    const transitions = extractTransitions({ replays: [CHECKOUT_REPLAY] }, { windowSize: 1 });
    // Second action's window is only the immediately preceding observation.
    expect(transitions[1].contextTokens).not.toContain("cart");
  });
});

describe("generalizeSummary", () => {
  it("rewrites the recalled focus keyword to the live focus keyword", () => {
    expect(generalizeSummary("tapped Checkout submit", "Checkout screen", "Settings screen")).toBe("tapped settings submit");
  });

  it("returns the summary unchanged when focuses share a keyword", () => {
    expect(generalizeSummary("tapped Checkout button", "Checkout", "Checkout elsewhere")).toBe("tapped Checkout button");
  });

  it("returns the summary unchanged when a focus is missing", () => {
    expect(generalizeSummary("tapped Checkout button", undefined, "Settings")).toBe("tapped Checkout button");
  });
});

describe("NearestNeighborMovementBackend", () => {
  it("returns undefined before training / with no data", () => {
    const backend = new NearestNeighborMovementBackend();
    expect(backend.predict({ recentSummaries: ["anything"] })).toBeUndefined();
  });

  it("recalls the exact recorded action for a matching context", () => {
    const backend = new NearestNeighborMovementBackend();
    backend.train(extractTransitions({ replays: [CHECKOUT_REPLAY] }));
    const prediction = backend.predict({ recentSummaries: ["Checkout on Cart"], focus: "Checkout on Cart" });
    expect(prediction?.summary).toBe("tapped Checkout button");
    expect(prediction?.source).toBe("recall");
    expect(prediction?.confidence).toBeGreaterThan(0);
    expect(prediction?.basisTrajectoryId).toBe("t1");
  });

  it("generalizes a recalled action to a new-but-related focus", () => {
    const backend = new NearestNeighborMovementBackend();
    backend.train(extractTransitions({ replays: [CHECKOUT_REPLAY] }));
    const prediction = backend.predict({
      recentSummaries: ["Settings on Cart"],
      focus: "Settings on Cart",
    });
    expect(prediction?.source).toBe("generalized");
    expect(prediction?.summary).toContain("settings");
  });

  it("honors the minConfidence threshold", () => {
    const backend = new NearestNeighborMovementBackend({ minConfidence: 0.99 });
    backend.train(extractTransitions({ replays: [CHECKOUT_REPLAY] }));
    expect(backend.predict({ recentSummaries: ["totally unrelated words here"] })).toBeUndefined();
  });

  it("is deterministic across repeated predictions", () => {
    const backend = new NearestNeighborMovementBackend();
    backend.train(extractTransitions({ replays: [CHECKOUT_REPLAY] }));
    const ctx: MovementContext = { recentSummaries: ["CheckoutApp on Cart screen"] };
    expect(backend.predict(ctx)).toEqual(backend.predict(ctx));
  });
});

describe("MovementPolicyEngine", () => {
  it("trains from an export and reports transition count + backend name", () => {
    const engine = new MovementPolicyEngine();
    engine.trainFromExport({ replays: [CHECKOUT_REPLAY] });
    expect(engine.backendName).toBe("nearest-neighbor");
    expect(engine.learnedTransitionCount).toBe(2);
  });

  it("throws if asked to predict before training", () => {
    const engine = new MovementPolicyEngine();
    expect(() => engine.predictNext({ recentSummaries: [] })).toThrow(/before training/);
  });

  it("falls back when the backend cannot answer", () => {
    const engine = new MovementPolicyEngine({ fallback: { tool: "noop", summary: "idle", confidence: 0, source: "fallback" } });
    engine.trainFromTransitions([]);
    expect(engine.predictNext({ recentSummaries: ["x"] })).toEqual({
      tool: "noop",
      summary: "idle",
      confidence: 0,
      source: "fallback",
    });
  });

  it("rolls out an autoregressive movement sequence", () => {
    const engine = new MovementPolicyEngine();
    engine.trainFromExport({ replays: [CHECKOUT_REPLAY] });
    const sequence = engine.predictSequence(
      { recentSummaries: ["CheckoutApp on Cart screen"] },
      { maxSteps: 3 },
    );
    expect(sequence.length).toBeGreaterThan(0);
    expect(sequence.every((step) => step.source !== "fallback")).toBe(true);
  });

  it("accepts a pluggable custom backend (real-model seam)", () => {
    const calls: MovementTransition[][] = [];
    const custom: MovementModelBackend = {
      name: "stub-onnx",
      train(transitions) {
        calls.push([...transitions]);
      },
      predict(): PredictedMovement {
        return { tool: "device", summary: "stub action", confidence: 0.5, source: "recall" };
      },
    };
    const engine = new MovementPolicyEngine({ backend: custom });
    engine.trainFromExport({ replays: [CHECKOUT_REPLAY] });
    expect(engine.backendName).toBe("stub-onnx");
    expect(calls[0]).toHaveLength(2);
    expect(engine.predictNext({ recentSummaries: ["x"] }).summary).toBe("stub action");
  });
});
