import { describe, expect, it } from "vitest";
import { buildTrajectorySpan, type TrajectoryAction, type TrajectorySpan } from "../capture/trajectory.js";
import {
  NgramMovementPolicy,
  evaluateMovementPolicy,
  extractMovementSequence,
  generateMovementSequence,
  sequenceFromActions,
  tokenizeMovementAction,
  trainMovementPolicy,
  type MovementSequence,
} from "./movement-policy.js";

function action(tool: string, summary: string, ts: number, metadata?: Record<string, unknown>): TrajectoryAction {
  return { kind: "action", tool, summary, ts, ...(metadata ? { metadata } : {}) };
}

function gesture(kind: string, direction: string | undefined, ts: number): TrajectoryAction {
  return action("device", `${kind} ${direction ?? ""}`.trim(), ts, {
    gesture: kind,
    ...(direction ? { direction } : {}),
  });
}

function spanFrom(id: string, actions: TrajectoryAction[]): TrajectorySpan {
  return buildTrajectorySpan({ id, sessionId: "s1", captureTier: "full", actions });
}

describe("tokenizeMovementAction", () => {
  it("prefers structured gesture metadata over free-text summary", () => {
    expect(tokenizeMovementAction(gesture("swipe", "down", 1))).toBe("device:swipe:down");
    expect(tokenizeMovementAction(gesture("tap", undefined, 1))).toBe("device:tap");
  });

  it("uses os event metadata when present, else slugs the summary", () => {
    expect(
      tokenizeMovementAction({ tool: "os", summary: "focused Editor", ts: 1, metadata: { event: "focus-changed" } }),
    ).toBe("os:focus-changed");
    expect(tokenizeMovementAction({ tool: "cli", summary: "Ran git commit -m x", ts: 1 })).toBe("cli:ran-git-commit-m-x");
  });

  it("collapses semantically identical movements to the same token", () => {
    const a = tokenizeMovementAction(gesture("scroll", "up", 5));
    const b = tokenizeMovementAction(gesture("scroll", "up", 900));
    expect(a).toBe(b);
  });
});

describe("sequence extraction", () => {
  it("orders actions by timestamp", () => {
    const seq = sequenceFromActions("t1", [gesture("tap", undefined, 30), gesture("swipe", "up", 10)]);
    expect(seq.tokens).toEqual(["device:swipe:up", "device:tap"]);
  });

  it("uses reviewed redacted actions when requested", () => {
    const span: TrajectorySpan = {
      ...spanFrom("t1", [gesture("swipe", "down", 1)]),
      review: {
        status: "approved",
        reviewedAt: "2026-01-01T00:00:00Z",
        reviewedBy: "user",
        redactedActions: [{ ts: 1, tool: "device", summary: "tapped Send" }],
      },
    };
    expect(extractMovementSequence(span, { useReviewed: true }).tokens).toEqual(["device:tapped-send"]);
    expect(extractMovementSequence(span).tokens).toEqual(["device:swipe:down"]);
  });
});

describe("NgramMovementPolicy — repetition", () => {
  it("reproduces a recorded sequence exactly from its own prefixes", () => {
    const seq: MovementSequence = { trajectoryId: "t1", tokens: ["a", "b", "c", "d"] };
    const policy = new NgramMovementPolicy(3);
    policy.train([seq]);

    expect(policy.predictNext(["a"])).toMatchObject({ token: "b", source: "exact", confidence: 1 });
    expect(policy.predictNext(["a", "b"])).toMatchObject({ token: "c", source: "exact" });
    expect(generateMovementSequence(policy, ["a"], { length: 3 })).toEqual(["b", "c", "d"]);
  });
});

describe("NgramMovementPolicy — generalization", () => {
  it("backs off to a shorter learned suffix for unseen full context", () => {
    const policy = new NgramMovementPolicy(3);
    // "open -> type -> save" seen; "search -> type -> ?" is novel but shares "type".
    policy.train([
      { trajectoryId: "t1", tokens: ["open", "type", "save"] },
      { trajectoryId: "t2", tokens: ["click", "type", "save"] },
    ]);
    const prediction = policy.predictNext(["search", "type"]);
    expect(prediction.token).toBe("save");
    expect(prediction.source).toBe("backoff");
    expect(prediction.order).toBe(1);
  });

  it("falls back to the unigram prior when no context matches", () => {
    const policy = new NgramMovementPolicy(2);
    policy.train([{ trajectoryId: "t1", tokens: ["x", "x", "x", "y"] }]);
    const prediction = policy.predictNext(["totally-unseen"]);
    expect(prediction.source).toBe("prior");
    expect(prediction.token).toBe("x"); // most frequent overall
  });

  it("returns an empty prediction when untrained", () => {
    const policy = new NgramMovementPolicy();
    expect(policy.predictNext(["a"])).toMatchObject({ token: undefined, source: "empty", confidence: 0 });
  });

  it("breaks ties deterministically (lexicographic)", () => {
    const policy = new NgramMovementPolicy(1);
    policy.train([
      { trajectoryId: "t1", tokens: ["ctx", "zebra"] },
      { trajectoryId: "t2", tokens: ["ctx", "alpha"] },
    ]);
    const prediction = policy.predictNext(["ctx"]);
    expect(prediction.token).toBe("alpha");
    expect(prediction.confidence).toBeCloseTo(0.5);
    expect(prediction.alternatives.map((c) => c.token)).toEqual(["alpha", "zebra"]);
  });
});

describe("trainMovementPolicy + evaluateMovementPolicy", () => {
  it("scores high top-1 accuracy on the training distribution", () => {
    const trajectories = [
      spanFrom("t1", [gesture("swipe", "down", 1), gesture("tap", undefined, 2), gesture("type", undefined, 3)]),
      spanFrom("t2", [gesture("swipe", "down", 1), gesture("tap", undefined, 2), gesture("type", undefined, 3)]),
    ];
    const policy = trainMovementPolicy(trajectories, { maxOrder: 3 });
    const testSeqs = trajectories.map((t) => extractMovementSequence(t));
    const result = evaluateMovementPolicy(policy, testSeqs);
    expect(result.totalPredictions).toBe(4); // 2 sequences * 2 predictable positions
    expect(result.accuracy).toBe(1);
    expect(result.bySource.exact.correct).toBeGreaterThan(0);
  });

  it("measures generalization on held-out related sequences and supports top-k", () => {
    const train: MovementSequence[] = [
      { trajectoryId: "t1", tokens: ["nav", "select", "confirm"] },
      { trajectoryId: "t2", tokens: ["nav", "select", "cancel"] },
    ];
    const policy = new NgramMovementPolicy(2);
    policy.train(train);
    // Held-out: same "select ->" transition, previously-unseen prefix "back".
    const heldOut: MovementSequence[] = [{ trajectoryId: "h1", tokens: ["back", "select", "confirm"] }];
    const top1 = evaluateMovementPolicy(policy, heldOut, { minContext: 2 });
    const top2 = evaluateMovementPolicy(policy, heldOut, { k: 2, minContext: 2 });
    expect(top1.totalPredictions).toBe(1);
    expect(top2.topKAccuracy).toBe(1); // confirm is among the two learned continuations
    expect(top2.k).toBe(2);
  });
});

describe("serialization", () => {
  it("round-trips a trained model to JSON and back", () => {
    const policy = new NgramMovementPolicy(2);
    policy.train([{ trajectoryId: "t1", tokens: ["a", "b", "c"] }]);
    const restored = NgramMovementPolicy.fromJSON(policy.toJSON());
    expect(restored.maxOrder).toBe(2);
    expect(restored.vocabulary()).toEqual(policy.vocabulary());
    expect(restored.predictNext(["a", "b"])).toMatchObject({ token: "c", source: "exact" });
  });

  it("supports incremental online observation", () => {
    const policy = new NgramMovementPolicy(1);
    policy.observe(["p", "q"]);
    policy.observe(["p", "q"]);
    expect(policy.predictNext(["p"]).token).toBe("q");
  });
});
