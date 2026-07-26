import { describe, expect, it } from "vitest";
import type { ReplayManifest } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  MOVEMENT_END_TOKEN,
  NgramMovementModelBackend,
  actionEventToToken,
  evaluateMovementModel,
  extractMovementSamples,
  generateSyntheticMovementSamples,
  replayToMovementSample,
  rolloutMovements,
  trajectoryToMovementSample,
  type MovementSample,
} from "./movement-policy.js";

const backend = new NgramMovementModelBackend();

function sample(...tokens: string[]): MovementSample {
  return { tokens };
}

describe("movement token extraction", () => {
  it("tokenizes an action event with a coarse summary bucket", () => {
    expect(
      actionEventToToken({ kind: "action", ts: 1, trajectoryId: "t", tool: "mouse.move", summary: "Move to (12, 40)" }),
    ).toBe("action:mouse.move:move");
    expect(
      actionEventToToken({ kind: "action", ts: 1, trajectoryId: "t", tool: "key.press", summary: "" }),
    ).toBe("action:key.press");
  });

  it("extracts ordered action tokens from a replay manifest, ignoring non-action events", () => {
    const replay: ReplayManifest = {
      version: 1,
      sessionId: "s1",
      trajectoryIds: ["t1"],
      eventCount: 3,
      events: [
        { kind: "transcript", ts: 1, messageId: "m", role: "user", content: "go" },
        { kind: "action", ts: 2, trajectoryId: "t1", tool: "mouse.click", summary: "menu" },
        { kind: "observation", ts: 3, trajectoryId: "t1", source: "ui", summary: "open" },
        { kind: "action", ts: 4, trajectoryId: "t1", tool: "key.press", summary: "save" },
      ],
    };
    expect(replayToMovementSample(replay).tokens).toEqual(["action:mouse.click:menu", "action:key.press:save"]);
    expect(extractMovementSamples([replay, { ...replay, events: [] }])).toHaveLength(1);
  });

  it("extracts tokens from a trajectory span sorted by timestamp", () => {
    const trajectory: TrajectorySpan = {
      id: "t9",
      sessionId: "s9",
      createdAt: "2026-01-01T00:00:00.000Z",
      captureTier: "full",
      observations: [],
      actions: [
        { kind: "action", ts: 20, tool: "key.press", summary: "save" },
        { kind: "action", ts: 10, tool: "mouse.click", summary: "menu" },
      ],
    };
    expect(trajectoryToMovementSample(trajectory).tokens).toEqual(["action:mouse.click:menu", "action:key.press:save"]);
  });
});

describe("n-gram backend — repeats recorded movements (objective c)", () => {
  it("reproduces a recorded sequence exactly via rollout", () => {
    const recorded = sample("action:a", "action:b", "action:c", "action:d");
    const model = backend.train([recorded], { order: 3 });
    const rollout = rolloutMovements(backend, model, { maxSteps: 16 });
    expect(rollout.tokens).toEqual(recorded.tokens);
    expect(rollout.stoppedReason).toBe("end-token");
  });

  it("predicts the exact next token for a seen context with full confidence", () => {
    const model = backend.train([sample("action:a", "action:b", "action:c")], { order: 2 });
    const prediction = backend.predict(model, { history: ["action:a", "action:b"] });
    expect(prediction.token).toBe("action:c");
    expect(prediction.source).toBe("exact");
    expect(prediction.confidence).toBe(1);
  });
});

describe("n-gram backend — generalizes to new-but-related movements (objective d)", () => {
  it("emits a shared continuation for an unseen context via backoff", () => {
    // Two workflows that both end ...x -> save. A novel prefix ending in "x"
    // should still generalize to "save" even though this exact context was
    // never recorded.
    const model = backend.train(
      [
        sample("action:open", "action:type", "action:x", "action:save"),
        sample("action:launch", "action:edit", "action:x", "action:save"),
      ],
      { order: 2 },
    );
    const prediction = backend.predict(model, { history: ["action:brandnew", "action:x"] });
    expect(prediction.token).toBe("action:save");
    expect(prediction.source).toBe("backoff");
    expect(prediction.matchedOrder).toBeLessThan(2);
  });

  it("falls back to the unigram prior for a fully-unseen context", () => {
    const model = backend.train([sample("action:a", "action:a", "action:b")], { order: 3 });
    const prediction = backend.predict(model, { history: ["action:totally-unseen"] });
    // "action:a" is the most frequent next token overall, so the prior picks it.
    expect(prediction.token).toBe("action:a");
    expect(prediction.source).toBe("prior");
    expect(prediction.matchedOrder).toBe(0);
  });
});

describe("training is deterministic", () => {
  it("produces identical models and rollouts across runs", () => {
    const samples = generateSyntheticMovementSamples({ seed: 42, sampleCount: 20 });
    const a = backend.train(samples, { order: 3 });
    const b = backend.train(samples, { order: 3 });
    expect(JSON.stringify(a)).toBe(JSON.stringify(b));
    expect(a.vocabulary).toEqual([...a.vocabulary].sort());
    expect(rolloutMovements(backend, a).tokens).toEqual(rolloutMovements(backend, b).tokens);
  });
});

describe("synthetic generator", () => {
  it("is reproducible for a given seed and varies across seeds", () => {
    const s1 = generateSyntheticMovementSamples({ seed: 7, sampleCount: 12 });
    const s1again = generateSyntheticMovementSamples({ seed: 7, sampleCount: 12 });
    const s2 = generateSyntheticMovementSamples({ seed: 8, sampleCount: 12 });
    expect(s1).toEqual(s1again);
    expect(s1).not.toEqual(s2);
    expect(s1).toHaveLength(12);
    expect(s1.every((entry) => entry.tokens.length > 0)).toBe(true);
  });

  it("produces variations (novel sequences) not present in the base grammar", () => {
    const samples = generateSyntheticMovementSamples({ seed: 3, sampleCount: 60, variationRate: 1 });
    const lengths = new Set(samples.map((entry) => entry.tokens.length));
    // With variationRate=1 every eligible sample is perturbed, so we should see
    // more than one distinct sequence length (skips/dupes change length).
    expect(lengths.size).toBeGreaterThan(1);
  });
});

describe("generalization eval harness", () => {
  it("scores high on held-out variations drawn from the trained grammar", () => {
    const train = generateSyntheticMovementSamples({ seed: 1, sampleCount: 120 });
    const heldOut = generateSyntheticMovementSamples({ seed: 999, sampleCount: 40 });
    const model = backend.train(train, { order: 3 });
    const result = evaluateMovementModel(backend, model, heldOut);
    expect(result.total).toBeGreaterThan(0);
    // The model has learned the grammar well enough to predict most next moves.
    expect(result.accuracy).toBeGreaterThan(0.6);
    // Some correct predictions must come from backed-off (generalizing) context,
    // proving it isn't merely memorizing exact prefixes.
    expect(result.generalizedCorrect).toBeGreaterThan(0);
    expect(result.meanConfidence).toBeGreaterThan(0);
  });

  it("reports lower accuracy on an unrelated grammar (no false generalization)", () => {
    const train = generateSyntheticMovementSamples({ seed: 5, sampleCount: 80 });
    const model = backend.train(train, { order: 3 });
    const unrelated: MovementSample[] = [
      sample("action:zzz.one", "action:zzz.two", "action:zzz.three"),
      sample("action:qqq.alpha", "action:qqq.beta"),
    ];
    const related = generateSyntheticMovementSamples({ seed: 6, sampleCount: 40 });
    const unrelatedScore = evaluateMovementModel(backend, model, unrelated).accuracy;
    const relatedScore = evaluateMovementModel(backend, model, related).accuracy;
    expect(relatedScore).toBeGreaterThan(unrelatedScore);
  });
});

describe("rollout guards", () => {
  it("stops at maxSteps for a self-looping model", () => {
    const model = backend.train([sample("action:loop", "action:loop", "action:loop")], { order: 1 });
    const rollout = rolloutMovements(backend, model, { maxSteps: 5 });
    expect(rollout.tokens.length).toBeLessThanOrEqual(5);
  });

  it("returns an unknown-stop for an empty model", () => {
    const model = backend.train([], { order: 2 });
    const rollout = rolloutMovements(backend, model, { maxSteps: 4 });
    expect(rollout.steps).toHaveLength(0);
    expect(rollout.stoppedReason).toBe("unknown");
    expect(backend.predict(model, { history: [] }).token).toBe(MOVEMENT_END_TOKEN);
  });
});
