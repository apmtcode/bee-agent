import { describe, expect, it } from "vitest";
import type { ReplayTimelineEvent } from "../capture/replay.js";
import {
  DEFAULT_MOVEMENT_MODEL_CONFIG,
  MOVEMENT_BOS,
  MOVEMENT_EOS,
  MarkovMovementBackend,
  evaluateMovementModel,
  movementTokenForEvent,
  tokenizeEvents,
  trainMovementModel,
  type MovementModelBackend,
  type MovementSequence,
  type TrainedMovementModel,
} from "./movement-model.js";
import {
  generateSyntheticTrajectories,
  syntheticReplayManifest,
} from "./movement-synthetic.js";

function seq(...tokens: string[]): MovementSequence {
  return { tokens };
}

describe("tokenizeEvents", () => {
  it("maps timeline events to discrete movement tokens with boundaries", () => {
    const events: ReplayTimelineEvent[] = [
      { kind: "observation", ts: 1, trajectoryId: "t", source: "Screen", summary: "x" },
      { kind: "action", ts: 2, trajectoryId: "t", tool: "Mouse Click", summary: "y" },
      { kind: "transcript", ts: 3, messageId: "m", role: "assistant", content: "z" },
    ];
    expect(tokenizeEvents(events)).toEqual([
      MOVEMENT_BOS,
      "observation:screen",
      "action:mouse-click",
      "transcript:assistant",
      MOVEMENT_EOS,
    ]);
  });

  it("normalizes and falls back on empty features", () => {
    expect(movementTokenForEvent({ kind: "action", ts: 0, trajectoryId: "t", tool: "  ", summary: "" })).toBe(
      "action:unknown",
    );
  });
});

describe("MarkovMovementModel — repeat recorded movements", () => {
  it("greedily reproduces a memorised sequence (replay fidelity)", () => {
    const recorded = seq("action:a", "action:b", "action:c", "action:d");
    const model = trainMovementModel([recorded], { order: 2 });
    const generated = model.generate([MOVEMENT_BOS], { maxSteps: 10 });
    expect(generated).toEqual(["action:a", "action:b", "action:c", "action:d"]);
  });

  it("predictNext returns the observed successor at full context order", () => {
    const model = trainMovementModel([seq("action:a", "action:b", "action:c")], { order: 2 });
    const prediction = model.predictNext(["action:a", "action:b"]);
    expect(prediction.token).toBe("action:c");
    expect(prediction.backoffOrder).toBe(2);
    expect(prediction.probability).toBeGreaterThan(0.5);
  });

  it("stops generation at <eos> and never emits boundary tokens", () => {
    const model = trainMovementModel([seq("action:only")], { order: 2 });
    const generated = model.generate([MOVEMENT_BOS], { maxSteps: 25 });
    expect(generated).toEqual(["action:only"]);
    expect(generated).not.toContain(MOVEMENT_BOS);
    expect(generated).not.toContain(MOVEMENT_EOS);
  });
});

describe("MarkovMovementModel — generalize to new-but-related movements", () => {
  it("backs off to a shorter context when the full context is unseen", () => {
    // Train two sequences that share the successor of "action:b".
    const model = trainMovementModel(
      [seq("action:x", "action:b", "action:c"), seq("action:y", "action:b", "action:c")],
      { order: 2 },
    );
    // Context ("action:z","action:b") was never seen at order-2, but order-1
    // "action:b" -> "action:c" was. Backoff should still predict "action:c".
    const prediction = model.predictNext(["action:z", "action:b"]);
    expect(prediction.token).toBe("action:c");
    expect(prediction.backoffOrder).toBe(1);
  });

  it("falls back to the unigram prior for a wholly unseen context", () => {
    const model = trainMovementModel([seq("action:a", "action:a", "action:a", "action:b")], { order: 2 });
    const prediction = model.predictNext(["action:unseen"]);
    expect(prediction.backoffOrder).toBe(0);
    // "action:a" is the most frequent token overall.
    expect(prediction.token).toBe("action:a");
  });
});

describe("determinism", () => {
  it("produces identical output for identical input (greedy)", () => {
    const dataset = [seq("action:a", "action:b"), seq("action:a", "action:c")];
    const a = trainMovementModel(dataset).generate([MOVEMENT_BOS], { maxSteps: 5 });
    const b = trainMovementModel(dataset).generate([MOVEMENT_BOS], { maxSteps: 5 });
    expect(a).toEqual(b);
  });

  it("seeded sampling is reproducible", () => {
    const dataset = [seq("action:a", "action:b", "action:c"), seq("action:a", "action:c", "action:b")];
    const model = trainMovementModel(dataset);
    const opts = { maxSteps: 6, temperature: 1, seedValue: 42 } as const;
    expect(model.generate([MOVEMENT_BOS], opts)).toEqual(model.generate([MOVEMENT_BOS], opts));
  });
});

describe("pluggable backend seam", () => {
  it("accepts a custom backend implementation", () => {
    const constant: MovementModelBackend = {
      name: "constant",
      train(): TrainedMovementModel {
        return {
          stats: { backend: "constant", order: 0, vocabulary: 1, sequences: 0, observedTokens: 0 },
          predictNext: () => ({ token: "action:fixed", probability: 1, distribution: [], backoffOrder: 0 }),
          generate: () => ["action:fixed"],
          scoreSequence: () => 0,
        };
      },
    };
    const model = trainMovementModel([seq("action:a")], {}, constant);
    expect(model.stats.backend).toBe("constant");
    expect(model.generate([], { maxSteps: 1 })).toEqual(["action:fixed"]);
  });

  it("exposes training stats from the markov backend", () => {
    const backend = new MarkovMovementBackend();
    const model = backend.train([seq("action:a", "action:b")], DEFAULT_MOVEMENT_MODEL_CONFIG);
    expect(model.stats.backend).toBe("markov");
    expect(model.stats.sequences).toBe(1);
    expect(model.stats.vocabulary).toBeGreaterThanOrEqual(4); // a, b, <bos>, <eos>
  });
});

describe("synthetic capture → dataset → train → eval loop", () => {
  it("round-trips synthetic streams through tokenization and replay manifest", () => {
    const trajectories = generateSyntheticTrajectories({ seed: 7, count: 3 });
    expect(trajectories).toHaveLength(3);
    const manifest = syntheticReplayManifest("session-1", trajectories);
    expect(manifest.eventCount).toBe(manifest.events.length);
    expect(manifest.trajectoryIds).toHaveLength(3);
    const tokens = tokenizeEvents(manifest.events);
    expect(tokens[0]).toBe(MOVEMENT_BOS);
    expect(tokens.at(-1)).toBe(MOVEMENT_EOS);
  });

  it("is deterministic for a fixed seed", () => {
    const a = generateSyntheticTrajectories({ seed: 99, count: 4 });
    const b = generateSyntheticTrajectories({ seed: 99, count: 4 });
    expect(a).toEqual(b);
  });

  it("learns synthetic patterns and scores held-out trajectories well", () => {
    // Train on one pattern family; hold out fresh draws of the same pattern.
    const train = generateSyntheticTrajectories({ seed: 1, count: 20, patterns: ["form-fill"] }).map(
      (trajectory) => ({ id: trajectory.id, tokens: tokenizeEvents(trajectory.events) }),
    );
    const heldOut = generateSyntheticTrajectories({ seed: 2, count: 5, patterns: ["form-fill"] }).map(
      (trajectory) => ({ id: trajectory.id, tokens: tokenizeEvents(trajectory.events) }),
    );
    const model = trainMovementModel(train, { order: 3 });
    const evalResult = evaluateMovementModel(model, heldOut);
    // The form-fill pattern is deterministic, so a well-trained model should
    // reproduce held-out sequences with high next-movement accuracy.
    expect(evalResult.accuracy).toBeGreaterThan(0.9);
    expect(evalResult.meanLogProb).toBeLessThanOrEqual(0);
    expect(evalResult.predictedTokens).toBeGreaterThan(0);
  });

  it("generation reconstructs a learned synthetic pattern", () => {
    const train = generateSyntheticTrajectories({ seed: 5, count: 10, patterns: ["copy-paste"] }).map(
      (trajectory) => ({ tokens: tokenizeEvents(trajectory.events) }),
    );
    const model = trainMovementModel(train, { order: 3 });
    const generated = model.generate([MOVEMENT_BOS], { maxSteps: 20 });
    expect(generated).toEqual([
      "action:mouse.drag",
      "action:keyboard.shortcut",
      "action:mouse.click",
      "action:keyboard.shortcut",
    ]);
  });
});
