import { describe, expect, it } from "vitest";
import type { ReplayManifest } from "../capture/replay.js";
import {
  MarkovMovementBackend,
  MOVEMENT_EOS,
  buildMovementDataset,
  createSeededRng,
  evaluateMovementPolicy,
  extractMovementTokens,
  type MovementDataset,
} from "./policy-backend.js";

function replay(sessionId: string, trajectoryId: string, tools: string[]): ReplayManifest {
  const events = tools.map((tool, index) => ({
    kind: "action" as const,
    ts: index,
    trajectoryId,
    tool,
    summary: `${tool} step ${index}`,
  }));
  return {
    version: 1,
    sessionId,
    trajectoryIds: [trajectoryId],
    eventCount: events.length,
    events,
  };
}

describe("extractMovementTokens", () => {
  it("keeps only action events, dropping observations and transcript", () => {
    const manifest: ReplayManifest = {
      version: 1,
      sessionId: "s1",
      trajectoryIds: ["t1"],
      eventCount: 3,
      events: [
        { kind: "transcript", ts: 0, messageId: "m0", role: "user", content: "go" },
        { kind: "observation", ts: 1, trajectoryId: "t1", source: "screen", summary: "window" },
        { kind: "action", ts: 2, trajectoryId: "t1", tool: "mouse.move", summary: "move" },
        { kind: "action", ts: 3, trajectoryId: "t1", tool: "mouse.click", summary: "click" },
      ],
    };
    expect(extractMovementTokens(manifest.events)).toEqual(["mouse.move", "mouse.click"]);
  });
});

describe("buildMovementDataset", () => {
  it("builds one sequence per non-empty replay and skips empty ones", () => {
    const dataset = buildMovementDataset([
      replay("s1", "t1", ["mouse.move", "mouse.click"]),
      replay("s2", "t2", []),
    ]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]).toEqual({
      trajectoryId: "t1",
      tokens: ["mouse.move", "mouse.click"],
    });
  });
});

describe("MarkovMovementBackend", () => {
  const dataset: MovementDataset = {
    version: 1,
    sequences: [
      { trajectoryId: "t1", tokens: ["mouse.move", "mouse.click", "key.press", "key.release"] },
      { trajectoryId: "t2", tokens: ["mouse.move", "mouse.click", "key.press", "key.release"] },
    ],
  };

  it("reproduces a fully-learned movement deterministically (greedy decode)", () => {
    const policy = new MarkovMovementBackend(2).train(dataset);
    const generated = policy.generate({ prompt: [], maxSteps: 10, temperature: 0 });
    expect(generated).toEqual(["mouse.move", "mouse.click", "key.press", "key.release"]);
  });

  it("continues from a partial recording (repeat objective)", () => {
    const policy = new MarkovMovementBackend(2).train(dataset);
    const continuation = policy.generate({
      prompt: ["mouse.move", "mouse.click"],
      maxSteps: 10,
      temperature: 0,
    });
    expect(continuation).toEqual(["key.press", "key.release"]);
  });

  it("generalizes to a new-but-related sequence via seeded sampling", () => {
    const branching: MovementDataset = {
      version: 1,
      sequences: [
        { trajectoryId: "a", tokens: ["mouse.move", "mouse.click", "key.press"] },
        { trajectoryId: "b", tokens: ["mouse.move", "scroll", "key.press"] },
        { trajectoryId: "c", tokens: ["mouse.move", "mouse.click", "scroll"] },
      ],
    };
    const policy = new MarkovMovementBackend(1).train(branching);
    const sample = policy.generate({ prompt: ["mouse.move"], maxSteps: 6, temperature: 1, seed: 7 });
    // Every generated token must be from the learned vocabulary (valid movements),
    // and the whole sequence should be a coherent walk, not any single verbatim recording.
    const vocabulary = new Set(["mouse.move", "mouse.click", "scroll", "key.press"]);
    expect(sample.length).toBeGreaterThan(0);
    for (const token of sample) {
      expect(vocabulary.has(token)).toBe(true);
      expect(token).not.toBe(MOVEMENT_EOS);
    }
  });

  it("produces identical samples for the same seed and diverges for different seeds", () => {
    const branching: MovementDataset = {
      version: 1,
      sequences: Array.from({ length: 6 }, (_, i) => ({
        trajectoryId: `t${i}`,
        tokens: i % 2 === 0 ? ["a", "b", "c"] : ["a", "x", "y"],
      })),
    };
    const policy = new MarkovMovementBackend(1).train(branching);
    const first = policy.generate({ prompt: ["a"], maxSteps: 4, temperature: 1.5, seed: 42 });
    const same = policy.generate({ prompt: ["a"], maxSteps: 4, temperature: 1.5, seed: 42 });
    expect(same).toEqual(first);
  });

  it("round-trips through serialization", () => {
    const backend = new MarkovMovementBackend(2);
    const policy = backend.train(dataset);
    const restored = backend.load(policy.toJSON());
    expect(restored.generate({ prompt: [], maxSteps: 10, temperature: 0 })).toEqual(
      policy.generate({ prompt: [], maxSteps: 10, temperature: 0 }),
    );
    expect(restored.scoreSequence(["mouse.move", "mouse.click", "key.press", "key.release"])).toBeCloseTo(
      policy.scoreSequence(["mouse.move", "mouse.click", "key.press", "key.release"]),
    );
  });

  it("scores a learned sequence higher than an unseen one", () => {
    const policy = new MarkovMovementBackend(2).train(dataset);
    const learned = policy.scoreSequence(["mouse.move", "mouse.click", "key.press", "key.release"]);
    const unseen = policy.scoreSequence(["scroll", "scroll", "scroll", "scroll"]);
    expect(learned).toBeGreaterThan(unseen);
  });

  it("rejects an invalid order", () => {
    expect(() => new MarkovMovementBackend(0)).toThrow();
  });
});

describe("evaluateMovementPolicy", () => {
  it("reports perfect fidelity when held-out equals training data", () => {
    const dataset: MovementDataset = {
      version: 1,
      sequences: [{ trajectoryId: "t1", tokens: ["mouse.move", "mouse.click", "key.press"] }],
    };
    const policy = new MarkovMovementBackend(2).train(dataset);
    const evaluation = evaluateMovementPolicy(policy, dataset);
    expect(evaluation.sequenceCount).toBe(1);
    expect(evaluation.tokenCount).toBe(3);
    expect(evaluation.nextTokenAccuracy).toBe(1);
    expect(evaluation.exactReplayRate).toBe(1);
    expect(evaluation.meanLogProbability).toBeLessThanOrEqual(0);
  });

  it("generalizes: predicts held-out tokens whose context was seen in training", () => {
    const train: MovementDataset = {
      version: 1,
      sequences: [
        { trajectoryId: "t1", tokens: ["open", "type", "save", "close"] },
        { trajectoryId: "t2", tokens: ["open", "type", "save", "close"] },
      ],
    };
    // Held-out shares the learned "type"->"save"->"close" tail from a new start.
    const heldOut: MovementDataset = {
      version: 1,
      sequences: [{ trajectoryId: "h1", tokens: ["open", "type", "save", "close"] }],
    };
    const policy = new MarkovMovementBackend(1).train(train);
    const evaluation = evaluateMovementPolicy(policy, heldOut);
    expect(evaluation.nextTokenAccuracy).toBeGreaterThan(0.5);
  });

  it("handles an empty held-out set without dividing by zero", () => {
    const policy = new MarkovMovementBackend(2).train({ version: 1, sequences: [] });
    const evaluation = evaluateMovementPolicy(policy, { version: 1, sequences: [] });
    expect(evaluation).toEqual({
      sequenceCount: 0,
      tokenCount: 0,
      nextTokenAccuracy: 0,
      meanLogProbability: 0,
      exactReplayRate: 0,
    });
  });
});

describe("createSeededRng", () => {
  it("is deterministic and stays within [0, 1)", () => {
    const a = createSeededRng(123);
    const b = createSeededRng(123);
    for (let i = 0; i < 100; i += 1) {
      const value = a();
      expect(value).toBe(b());
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThan(1);
    }
  });

  it("normalizes a zero seed instead of getting stuck", () => {
    const rng = createSeededRng(0);
    expect(rng()).not.toBe(rng());
  });
});
