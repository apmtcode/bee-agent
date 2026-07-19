import { describe, expect, it } from "vitest";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import {
  MarkovMovementBackend,
  MOVEMENT_END,
  buildMovementDataset,
  deserializeMovementPolicy,
  evaluateMovementPolicy,
  generateSyntheticMovementDataset,
  tokenizeTrajectory,
  type MovementDataset,
  type MovementPolicyBackend,
  type MovementSequence,
  type TrainedMovementPolicy,
} from "./movement-policy.js";

function dataset(...sequences: Array<[string, string[]]>): MovementDataset {
  return { sequences: sequences.map(([id, tokens]): MovementSequence => ({ id, tokens })) };
}

describe("tokenizeTrajectory", () => {
  it("orders observations and actions by timestamp into stable movement tokens", () => {
    const trajectory = buildTrajectorySpan({
      id: "traj-1",
      sessionId: "sess-1",
      observations: [{ kind: "observation", source: "screen", summary: "Menu opened", ts: 5 }],
      actions: [
        { kind: "action", tool: "mouse", summary: "Click file", ts: 10 },
        { kind: "action", tool: "keyboard", summary: "Type name", ts: 20 },
      ],
    });

    const sequence = tokenizeTrajectory(trajectory);
    expect(sequence.id).toBe("traj-1");
    expect(sequence.tokens).toEqual(["obs:screen:menu", "action:mouse:click", "action:keyboard:type"]);
  });

  it("prefers the reviewed/redacted view when present", () => {
    const trajectory = buildTrajectorySpan({
      id: "traj-2",
      sessionId: "sess-1",
      actions: [{ kind: "action", tool: "mouse", summary: "Click secret", ts: 10 }],
    });
    trajectory.review = {
      status: "approved",
      reviewedAt: "2026-01-01T00:00:00.000Z",
      reviewedBy: "tester",
      redactedActions: [{ ts: 10, tool: "mouse", summary: "Redacted gesture" }],
    };
    expect(tokenizeTrajectory(trajectory).tokens).toEqual(["action:mouse:redacted"]);
  });

  it("skips empty trajectories when building a dataset", () => {
    const withActions = buildTrajectorySpan({
      id: "a",
      sessionId: "s",
      actions: [{ kind: "action", tool: "mouse", summary: "click", ts: 1 }],
    });
    const empty = buildTrajectorySpan({ id: "b", sessionId: "s" });
    const built = buildMovementDataset([withActions, empty]);
    expect(built.sequences.map((sequence) => sequence.id)).toEqual(["a"]);
  });
});

describe("MarkovMovementBackend", () => {
  const backend: MovementPolicyBackend = new MarkovMovementBackend();

  it("reproduces a recorded movement sequence exactly (repeat)", () => {
    const training = dataset(["t1", ["action:mouse:move", "action:mouse:click", "action:keyboard:type"]]);
    const policy = backend.train(training, { order: 2 });
    expect(policy.rollout([])).toEqual(["action:mouse:move", "action:mouse:click", "action:keyboard:type"]);
  });

  it("predicts the most likely next movement with backoff on novel contexts", () => {
    const training = dataset(
      ["t1", ["action:mouse:move", "action:mouse:click"]],
      ["t2", ["action:mouse:move", "action:mouse:click"]],
      ["t3", ["action:mouse:move", "action:keyboard:type"]],
    );
    const policy = backend.train(training, { order: 2 });

    // Exact context seen twice -> click preferred over type.
    const seen = policy.predictNext(["action:mouse:move"]);
    expect(seen?.token).toBe("action:mouse:click");
    expect(seen?.ranked.map((r) => r.token)).toContain("action:keyboard:type");

    // Novel longer context falls back to a shorter seen context.
    const novel = policy.predictNext(["action:window:focus", "action:mouse:move"]);
    expect(novel?.token).toBe("action:mouse:click");
    expect(novel?.backoffOrder).toBeLessThan(2);
  });

  it("is deterministic across trainings and round-trips through JSON", () => {
    const training = dataset(
      ["t1", ["action:mouse:move", "action:mouse:click", "action:keyboard:type"]],
      ["t2", ["action:mouse:move", "action:mouse:drag"]],
    );
    const a = backend.train(training, { order: 2 });
    const b = backend.train(training, { order: 2 });
    expect(a.toJSON()).toEqual(b.toJSON());

    const restored = deserializeMovementPolicy(a.toJSON());
    expect(restored.rollout([])).toEqual(a.rollout([]));
    expect(restored.predictNext(["action:mouse:move"])).toEqual(a.predictNext(["action:mouse:move"]));
  });

  it("terminates rollouts at the END sentinel and never emits it", () => {
    const training = dataset(["t1", ["action:mouse:click"]]);
    const policy = backend.train(training, { order: 1 });
    const rolled = policy.rollout([], { maxLength: 50 });
    expect(rolled).toEqual(["action:mouse:click"]);
    expect(rolled).not.toContain(MOVEMENT_END);
  });
});

describe("evaluateMovementPolicy (generalization harness)", () => {
  it("reports perfect fidelity on the training sequence", () => {
    const backend = new MarkovMovementBackend();
    const training = dataset(["t1", ["action:mouse:move", "action:mouse:click", "action:keyboard:type"]]);
    const policy = backend.train(training, { order: 3 });
    const result = evaluateMovementPolicy(policy, training, { topK: 2 });
    expect(result.sequenceCount).toBe(1);
    expect(result.nextTokenAccuracy).toBe(1);
    expect(result.perfectReplayRate).toBe(1);
    expect(result.meanReciprocalRank).toBe(1);
  });

  it("generalizes above chance to held-out related synthetic sequences", () => {
    const backend = new MarkovMovementBackend();
    const all = generateSyntheticMovementDataset({ seed: 7, sequenceCount: 60, minLength: 4, maxLength: 9 });
    const train: MovementDataset = { sequences: all.sequences.slice(0, 45) };
    const heldOut: MovementDataset = { sequences: all.sequences.slice(45) };

    const policy = backend.train(train, { order: 2 });
    const result = evaluateMovementPolicy(policy, heldOut, { topK: 3 });

    // 6-symbol vocabulary -> chance top-1 ~1/7 (incl. END). The structured
    // stream must be learned well above chance and top-3 clearly better still.
    expect(result.predictionCount).toBeGreaterThan(0);
    expect(result.nextTokenAccuracy).toBeGreaterThan(0.33);
    expect(result.topKAccuracy).toBeGreaterThan(result.nextTokenAccuracy);
    expect(result.meanReciprocalRank).toBeGreaterThan(result.nextTokenAccuracy);
  });

  it("returns zeroed metrics for an empty held-out set", () => {
    const backend = new MarkovMovementBackend();
    const policy = backend.train(dataset(["t1", ["action:mouse:click"]]));
    const result = evaluateMovementPolicy(policy, { sequences: [] });
    expect(result).toMatchObject({
      sequenceCount: 0,
      predictionCount: 0,
      nextTokenAccuracy: 0,
      perfectReplayRate: 0,
    });
  });
});

describe("generateSyntheticMovementDataset", () => {
  it("is reproducible for a fixed seed and varies by seed", () => {
    const a = generateSyntheticMovementDataset({ seed: 42, sequenceCount: 5 });
    const b = generateSyntheticMovementDataset({ seed: 42, sequenceCount: 5 });
    const c = generateSyntheticMovementDataset({ seed: 43, sequenceCount: 5 });
    expect(a).toEqual(b);
    expect(a).not.toEqual(c);
    expect(a.sequences).toHaveLength(5);
    expect(a.sequences.every((sequence) => sequence.tokens.length >= 3)).toBe(true);
  });
});

describe("pluggable backend seam", () => {
  it("accepts an alternative backend implementing the same interface", () => {
    // A trivial "always repeat the first token" backend proves the seam is real.
    class ConstantBackend implements MovementPolicyBackend {
      readonly id = "constant";
      train(data: MovementDataset): TrainedMovementPolicy {
        const first = data.sequences[0]?.tokens[0] ?? MOVEMENT_END;
        return {
          backendId: this.id,
          order: 0,
          predictNext: () => ({ token: first, probability: 1, ranked: [{ token: first, probability: 1 }], backoffOrder: 0 }),
          rollout: () => [first],
          toJSON: () => ({ version: 1, backendId: this.id, order: 0, vocabulary: [first], counts: {} }),
        };
      }
    }
    const backend: MovementPolicyBackend = new ConstantBackend();
    const policy = backend.train(dataset(["t1", ["action:mouse:click", "action:keyboard:type"]]));
    expect(policy.rollout([])).toEqual(["action:mouse:click"]);
    expect(policy.backendId).toBe("constant");
  });
});
