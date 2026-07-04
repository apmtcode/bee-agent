import { describe, expect, it } from "vitest";
import { buildReplayManifest } from "../capture/replay.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import {
  DEFAULT_MOVEMENT_MODEL_CONFIG,
  NgramMovementBackend,
  buildMovementDataset,
  evaluateMovementPolicy,
  generateSyntheticMovementDataset,
  loadMovementPolicy,
  movementSequenceFromReplay,
  movementSequenceFromTrajectory,
  movementTokenFromAction,
  splitMovementDataset,
  type MovementSequence,
} from "./movement-model.js";

function seq(id: string, tokens: string[], label?: string): MovementSequence {
  return {
    id,
    ...(label ? { label } : {}),
    steps: tokens.map((token, index) => ({ token, tool: "device", summary: token, ts: index })),
  };
}

describe("movement tokenization", () => {
  it("normalizes summaries into stable, comparable tokens", () => {
    expect(movementTokenFromAction({ tool: "device", summary: "Tapped  Send." })).toBe("device:tapped send");
    expect(movementTokenFromAction({ tool: "device", summary: "tapped send" })).toBe("device:tapped send");
  });

  it("derives an ordered movement sequence from a trajectory", () => {
    const trajectory = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [
        { kind: "action", tool: "device", summary: "tapped send", ts: 30 },
        { kind: "action", tool: "device", summary: "tapped compose", ts: 10 },
      ],
    });
    const sequence = movementSequenceFromTrajectory(trajectory);
    expect(sequence.steps.map((step) => step.summary)).toEqual(["tapped compose", "tapped send"]);
  });

  it("derives a movement sequence from a replay manifest's action events", () => {
    const trajectory = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      observations: [{ kind: "observation", source: "device", summary: "app active", ts: 5 }],
      actions: [{ kind: "action", tool: "device", summary: "tapped send", ts: 10 }],
    });
    const manifest = buildReplayManifest({ sessionId: "s1", transcript: [], trajectories: [trajectory] });
    const sequence = movementSequenceFromReplay(manifest);
    expect(sequence.steps).toHaveLength(1);
    expect(sequence.steps[0]?.token).toBe("device:tapped send");
  });
});

describe("NgramMovementBackend", () => {
  it("repeats a recorded movement exactly (objective 2c)", async () => {
    const dataset = buildMovementDataset([seq("a", ["open", "compose", "type", "send"])]);
    const policy = await new NgramMovementBackend().train(dataset);
    expect(policy.rollout(["open"])).toEqual(["compose", "type", "send"]);
  });

  it("is deterministic across retrains and picks the majority next movement", async () => {
    const dataset = buildMovementDataset([
      seq("a", ["open", "send"]),
      seq("b", ["open", "send"]),
      seq("c", ["open", "cancel"]),
    ]);
    const backend = new NgramMovementBackend();
    const first = await backend.train(dataset);
    const second = await backend.train(dataset);
    expect(first.predictNext(["open"]).token).toBe("send");
    expect(second.predictNext(["open"]).token).toBe("send");
    expect(first.predictNext(["open"]).confidence).toBeCloseTo(2 / 3, 5);
  });

  it("generalizes to an unseen prefix via backoff (objective 2d)", async () => {
    // "compose" is always followed by "send"; the full 2-gram prefix
    // ["brandnew", "compose"] was never seen, so the policy must back off.
    const dataset = buildMovementDataset([
      seq("a", ["open", "compose", "send"]),
      seq("b", ["reply", "compose", "send"]),
    ]);
    const policy = await new NgramMovementBackend().train(dataset, { order: 2 });
    const prediction = policy.predictNext(["brandnew", "compose"]);
    expect(prediction.token).toBe("send");
    expect(prediction.contextOrderUsed).toBe(1);
  });

  it("terminates rollouts at the learned end-of-sequence", async () => {
    const dataset = buildMovementDataset([seq("a", ["open", "send"])]);
    const policy = await new NgramMovementBackend().train(dataset);
    const rollout = policy.rollout(["open"], { maxSteps: 50 });
    expect(rollout).toEqual(["send"]);
  });

  it("round-trips through a serialized snapshot", async () => {
    const dataset = buildMovementDataset([seq("a", ["open", "compose", "send"])]);
    const trained = await new NgramMovementBackend().train(dataset);
    const restored = loadMovementPolicy(trained.toJSON());
    expect(restored.rollout(["open"])).toEqual(trained.rollout(["open"]));
    expect(restored.order).toBe(trained.order);
    expect([...restored.vocabulary]).toEqual([...trained.vocabulary]);
  });
});

describe("synthetic dataset + generalization eval", () => {
  it("generates reproducible related sequences from a seed", () => {
    const first = generateSyntheticMovementDataset({ sequences: 12, seed: 7 });
    const second = generateSyntheticMovementDataset({ sequences: 12, seed: 7 });
    expect(first).toEqual(second);
    expect(first.sequences).toHaveLength(12);
    expect(first.sequences.every((sequence) => sequence.steps.length > 0)).toBe(true);
  });

  it("generalizes to held-out related trajectories above a fidelity floor", async () => {
    const dataset = generateSyntheticMovementDataset({ sequences: 60, seed: 3 });
    const { train, heldOut } = splitMovementDataset(dataset, 4);
    expect(heldOut.length).toBeGreaterThan(0);

    const policy = await new NgramMovementBackend().train(
      buildMovementDataset(train),
      DEFAULT_MOVEMENT_MODEL_CONFIG,
    );
    const result = evaluateMovementPolicy(policy, heldOut);

    // The grammar is fully learnable from related samples: the policy should
    // predict most next movements on trajectories it never trained on.
    expect(result.sequences).toBe(heldOut.length);
    expect(result.nextStepAccuracy).toBeGreaterThan(0.9);
    expect(result.rolloutMatchRate).toBeGreaterThan(0.5);
    expect(result.averageConfidence).toBeGreaterThan(0);
  });

  it("reports zeroed metrics for empty held-out sets", async () => {
    const policy = await new NgramMovementBackend().train(buildMovementDataset([seq("a", ["x", "y"])]));
    const result = evaluateMovementPolicy(policy, []);
    expect(result).toMatchObject({ sequences: 0, steps: 0, correct: 0, nextStepAccuracy: 0, rolloutMatchRate: 0 });
  });
});
