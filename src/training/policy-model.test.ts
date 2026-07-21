import { describe, expect, it } from "vitest";
import type { ReplayTimelineEvent } from "../capture/replay.js";
import {
  NgramMovementPolicyBackend,
  SEQUENCE_END,
  buildMovementDataset,
  buildSyntheticMovementSequences,
  deserializeMovementPolicy,
  evaluateMovementPolicy,
  rolloutMovementPolicy,
  tokenizeMovementEvent,
  type MovementSequence,
} from "./policy-model.js";

function actionEvent(trajectoryId: string, ts: number, summary: string): ReplayTimelineEvent {
  return { kind: "action", ts, trajectoryId, tool: "device", summary };
}

describe("tokenizeMovementEvent", () => {
  it("normalizes action summaries and ignores non-actions", () => {
    expect(tokenizeMovementEvent(actionEvent("t1", 1, "Tapped  Compose."))).toBe("device::tapped compose");
    expect(
      tokenizeMovementEvent({ kind: "observation", ts: 1, trajectoryId: "t1", source: "device", summary: "x" }),
    ).toBeUndefined();
  });
});

describe("buildMovementDataset", () => {
  it("groups action events by trajectory and orders them by timestamp", () => {
    const dataset = buildMovementDataset([
      {
        events: [
          actionEvent("t1", 30, "tapped inbox"),
          actionEvent("t1", 10, "tapped compose"),
          { kind: "observation", ts: 5, trajectoryId: "t1", source: "device", summary: "ignored" },
          actionEvent("t2", 20, "swiped up sidebar"),
          actionEvent("t1", 20, "typed into body"),
        ],
      },
    ]);

    const t1 = dataset.sequences.find((sequence) => sequence.trajectoryId === "t1");
    expect(t1?.tokens).toEqual(["device::tapped compose", "device::typed into body", "device::tapped inbox"]);
    expect(dataset.sequences).toHaveLength(2);
    expect(dataset.vocabulary).toContain("device::swiped up sidebar");
  });
});

describe("NgramMovementPolicyBackend", () => {
  it("repeats a recorded movement sequence exactly (objective 2c)", async () => {
    const dataset = buildMovementDataset([
      {
        events: [
          actionEvent("t1", 1, "tapped compose"),
          actionEvent("t1", 2, "typed into body"),
          actionEvent("t1", 3, "tapped send"),
        ],
      },
    ]);
    const policy = await new NgramMovementPolicyBackend().train(dataset, { order: 2 });
    const replayed = rolloutMovementPolicy(policy, { maxSteps: 10 });
    expect(replayed).toEqual(["device::tapped compose", "device::typed into body", "device::tapped send"]);
  });

  it("generalizes to a new-but-related movement via back-off (objective 2d)", async () => {
    // Two trajectories share the continuation after "typed into body": send.
    // A novel 2-gram context that was never seen backs off to the 1-gram, which
    // predicts the shared "tapped send" continuation.
    const dataset = buildMovementDataset([
      {
        events: [
          actionEvent("t1", 1, "tapped compose"),
          actionEvent("t1", 2, "typed into body"),
          actionEvent("t1", 3, "tapped send"),
          actionEvent("t2", 1, "tapped reply"),
          actionEvent("t2", 2, "typed into body"),
          actionEvent("t2", 3, "tapped send"),
        ],
      },
    ]);
    const policy = await new NgramMovementPolicyBackend().train(dataset, { order: 2 });

    // "tapped inbox" was never a predecessor of "typed into body" in training,
    // so the full 2-gram context is novel and must back off.
    const prediction = policy.predict(["device::tapped inbox", "device::typed into body"]);
    expect(prediction.token).toBe("device::tapped send");
    expect(prediction.contextOrder).toBeLessThan(2);
  });

  it("uses the full context when it was observed (no unnecessary back-off)", async () => {
    const dataset = buildMovementDataset([
      {
        events: [
          actionEvent("t1", 1, "tapped compose"),
          actionEvent("t1", 2, "typed into body"),
          actionEvent("t1", 3, "tapped send"),
        ],
      },
    ]);
    const policy = await new NgramMovementPolicyBackend().train(dataset, { order: 2 });
    const prediction = policy.predict(["device::tapped compose", "device::typed into body"]);
    expect(prediction.token).toBe("device::tapped send");
    expect(prediction.contextOrder).toBe(2);
  });

  it("is deterministic across repeated training", async () => {
    const sequences = buildSyntheticMovementSequences({ count: 8, seed: 42 });
    const dataset = { version: 1 as const, sequences, vocabulary: [] };
    const backend = new NgramMovementPolicyBackend();
    const a = await backend.train(dataset, { order: 2 });
    const b = await backend.train(dataset, { order: 2 });
    expect(a.serialize()).toEqual(b.serialize());
    expect(rolloutMovementPolicy(a, { maxSteps: 20 })).toEqual(rolloutMovementPolicy(b, { maxSteps: 20 }));
  });
});

describe("evaluateMovementPolicy", () => {
  it("scores next-token accuracy and records the back-off histogram", async () => {
    const training = buildSyntheticMovementSequences({ count: 30, seed: 7 });
    const dataset = { version: 1 as const, sequences: training, vocabulary: [] };
    const policy = await new NgramMovementPolicyBackend().train(dataset, { order: 2 });

    // Held-out set drawn from the same generator (related, not identical).
    const heldOut = buildSyntheticMovementSequences({ count: 10, seed: 99 });
    const evaluation = evaluateMovementPolicy(policy, heldOut);

    expect(evaluation.totalPredictions).toBeGreaterThan(0);
    expect(evaluation.accuracy).toBeGreaterThanOrEqual(0);
    expect(evaluation.accuracy).toBeLessThanOrEqual(1);
    expect(evaluation.perSequence).toHaveLength(10);
    const histogramTotal = Object.values(evaluation.backoffHistogram).reduce((sum, value) => sum + value, 0);
    expect(histogramTotal).toBe(evaluation.totalPredictions);
  });

  it("perfectly predicts sequences it memorized", async () => {
    const sequences: MovementSequence[] = [
      { trajectoryId: "a", tokens: ["device::tapped compose", "device::tapped send"] },
    ];
    const dataset = { version: 1 as const, sequences, vocabulary: [] };
    const policy = await new NgramMovementPolicyBackend().train(dataset, { order: 2 });
    const evaluation = evaluateMovementPolicy(policy, sequences);
    expect(evaluation.accuracy).toBe(1);
    expect(evaluation.correct).toBe(evaluation.totalPredictions);
  });
});

describe("serialize / deserialize", () => {
  it("round-trips a trained policy", async () => {
    const sequences = buildSyntheticMovementSequences({ count: 5, seed: 3 });
    const dataset = { version: 1 as const, sequences, vocabulary: [] };
    const policy = await new NgramMovementPolicyBackend().train(dataset, { order: 2, smoothing: 0.1 });
    const restored = deserializeMovementPolicy(policy.serialize());

    expect(restored.order).toBe(policy.order);
    expect(restored.vocabulary).toEqual(policy.vocabulary);
    const context = sequences[0]?.tokens.slice(0, 1) ?? [];
    expect(restored.predict(context)).toEqual(policy.predict(context));
    expect(rolloutMovementPolicy(restored, { maxSteps: 20 })).toEqual(rolloutMovementPolicy(policy, { maxSteps: 20 }));
  });
});

describe("buildSyntheticMovementSequences", () => {
  it("is deterministic for a given seed and varies by seed", () => {
    expect(buildSyntheticMovementSequences({ count: 4, seed: 5 })).toEqual(
      buildSyntheticMovementSequences({ count: 4, seed: 5 }),
    );
    expect(buildSyntheticMovementSequences({ count: 4, seed: 5 })).not.toEqual(
      buildSyntheticMovementSequences({ count: 4, seed: 6 }),
    );
  });

  it("respects length bounds", () => {
    const sequences = buildSyntheticMovementSequences({ count: 20, seed: 11, minLength: 2, maxLength: 4 });
    for (const sequence of sequences) {
      expect(sequence.tokens.length).toBeGreaterThanOrEqual(2);
      expect(sequence.tokens.length).toBeLessThanOrEqual(4);
    }
  });
});

describe("rollout guards", () => {
  it("can include the terminal end token", async () => {
    const sequences: MovementSequence[] = [{ trajectoryId: "a", tokens: ["device::tapped compose"] }];
    const dataset = { version: 1 as const, sequences, vocabulary: [] };
    const policy = await new NgramMovementPolicyBackend().train(dataset, { order: 2 });
    expect(rolloutMovementPolicy(policy, { maxSteps: 5, includeEnd: true })).toEqual([
      "device::tapped compose",
      SEQUENCE_END,
    ]);
  });
});
