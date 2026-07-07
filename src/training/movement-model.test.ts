import { describe, expect, it } from "vitest";
import type { ReplayManifest } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  NgramMovementBackend,
  buildMovementDataset,
  deserializeMovementPolicy,
  evaluateMovementPolicy,
  generateSyntheticMovementSequences,
  sequenceFromReplay,
  sequenceFromTrajectory,
  tokensOf,
  trainMovementPolicy,
  type MovementSequence,
} from "./movement-model.js";

function seq(id: string, tokens: string[]): MovementSequence {
  return {
    id,
    events: tokens.map((token, index) => ({
      kind: token.startsWith("act:") ? "action" : "observation",
      token,
      label: token,
      ts: index * 1000,
    })),
  };
}

describe("movement sequence extraction", () => {
  it("tokenizes a replay manifest, dropping transcript events", () => {
    const replay: Pick<ReplayManifest, "sessionId" | "events"> = {
      sessionId: "session-a",
      events: [
        { kind: "transcript", ts: 0, messageId: "m1", role: "user", content: "hi" },
        { kind: "observation", ts: 1, trajectoryId: "t1", source: "Screen", summary: "window shown" },
        { kind: "action", ts: 2, trajectoryId: "t1", tool: "Click", summary: "clicked ok" },
      ],
    };
    const sequence = sequenceFromReplay(replay);
    expect(tokensOf(sequence)).toEqual(["obs:screen", "act:click"]);
    expect(sequence.events[1]?.label).toBe("Click: clicked ok");
  });

  it("can exclude observations", () => {
    const replay: Pick<ReplayManifest, "sessionId" | "events"> = {
      sessionId: "s",
      events: [
        { kind: "observation", ts: 1, trajectoryId: "t", source: "screen", summary: "x" },
        { kind: "action", ts: 2, trajectoryId: "t", tool: "type", summary: "y" },
      ],
    };
    expect(tokensOf(sequenceFromReplay(replay, { includeObservations: false }))).toEqual(["act:type"]);
  });

  it("orders trajectory events by timestamp", () => {
    const trajectory: Pick<TrajectorySpan, "id" | "observations" | "actions"> = {
      id: "t1",
      observations: [{ kind: "observation", source: "screen", summary: "s", ts: 5 }],
      actions: [
        { kind: "action", tool: "click", summary: "c", ts: 1 },
        { kind: "action", tool: "save", summary: "v", ts: 9 },
      ],
    };
    expect(tokensOf(sequenceFromTrajectory(trajectory))).toEqual(["act:click", "obs:screen", "act:save"]);
  });
});

describe("dataset building", () => {
  it("windows examples up to the given order and collects vocabulary", () => {
    const dataset = buildMovementDataset([seq("a", ["act:a", "act:b", "act:c"])], { order: 2 });
    expect(dataset.order).toBe(2);
    expect(dataset.vocabulary).toEqual(["act:a", "act:b", "act:c"]);
    expect(dataset.examples).toEqual([
      { context: ["act:a"], next: "act:b" },
      { context: ["act:a", "act:b"], next: "act:c" },
    ]);
  });

  it("drops empty sequences", () => {
    const dataset = buildMovementDataset([seq("empty", [])], { order: 2 });
    expect(dataset.sequences).toEqual([]);
    expect(dataset.examples).toEqual([]);
  });
});

describe("ngram backend training + prediction", () => {
  it("learns to reproduce a deterministic recorded movement", () => {
    const sequences = [seq("a", ["act:focus", "act:click", "act:type", "act:save"])];
    const { policy } = trainMovementPolicy(sequences, { order: 2 });
    expect(policy.backendId).toBe("ngram-backoff");
    expect(policy.predict(["act:focus"]).token).toBe("act:click");
    expect(policy.predict(["act:focus", "act:click"]).token).toBe("act:type");
    expect(policy.predict(["act:click", "act:type"]).token).toBe("act:save");
    expect(policy.predict(["act:click", "act:type"]).confidence).toBe(1);
  });

  it("generalizes to an unseen context via backoff", () => {
    // 'click' is always followed by 'type' across the training set.
    const sequences = [
      seq("a", ["act:focus", "act:click", "act:type"]),
      seq("b", ["act:open", "act:click", "act:type"]),
    ];
    const { policy } = trainMovementPolicy(sequences, { order: 2 });
    // A brand-new prefix ('scroll click') was never seen at order 2...
    const prediction = policy.predict(["act:scroll", "act:click"]);
    // ...but backoff to the seen 1-gram 'click' still predicts 'type'.
    expect(prediction.token).toBe("act:type");
    expect(prediction.backoffOrder).toBe(1);
  });

  it("returns an empty prediction for an untrained model", () => {
    const policy = new NgramMovementBackend().train(buildMovementDataset([], { order: 2 }));
    const prediction = policy.predict(["act:anything"]);
    expect(prediction.token).toBeUndefined();
    expect(prediction.backoffOrder).toBe(-1);
    expect(prediction.candidates).toEqual([]);
  });

  it("ranks candidates by probability with a deterministic tie-break", () => {
    const sequences = [
      seq("a", ["act:x", "act:a"]),
      seq("b", ["act:x", "act:a"]),
      seq("c", ["act:x", "act:b"]),
    ];
    const { policy } = trainMovementPolicy(sequences, { order: 1 });
    const prediction = policy.predict(["act:x"]);
    expect(prediction.candidates.map((c) => c.token)).toEqual(["act:a", "act:b"]);
    expect(prediction.candidates[0]?.probability).toBeCloseTo(2 / 3, 10);
  });
});

describe("policy serialization", () => {
  it("round-trips through serialize/deserialize with identical predictions", () => {
    const sequences = [seq("a", ["act:focus", "act:click", "act:type", "act:save"])];
    const { policy } = trainMovementPolicy(sequences, { order: 2 });
    const restored = deserializeMovementPolicy(policy.serialize());
    expect(restored.order).toBe(policy.order);
    expect(restored.backendId).toBe(policy.backendId);
    for (const context of [["act:focus"], ["act:focus", "act:click"], ["act:click", "act:type"]]) {
      expect(restored.predict(context)).toEqual(policy.predict(context));
    }
  });

  it("produces JSON-safe output", () => {
    const { policy } = trainMovementPolicy([seq("a", ["act:a", "act:b"])], { order: 1 });
    const serialized = policy.serialize();
    expect(() => JSON.parse(JSON.stringify(serialized))).not.toThrow();
    expect(serialized.version).toBe(1);
  });
});

describe("synthetic generator", () => {
  it("is deterministic for a fixed seed", () => {
    const a = generateSyntheticMovementSequences({ count: 5, seed: 42, noise: 0.3 });
    const b = generateSyntheticMovementSequences({ count: 5, seed: 42, noise: 0.3 });
    expect(a.map(tokensOf)).toEqual(b.map(tokensOf));
  });

  it("varies with the seed and yields non-empty sequences", () => {
    const a = generateSyntheticMovementSequences({ count: 8, seed: 1, noise: 0.4 });
    const b = generateSyntheticMovementSequences({ count: 8, seed: 2, noise: 0.4 });
    expect(a).toHaveLength(8);
    expect(a.every((s) => s.events.length > 0)).toBe(true);
    expect(a.map(tokensOf)).not.toEqual(b.map(tokensOf));
  });

  it("produces noise-free sequences that exactly match a template", () => {
    const sequences = generateSyntheticMovementSequences({
      count: 3,
      seed: 7,
      noise: 0,
      workflows: [["act:a", "act:b", "act:c"]],
    });
    for (const sequence of sequences) {
      expect(tokensOf(sequence)).toEqual(["act:a", "act:b", "act:c"]);
    }
  });
});

describe("generalization eval harness", () => {
  it("scores perfect recall on an unambiguous single-workflow distribution", () => {
    // A single deterministic workflow has no branching, so a trained model
    // recalls every next movement exactly.
    const train = generateSyntheticMovementSequences({
      count: 30,
      seed: 1,
      noise: 0,
      workflows: [["act:focus", "act:click", "act:type", "act:save"]],
    });
    const { policy } = trainMovementPolicy(train, { order: 3 });
    const result = evaluateMovementPolicy(policy, train, { topK: 2 });
    expect(result.totalPredictions).toBeGreaterThan(0);
    expect(result.accuracy).toBe(1);
    expect(result.topKAccuracy).toBe(1);
    expect(result.meanConfidence).toBe(1);
  });

  it("generalizes to held-out related sequences well above chance", () => {
    // Train on clean workflows; evaluate on noisy variants of the same tasks.
    const train = generateSyntheticMovementSequences({ count: 60, seed: 1, noise: 0 });
    const heldOut = generateSyntheticMovementSequences({ count: 40, seed: 999, noise: 0.35 });
    const { policy, dataset } = trainMovementPolicy(train, { order: 3 });
    const result = evaluateMovementPolicy(policy, heldOut, { topK: 3 });
    // Chance is ~1/vocabulary; top-1 must clear several times that, and the
    // top-3 set must contain the true movement even more often.
    const chance = 1 / dataset.vocabulary.length;
    expect(result.accuracy).toBeGreaterThan(chance * 3);
    expect(result.topKAccuracy).toBeGreaterThan(result.accuracy);
    expect(result.topKAccuracy).toBeGreaterThan(chance * 4);
    // Some predictions must have come from backoff (order < 3), proving
    // the model handled contexts it never saw verbatim.
    const backoffBuckets = Object.entries(result.byBackoffOrder)
      .filter(([order]) => Number(order) >= 0 && Number(order) < 3)
      .reduce((sum, [, bucket]) => sum + bucket.predictions, 0);
    expect(backoffBuckets).toBeGreaterThan(0);
  });

  it("reports zero predictions for empty eval input", () => {
    const { policy } = trainMovementPolicy([seq("a", ["act:a", "act:b"])], { order: 1 });
    const result = evaluateMovementPolicy(policy, []);
    expect(result.totalPredictions).toBe(0);
    expect(result.accuracy).toBe(0);
    expect(result.meanConfidence).toBe(0);
  });
});
