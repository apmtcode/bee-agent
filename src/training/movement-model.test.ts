import { describe, expect, it } from "vitest";
import { buildTrajectorySpan, type TrajectoryAction } from "../capture/trajectory.js";
import {
  MarkovMovementBackend,
  MOVEMENT_STOP_TOKEN,
  buildMovementDataset,
  datasetFromSequences,
  evaluateMovementModel,
  generateSyntheticMovementSequences,
  sequenceFromTrajectory,
  tokenizeAction,
  tokenizeGesture,
  type MovementSequence,
} from "./movement-model.js";

function gestureAction(gesture: string, target: string, ts: number): TrajectoryAction {
  return {
    kind: "action",
    tool: "device",
    summary: `${gesture} ${target}`,
    ts,
    metadata: { gesture, target },
  };
}

describe("movement tokenization", () => {
  it("canonicalizes gestures into stable tokens", () => {
    expect(tokenizeGesture({ kind: "tap", target: "Submit Button", ts: 1 })).toBe("tap:submit-button");
    expect(tokenizeGesture({ kind: "swipe", direction: "up", ts: 1 })).toBe("swipe:up");
    expect(tokenizeGesture({ kind: "scroll", ts: 1 })).toBe("scroll");
  });

  it("prefers gesture metadata when tokenizing a recorded action", () => {
    expect(tokenizeAction(gestureAction("tap", "Save", 1))).toBe("tap:save");
    expect(
      tokenizeAction({ kind: "action", tool: "shell", summary: "run npm test", ts: 1 }),
    ).toBe("shell:run");
  });

  it("orders trajectory actions by timestamp", () => {
    const trajectory = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [gestureAction("tap", "b", 20), gestureAction("tap", "a", 10)],
    });
    expect(sequenceFromTrajectory(trajectory).tokens).toEqual(["tap:a", "tap:b"]);
  });
});

describe("dataset assembly", () => {
  it("builds a sorted, de-duplicated vocabulary and drops empty sequences", () => {
    const dataset = datasetFromSequences([
      { id: "a", tokens: ["tap:b", "tap:a", "tap:b"] },
      { id: "empty", tokens: [] },
    ]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.vocabulary).toEqual(["tap:a", "tap:b"]);
  });

  it("builds a dataset from trajectories", () => {
    const trajectory = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [gestureAction("tap", "search", 1), gestureAction("type", "query", 2)],
    });
    const dataset = buildMovementDataset([trajectory]);
    expect(dataset.sequences[0].tokens).toEqual(["tap:search", "type:query"]);
  });
});

describe("MarkovMovementBackend — repeat recorded movements", () => {
  it("reproduces a dominant recorded path via deterministic argmax rollout", async () => {
    const pattern = ["tap:search", "type:query", "tap:result", "swipe:up"];
    const dataset = datasetFromSequences(
      Array.from({ length: 5 }, (_, i) => ({ id: `s${i}`, tokens: pattern })),
    );
    const model = await new MarkovMovementBackend().train(dataset, { maxOrder: 3 });
    expect(model.generate([])).toEqual(pattern);
  });

  it("continues from a seed prefix", async () => {
    const pattern = ["tap:menu", "tap:settings", "tap:save"];
    const dataset = datasetFromSequences(
      Array.from({ length: 3 }, (_, i) => ({ id: `s${i}`, tokens: pattern })),
    );
    const model = await new MarkovMovementBackend().train(dataset);
    expect(model.generate(["tap:menu"])).toEqual(pattern);
  });

  it("predicts the stop token at the end of a recorded movement", async () => {
    const dataset = datasetFromSequences([{ id: "s0", tokens: ["tap:a", "tap:b"] }]);
    const model = await new MarkovMovementBackend().train(dataset);
    expect(model.predictNext(["tap:a", "tap:b"]).token).toBe(MOVEMENT_STOP_TOKEN);
  });
});

describe("MarkovMovementBackend — generalize to related movements", () => {
  it("backs off to a shorter context for an unseen prefix", async () => {
    // Every recorded sequence ends "type:query" -> "tap:submit".
    const dataset = datasetFromSequences([
      { id: "s0", tokens: ["tap:search", "type:query", "tap:submit"] },
      { id: "s1", tokens: ["tap:filter", "type:query", "tap:submit"] },
    ]);
    const model = await new MarkovMovementBackend().train(dataset, { maxOrder: 3 });
    // Novel prefix the model never saw verbatim, but ending in "type:query".
    const prediction = model.predictNext(["tap:brandnew", "type:query"]);
    expect(prediction.token).toBe("tap:submit");
    // It generalized: the used order is shorter than the 2-token context.
    expect(prediction.order).toBeLessThan(2);
  });

  it("returns ranked candidates and respects topK", async () => {
    const dataset = datasetFromSequences([
      { id: "s0", tokens: ["tap:a", "tap:x"] },
      { id: "s1", tokens: ["tap:a", "tap:y"] },
      { id: "s2", tokens: ["tap:a", "tap:x"] },
    ]);
    const model = await new MarkovMovementBackend().train(dataset);
    const prediction = model.predictNext(["tap:a"], { topK: 1 });
    expect(prediction.candidates).toHaveLength(1);
    expect(prediction.token).toBe("tap:x"); // seen twice > once
  });
});

describe("snapshot round-trip", () => {
  it("reloads an equivalent model from a snapshot", async () => {
    const dataset = datasetFromSequences([{ id: "s0", tokens: ["tap:a", "tap:b", "tap:c"] }]);
    const backend = new MarkovMovementBackend();
    const model = await backend.train(dataset);
    const reloaded = backend.load(JSON.parse(JSON.stringify(model.snapshot())));
    expect(reloaded.generate([])).toEqual(model.generate([]));
  });

  it("rejects a snapshot from a different backend", () => {
    const backend = new MarkovMovementBackend();
    expect(() =>
      backend.load({ backendId: "other", version: 1, vocabulary: [], payload: {} }),
    ).toThrow(/not "markov-backoff"/);
  });
});

describe("synthetic generator + eval harness", () => {
  it("is deterministic for a fixed seed", () => {
    const scenario = { pattern: ["tap:a", "tap:b", "tap:c"], count: 4, dropRate: 0.3 };
    const first = generateSyntheticMovementSequences([scenario], 42);
    const second = generateSyntheticMovementSequences([scenario], 42);
    expect(first).toEqual(second);
    expect(first).toHaveLength(4);
  });

  it("trains on synthetic data and generalizes to held-out variants", async () => {
    const scenarios = [
      { pattern: ["tap:home", "tap:search", "type:query", "tap:result"], count: 20, dropRate: 0.1 },
    ];
    const all = generateSyntheticMovementSequences(scenarios, 7);
    const train = all.slice(0, 16);
    const heldOut: MovementSequence[] = all.slice(16);
    const model = await new MarkovMovementBackend().train(datasetFromSequences(train), { maxOrder: 2 });

    const report = evaluateMovementModel(model, heldOut, { topK: 3 });
    expect(report.steps).toBeGreaterThan(0);
    // A learnable pattern should be predicted well above chance.
    expect(report.topKAccuracy).toBeGreaterThanOrEqual(0.75);
    expect(report.top1Accuracy).toBeGreaterThan(0.5);
  });

  it("reports zero accuracy for an empty held-out set without throwing", async () => {
    const model = await new MarkovMovementBackend().train(
      datasetFromSequences([{ id: "s0", tokens: ["tap:a"] }]),
    );
    expect(evaluateMovementModel(model, [])).toEqual({
      steps: 0,
      top1Accuracy: 0,
      topKAccuracy: 0,
      backoffRate: 0,
    });
  });
});
