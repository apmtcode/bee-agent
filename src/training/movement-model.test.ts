import { describe, expect, it } from "vitest";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import {
  MOVEMENT_END_TOKEN,
  createMarkovMovementBackend,
  datasetFromTrajectories,
  evaluateNextTokenAccuracy,
  evaluateReplayFidelity,
  generateSyntheticMovementDataset,
  splitMovementDataset,
  tokenizeAction,
  type MovementDataset,
} from "./movement-model.js";

function action(tool: string, summary: string, ts: number, metadata?: Record<string, unknown>) {
  return { kind: "action" as const, tool, summary, ts, ...(metadata ? { metadata } : {}) };
}

describe("movement tokenization", () => {
  it("prefers structured gesture metadata and folds in direction", () => {
    expect(tokenizeAction(action("device", "swiped up", 1, { gesture: "swipe", direction: "up" }))).toBe(
      "device:swipe-up",
    );
    expect(tokenizeAction(action("device", "tapped Submit", 2, { gesture: "tap" }))).toBe("device:tap");
  });

  it("falls back to the first summary word for free-form actions", () => {
    expect(tokenizeAction(action("editor", "Save the file", 1))).toBe("editor:save");
    expect(tokenizeAction(action("shell", "", 1))).toBe("shell:action");
  });

  it("builds a time-ordered dataset from trajectories, dropping empty ones", () => {
    const withActions = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [action("device", "b", 20, { gesture: "tap" }), action("device", "a", 10, { gesture: "type" })],
    });
    const empty = buildTrajectorySpan({ id: "t2", sessionId: "s1" });
    const dataset = datasetFromTrajectories([withActions, empty]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]).toEqual({ id: "t1", tokens: ["device:type", "device:tap"] });
  });
});

describe("MarkovMovementBackend — reproduce recorded movements", () => {
  it("regenerates a recorded movement exactly and stops at the end sentinel", () => {
    const backend = createMarkovMovementBackend({ maxOrder: 3 });
    const dataset: MovementDataset = { sequences: [{ id: "m", tokens: ["a", "b", "c", "d"] }] };
    const model = backend.train(dataset);

    expect(model.generate(["a"])).toEqual(["a", "b", "c", "d"]);

    const endPrediction = model.predictNext(["a", "b", "c", "d"]);
    expect(endPrediction?.token).toBe(MOVEMENT_END_TOKEN);
    expect(endPrediction?.isEnd).toBe(true);
  });

  it("is deterministic under ties (lexicographically smallest token wins)", () => {
    const backend = createMarkovMovementBackend({ maxOrder: 1 });
    const dataset: MovementDataset = {
      sequences: [
        { id: "1", tokens: ["start", "zebra"] },
        { id: "2", tokens: ["start", "apple"] },
      ],
    };
    const model = backend.train(dataset);
    // Both follow "start" once — tie broken toward "apple".
    expect(model.predictNext(["start"])?.token).toBe("apple");
  });
});

describe("MarkovMovementBackend — generalize via back-off", () => {
  it("predicts a sensible next token for an unseen higher-order prefix", () => {
    const backend = createMarkovMovementBackend({ maxOrder: 3 });
    const dataset: MovementDataset = {
      sequences: [
        { id: "1", tokens: ["open", "type", "save"] },
        { id: "2", tokens: ["click", "type", "save"] },
      ],
    };
    const model = backend.train(dataset);
    // "scroll type" was never seen; back-off to order-1 context "type" → "save".
    const prediction = model.predictNext(["scroll", "type"]);
    expect(prediction?.token).toBe("save");
    expect(prediction?.order).toBe(1);
  });

  it("learns a shared transition bias from synthetic data and beats chance on held-out movements", () => {
    const full = generateSyntheticMovementDataset({ seed: 7, count: 40 });
    const { train, heldOut } = splitMovementDataset(full, 4);
    expect(train.sequences.length).toBeGreaterThan(0);
    expect(heldOut.sequences.length).toBeGreaterThan(0);

    const model = createMarkovMovementBackend({ maxOrder: 3 }).train(train);
    const report = evaluateNextTokenAccuracy(model, heldOut);

    // The stream has 9 tokens; random guessing ≈ 0.11. The 65% adjacency bias is learnable.
    expect(report.totalPredictions).toBeGreaterThan(0);
    expect(report.accuracy).toBeGreaterThan(0.4);
  });
});

describe("evaluation harness", () => {
  it("scores replay fidelity: exact reproduction on trained sequences", () => {
    const dataset: MovementDataset = {
      sequences: [
        { id: "1", tokens: ["a", "b", "c"] },
        { id: "2", tokens: ["x", "y", "z"] },
      ],
    };
    const model = createMarkovMovementBackend({ maxOrder: 3 }).train(dataset);
    const report = evaluateReplayFidelity(model, dataset, { promptLength: 1 });
    expect(report.exactMatches).toBe(2);
    expect(report.averageOverlap).toBe(1);
  });

  it("reports zero accuracy for an empty model without throwing", () => {
    const model = createMarkovMovementBackend().train({ sequences: [] });
    const report = evaluateNextTokenAccuracy(model, { sequences: [{ id: "1", tokens: ["a", "b"] }] });
    expect(report.accuracy).toBe(0);
    expect(model.predictNext(["a"])).toBeUndefined();
    expect(model.generate(["a"])).toEqual(["a"]);
  });
});

describe("serialization", () => {
  it("round-trips a trained model through JSON with identical predictions", () => {
    const backend = createMarkovMovementBackend({ maxOrder: 2 });
    const dataset: MovementDataset = { sequences: [{ id: "1", tokens: ["a", "b", "c", "b", "c"] }] };
    const model = backend.train(dataset);
    const restored = backend.restore(JSON.parse(JSON.stringify(model.toJSON())));

    for (const context of [["a"], ["b"], ["a", "b"], ["b", "c"]]) {
      expect(restored.predictNext(context)?.token).toBe(model.predictNext(context)?.token);
    }
    expect(restored.maxOrder).toBe(2);
  });
});

describe("synthetic generator", () => {
  it("is deterministic for a fixed seed and varies with the seed", () => {
    const a = generateSyntheticMovementDataset({ seed: 3, count: 5 });
    const b = generateSyntheticMovementDataset({ seed: 3, count: 5 });
    const c = generateSyntheticMovementDataset({ seed: 4, count: 5 });
    expect(a).toEqual(b);
    expect(a).not.toEqual(c);
    expect(a.sequences).toHaveLength(5);
    for (const sequence of a.sequences) {
      expect(sequence.tokens.length).toBeGreaterThanOrEqual(3);
      expect(sequence.tokens.length).toBeLessThanOrEqual(7);
    }
  });
});
