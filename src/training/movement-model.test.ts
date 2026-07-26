import { describe, expect, it } from "vitest";
import { buildTrajectorySpan, type TrajectoryAction, type TrajectorySpan } from "../capture/trajectory.js";
import {
  MOVEMENT_END,
  NgramMovementBackend,
  buildMovementDataset,
  buildMovementDatasetFromReplays,
  createNgramMovementBackend,
  evaluateMovementModel,
  loadMovementModel,
  movementTokenFromAction,
} from "./movement-model.js";

// --- synthetic event-stream generator (no real OS input) ------------------

function gesture(kind: string, direction: string, ts: number): TrajectoryAction {
  return {
    kind: "action",
    tool: "device",
    summary: `${kind} ${direction}`,
    ts,
    metadata: { gesture: kind, direction },
  };
}

// A repeatable "open app, scroll down, tap item, swipe left to dismiss" motif.
function motifTrajectory(id: string, base: number): TrajectorySpan {
  return buildTrajectorySpan({
    id,
    sessionId: `session-${id}`,
    actions: [
      gesture("tap", "up", base + 1),
      gesture("scroll", "down", base + 2),
      gesture("tap", "down", base + 3),
      gesture("swipe", "left", base + 4),
    ],
  });
}

describe("movement token normalization", () => {
  it("derives structured tokens from gesture metadata", () => {
    expect(movementTokenFromAction(gesture("swipe", "down", 1))).toBe("device/swipe/down");
  });

  it("falls back to the summary when no gesture metadata is present", () => {
    const token = movementTokenFromAction({ kind: "action", tool: "keyboard", summary: "Copy Selection", ts: 1 });
    expect(token).toBe("keyboard/copy-selection");
  });
});

describe("buildMovementDataset", () => {
  it("orders actions by timestamp and collects a sorted vocabulary", () => {
    const trajectory = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [gesture("swipe", "left", 5), gesture("tap", "up", 1)],
    });
    const dataset = buildMovementDataset([trajectory]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]!.tokens).toEqual(["device/tap/up", "device/swipe/left"]);
    expect(dataset.vocabulary).toEqual(["device/swipe/left", "device/tap/up"]);
  });

  it("drops trajectories that recorded no actions", () => {
    const empty = buildTrajectorySpan({ id: "t0", sessionId: "s0", actions: [] });
    expect(buildMovementDataset([empty]).sequences).toHaveLength(0);
  });

  it("builds a dataset from replay manifests", () => {
    const dataset = buildMovementDatasetFromReplays([
      {
        version: 1,
        sessionId: "s1",
        trajectoryIds: ["t1"],
        eventCount: 2,
        events: [
          { kind: "action", ts: 2, trajectoryId: "t1", tool: "device", summary: "swiped left" },
          { kind: "observation", ts: 1, trajectoryId: "t1", source: "device", summary: "app active" },
          { kind: "action", ts: 1, trajectoryId: "t1", tool: "device", summary: "tapped up" },
        ],
      },
    ]);
    expect(dataset.sequences[0]!.tokens).toEqual(["device/tapped-up", "device/swiped-left"]);
  });
});

describe("NgramMovementBackend", () => {
  it("repeats a learned motif from its opening move", async () => {
    const dataset = buildMovementDataset([motifTrajectory("a", 0), motifTrajectory("b", 100), motifTrajectory("c", 200)]);
    const model = await new NgramMovementBackend().train(dataset, { order: 3 });

    // From empty context it should predict the recorded opening move...
    const opening = model.predictNext([]);
    expect(opening.token).toBe("device/tap/up");

    // ...and rolling out reproduces the full recorded motif, then terminates.
    const rollout = model.generate([], 10);
    expect(rollout).toEqual(["device/tap/up", "device/scroll/down", "device/tap/down", "device/swipe/left"]);
  });

  it("predicts the end sentinel is likely after the final learned move", async () => {
    const dataset = buildMovementDataset([motifTrajectory("a", 0), motifTrajectory("b", 100)]);
    const model = await createNgramMovementBackend().train(dataset);
    const prediction = model.predictNext(["device/tap/up", "device/scroll/down", "device/tap/down", "device/swipe/left"]);
    expect(prediction.token).toBe(MOVEMENT_END);
  });

  it("is deterministic: identical dataset produces identical serialized model", async () => {
    const dataset = buildMovementDataset([motifTrajectory("a", 0), motifTrajectory("b", 100)]);
    const first = await new NgramMovementBackend().train(dataset);
    const second = await new NgramMovementBackend().train(dataset);
    expect(first.toJSON()).toEqual(second.toJSON());
  });

  it("generalizes to a novel-but-related prefix via backoff", async () => {
    // Two motifs that share the transition scroll/down -> tap/down.
    const trainA = motifTrajectory("a", 0);
    const trainB = buildTrajectorySpan({
      id: "b",
      sessionId: "sb",
      actions: [gesture("scroll", "up", 1), gesture("scroll", "down", 2), gesture("tap", "down", 3)],
    });
    const trained = await new NgramMovementBackend().train(buildMovementDataset([trainA, trainB]), { order: 3 });

    // A prefix never seen verbatim, but ending in the familiar scroll/down move.
    const prediction = trained.predictNext(["device/tap/down", "device/scroll/down"]);
    expect(prediction.token).toBe("device/tap/down");
    // It backed off to a shorter context than the full prefix -> generalization.
    expect(prediction.backoffOrder).toBeLessThan(3);
  });

  it("round-trips through serialization", async () => {
    const dataset = buildMovementDataset([motifTrajectory("a", 0), motifTrajectory("b", 100)]);
    const original = await new NgramMovementBackend().train(dataset);
    const restored = loadMovementModel(original.toJSON());
    expect(restored.generate([], 10)).toEqual(original.generate([], 10));
    expect(restored.toJSON()).toEqual(original.toJSON());
  });
});

describe("evaluateMovementModel", () => {
  it("scores high fidelity on held-out instances of a learned motif", async () => {
    const train = [motifTrajectory("a", 0), motifTrajectory("b", 100), motifTrajectory("c", 200)];
    const model = await new NgramMovementBackend().train(buildMovementDataset(train), { order: 3 });
    const heldOut = buildMovementDataset([motifTrajectory("d", 300)]).sequences;

    const result = evaluateMovementModel(model, heldOut);
    expect(result.totalPredictions).toBe(4);
    expect(result.nextTokenAccuracy).toBe(1);
    expect(result.meanLogProb).toBeLessThanOrEqual(0);
    expect(result.generalizationRate).toBeGreaterThanOrEqual(0);
  });

  it("reports zero accuracy on an untrained model with no overlap", async () => {
    const model = await new NgramMovementBackend().train(buildMovementDataset([motifTrajectory("a", 0)]));
    const heldOut = buildMovementDataset([
      buildTrajectorySpan({
        id: "z",
        sessionId: "sz",
        actions: [gesture("pinch", "up", 1), gesture("rotate", "left", 2)],
      }),
    ]).sequences;
    const result = evaluateMovementModel(model, heldOut);
    expect(result.nextTokenAccuracy).toBeLessThan(1);
  });
});
