import { describe, expect, it } from "vitest";
import {
  buildMovementDataset,
  evaluateMovementGeneralization,
  MarkovMovementBackend,
  MOVEMENT_END_TOKEN,
  tokenizeTrajectory,
  trainMovementModelFromTrajectories,
  type MovementSequence,
} from "./movement-model.js";
import {
  createSeededRng,
  DEFAULT_MOVEMENT_MOTIFS,
  generateSyntheticMovementTrajectories,
} from "./synthetic-movements.js";
import { buildTrajectorySpan, type TrajectoryAction } from "../capture/trajectory.js";

function deviceAction(gesture: string, ts: number, extra: Record<string, unknown> = {}): TrajectoryAction {
  return {
    kind: "action",
    tool: "device",
    summary: gesture,
    ts,
    metadata: { gesture, ...extra },
  };
}

describe("movement tokenization + dataset", () => {
  it("tokenizes gestures into direction-aware movement tokens in ts order", () => {
    const trajectory = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [
        deviceAction("tap", 300, { target: "row" }),
        deviceAction("scroll", 100, { direction: "down" }),
        deviceAction("swipe", 200, { direction: "left" }),
      ],
    });
    const sequence = tokenizeTrajectory(trajectory);
    expect(sequence.tokens).toEqual(["scroll:down", "swipe:left", "tap"]);
  });

  it("builds a vocabulary that always includes the end token and skips empty trajectories", () => {
    const withActions = buildTrajectorySpan({ id: "a", sessionId: "s", actions: [deviceAction("tap", 1)] });
    const empty = buildTrajectorySpan({ id: "b", sessionId: "s", actions: [] });
    const dataset = buildMovementDataset([withActions, empty]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.vocabulary).toContain(MOVEMENT_END_TOKEN);
    expect(dataset.vocabulary).toContain("tap");
  });
});

describe("MarkovMovementBackend training + repeat", () => {
  it("greedily reproduces a memorized movement pattern", async () => {
    const trajectories = Array.from({ length: 5 }, (_unused, index) =>
      buildTrajectorySpan({
        id: `t${index}`,
        sessionId: "s",
        actions: [
          deviceAction("scroll", 1, { direction: "down" }),
          deviceAction("tap", 2, { target: "row" }),
          deviceAction("type", 3, { target: "field" }),
        ],
      }),
    );
    const { model } = await trainMovementModelFromTrajectories(trajectories, { order: 2 });
    // Greedy generation from an empty context should replay the learned pattern.
    expect(model.generate()).toEqual(["scroll:down", "tap", "type"]);
    // Next-token prediction is deterministic and correct.
    expect(model.predictNext(["scroll:down"])?.token).toBe("tap");
    expect(model.predictNext(["scroll:down", "tap"])?.token).toBe("type");
  });

  it("serializes and round-trips deterministically", async () => {
    const dataset = buildMovementDataset(
      generateSyntheticMovementTrajectories({ count: 6, seed: 7 }),
    );
    const backend = new MarkovMovementBackend();
    const a = await backend.train(dataset, { order: 2 });
    const b = await backend.train(dataset, { order: 2 });
    expect(a.toJSON()).toEqual(b.toJSON());
    expect(a.generate()).toEqual(b.generate());
  });

  it("samples reproducibly under a seeded rng", async () => {
    const { model } = await trainMovementModelFromTrajectories(
      generateSyntheticMovementTrajectories({ count: 20, seed: 3 }),
      { order: 2 },
    );
    const first = model.generate({ rng: createSeededRng(99), maxLength: 12 });
    const second = model.generate({ rng: createSeededRng(99), maxLength: 12 });
    expect(first).toEqual(second);
    expect(first.length).toBeGreaterThan(0);
  });
});

describe("generalization", () => {
  it("predicts held-out-but-related movements via back-off", async () => {
    // Train on two motifs, hold out a third distinct motif; the shared vocabulary
    // and lower-order stats should still yield above-chance next-token accuracy.
    const trainMotifs = DEFAULT_MOVEMENT_MOTIFS.slice(0, 2);
    const trajectories = generateSyntheticMovementTrajectories({
      count: 40,
      seed: 11,
      motifs: trainMotifs,
      variation: 0.3,
    });
    const { model } = await trainMovementModelFromTrajectories(trajectories, { order: 2 });

    // Held-out sequences: same two motifs, different seed => unseen exact orders.
    const heldOutTrajectories = generateSyntheticMovementTrajectories({
      count: 10,
      seed: 999,
      motifs: trainMotifs,
      variation: 0.5,
    });
    const heldOut: MovementSequence[] = heldOutTrajectories.map((trajectory) => tokenizeTrajectory(trajectory));
    const report = evaluateMovementGeneralization(model, heldOut);

    const dataset = buildMovementDataset(trajectories);
    const chanceAccuracy = 1 / dataset.vocabulary.length;

    expect(report.tokenCount).toBeGreaterThan(0);
    // Generalization: next-token accuracy on unseen-but-related sequences must
    // be well above uniform chance over the vocabulary.
    expect(report.nextTokenAccuracy).toBeGreaterThan(chanceAccuracy * 2);
    expect(Number.isFinite(report.perplexity)).toBe(true);
    expect(report.averageLikelihood).toBeGreaterThan(0);
  });

  it("assigns a novel context a finite, non-zero probability (no memorization required)", async () => {
    const { model } = await trainMovementModelFromTrajectories(
      generateSyntheticMovementTrajectories({ count: 12, seed: 5 }),
      { order: 2 },
    );
    // A context the model never saw at full order still resolves via back-off.
    const probability = model.conditionalProbability(["swipe:left", "tap", "type"], "tap");
    expect(probability).toBeGreaterThan(0);
    expect(probability).toBeLessThanOrEqual(1);
  });
});
