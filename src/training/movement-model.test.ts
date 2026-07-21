import { describe, expect, it } from "vitest";
import { buildReplayManifest } from "../capture/replay.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import type { TrajectoryAction } from "../capture/trajectory.js";
import {
  DEFAULT_SYNTHETIC_WORKFLOWS,
  MarkovMovementBackend,
  type MovementTrainingDataset,
  evaluateMovementModel,
  generateSyntheticMovementDataset,
  loadMovementModel,
  movementDatasetFromExport,
  movementSequenceFromReplay,
  movementSequenceFromTrajectory,
  movementTokenFromAction,
  splitMovementDataset,
} from "./movement-model.js";
import type { ReviewedExportManifest } from "./export-manifest.js";

function action(tool: string, summary: string, metadata?: Record<string, unknown>): TrajectoryAction {
  return { kind: "action", tool, summary, ts: 0, metadata };
}

describe("movement tokenization", () => {
  it("uses gesture + target metadata for a canonical token", () => {
    const token = movementTokenFromAction(action("device", "tapped submit", { gesture: "tap", target: "Submit Button" }));
    expect(token).toBe("device:tap:submit-button");
  });

  it("falls back to the summary when no gesture metadata is present", () => {
    expect(movementTokenFromAction(action("browser", "Clicked the Login link"))).toBe("browser:clicked-the-login-link");
  });

  it("uses direction when there is no explicit target", () => {
    expect(movementTokenFromAction(action("device", "scrolled", { gesture: "scroll", direction: "down" }))).toBe(
      "device:scroll:down",
    );
  });

  it("derives a sequence from a trajectory's actions", () => {
    const trajectory = buildTrajectorySpan({
      id: "traj-1",
      sessionId: "sess-1",
      actions: [
        action("device", "tap", { gesture: "tap", target: "field" }),
        action("device", "type", { gesture: "type", target: "field" }),
      ],
    });
    expect(movementSequenceFromTrajectory(trajectory)).toEqual({
      id: "traj-1",
      tokens: ["device:tap:field", "device:type:field"],
    });
  });

  it("derives a sequence from a replay manifest, ignoring non-action events", () => {
    const trajectory = buildTrajectorySpan({
      id: "traj-2",
      sessionId: "sess-2",
      observations: [{ kind: "observation", source: "device", summary: "app active", ts: 1 }],
      actions: [action("device", "tapped send", { gesture: "tap", target: "send" })],
    });
    const replay = buildReplayManifest({ sessionId: "sess-2", transcript: [], trajectories: [trajectory] });
    // Replay timeline events carry only tool + summary (no gesture metadata),
    // so tokenization falls back to a slugged summary.
    expect(movementSequenceFromReplay(replay).tokens).toEqual(["device:tapped-send"]);
  });
});

describe("MarkovMovementBackend", () => {
  const backend = new MarkovMovementBackend();

  it("learns and reproduces a recorded movement sequence deterministically", async () => {
    const dataset: MovementTrainingDataset = {
      version: 1,
      sequences: [{ id: "s1", tokens: ["a", "b", "c", "d"] }],
    };
    const model = await backend.train(dataset, { order: 2 });
    expect(model.info.backend).toBe("markov-backoff");
    expect(model.info.vocabulary).toEqual(["a", "b", "c", "d"]);

    expect(model.predictNext(["a"]).token).toBe("b");
    expect(model.predictNext(["b"]).token).toBe("c");
    expect(model.predictNext(["a", "b"]).token).toBe("c");

    // Full rollout reproduces the recorded continuation.
    expect(model.generate(["a"], 5)).toEqual(["b", "c", "d"]);
  });

  it("picks the most frequent continuation and exposes ranked candidates", async () => {
    const dataset: MovementTrainingDataset = {
      version: 1,
      sequences: [
        { id: "s1", tokens: ["x", "y"] },
        { id: "s2", tokens: ["x", "y"] },
        { id: "s3", tokens: ["x", "z"] },
      ],
    };
    const model = await backend.train(dataset, { order: 1 });
    const prediction = model.predictNext(["x"]);
    expect(prediction.token).toBe("y");
    expect(prediction.probability).toBeCloseTo(2 / 3, 6);
    expect(prediction.candidates.map((c) => c.token)).toEqual(["y", "z"]);
    expect(prediction.matchedOrder).toBe(1);
  });

  it("backs off to a shorter context for unseen prefixes (generalization)", async () => {
    const dataset: MovementTrainingDataset = {
      version: 1,
      sequences: [
        { id: "s1", tokens: ["open", "click", "type", "submit"] },
        { id: "s2", tokens: ["launch", "click", "type", "submit"] },
      ],
    };
    const model = await backend.train(dataset, { order: 2 });
    // The bigram ("focus","click") was never seen, but backoff to unigram context
    // of "click" still predicts the learned next movement.
    const prediction = model.predictNext(["focus", "click"]);
    expect(prediction.token).toBe("type");
    expect(prediction.matchedOrder).toBeLessThan(2);
  });

  it("returns an empty prediction for an empty model", async () => {
    const model = await backend.train({ version: 1, sequences: [] });
    expect(model.predictNext(["anything"])).toEqual({
      token: undefined,
      probability: 0,
      candidates: [],
      matchedOrder: 0,
    });
    expect(model.generate(["seed"], 3)).toEqual([]);
  });

  it("round-trips through serialization", async () => {
    const dataset: MovementTrainingDataset = {
      version: 1,
      sequences: [{ id: "s1", tokens: ["a", "b", "c"] }],
    };
    const model = await backend.train(dataset, { order: 2 });
    const reloaded = loadMovementModel(model.serialize());
    expect(reloaded.info).toEqual(model.info);
    expect(reloaded.generate(["a"], 4)).toEqual(model.generate(["a"], 4));
    expect(reloaded.predictNext(["b"]).token).toBe("c");
  });

  it("caps generation length and does not loop forever on cycles", async () => {
    const model = await backend.train(
      { version: 1, sequences: [{ id: "loop", tokens: ["a", "b", "a", "b", "a"] }] },
      { order: 1 },
    );
    expect(model.generate(["a"], 3)).toHaveLength(3);
  });
});

describe("synthetic generation + generalization eval", () => {
  it("is deterministic for a given seed", () => {
    const a = generateSyntheticMovementDataset({ seed: 42, sequenceCount: 10 });
    const b = generateSyntheticMovementDataset({ seed: 42, sequenceCount: 10 });
    expect(a).toEqual(b);
    expect(a.sequences).toHaveLength(10);
    expect(a.sequences.every((s) => s.tokens.length > 0)).toBe(true);
  });

  it("produces different streams for different seeds", () => {
    const a = generateSyntheticMovementDataset({ seed: 1, sequenceCount: 20 });
    const b = generateSyntheticMovementDataset({ seed: 2, sequenceCount: 20 });
    expect(a).not.toEqual(b);
  });

  it("splits deterministically into disjoint train/test", () => {
    const dataset = generateSyntheticMovementDataset({ seed: 7, sequenceCount: 12 });
    const { train, test } = splitMovementDataset(dataset, 4);
    expect(train.sequences.length + test.sequences.length).toBe(12);
    const trainIds = new Set(train.sequences.map((s) => s.id));
    expect(test.sequences.some((s) => trainIds.has(s.id))).toBe(false);
  });

  it("generalizes to held-out but related movement sequences", async () => {
    const backend = new MarkovMovementBackend();
    const dataset = generateSyntheticMovementDataset({ seed: 123, sequenceCount: 120 });
    const { train, test } = splitMovementDataset(dataset, 4);
    const model = await backend.train(train, { order: 2 });

    const eval2 = evaluateMovementModel(model, test);
    // The workflows are structured, so a backoff model should reproduce the
    // large majority of next movements on unseen-but-related sequences.
    expect(eval2.predictedCount).toBeGreaterThan(0);
    expect(eval2.topOneAccuracy).toBeGreaterThan(0.8);
    expect(eval2.perSequence.length).toBe(test.sequences.length);
  });

  it("higher order never hurts train-set fidelity vs order 0", async () => {
    const backend = new MarkovMovementBackend();
    const dataset = generateSyntheticMovementDataset({ seed: 9, sequenceCount: 60 });
    const model0 = await backend.train(dataset, { order: 0 });
    const model2 = await backend.train(dataset, { order: 2 });
    const acc0 = evaluateMovementModel(model0, dataset).topOneAccuracy;
    const acc2 = evaluateMovementModel(model2, dataset).topOneAccuracy;
    expect(acc2).toBeGreaterThanOrEqual(acc0);
  });

  it("covers all default workflows across enough samples", () => {
    const dataset = generateSyntheticMovementDataset({ seed: 5, sequenceCount: 200 });
    const seenPrefixes = new Set(dataset.sequences.map((s) => s.id.replace(/-\d+$/, "")));
    for (const workflow of DEFAULT_SYNTHETIC_WORKFLOWS) {
      expect(seenPrefixes.has(workflow.name)).toBe(true);
    }
  });
});

describe("dataset from reviewed export", () => {
  it("builds a training dataset from a manifest's replays", () => {
    const manifest = {
      version: 1,
      createdAt: "2026-07-21T00:00:00Z",
      reviewedBy: "reviewer",
      purpose: "test",
      targetPlatform: "apple-silicon",
      modes: ["sft"],
      rawCaptureIncluded: false,
      promotedSkills: [],
      executableSkills: [],
      executableSkillRuns: [],
      memories: [],
      trajectories: [],
      replays: [
        {
          sessionId: "sess-1",
          trajectoryIds: ["traj-1"],
          eventCount: 3,
          events: [
            { kind: "observation", ts: 1, trajectoryId: "traj-1", source: "device", summary: "app active" },
            { kind: "action", ts: 2, trajectoryId: "traj-1", tool: "device", summary: "tapped compose" },
            { kind: "action", ts: 3, trajectoryId: "traj-1", tool: "device", summary: "tapped send" },
          ],
        },
      ],
    } satisfies ReviewedExportManifest;

    const dataset = movementDatasetFromExport(manifest);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]?.tokens).toEqual(["device:tapped-compose", "device:tapped-send"]);
  });
});
