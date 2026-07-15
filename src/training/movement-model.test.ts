import { describe, expect, it } from "vitest";
import type { ReviewedExportManifest } from "./export-manifest.js";
import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  DEFAULT_MOVEMENT_ORDER,
  NGramMovementBackend,
  buildMovementDataset,
  buildMovementDatasetFromManifest,
  buildMovementSequenceFromTrajectory,
  evaluateMovementModel,
  movementActionToken,
  normalizeMovementLabel,
  tokenizeMovementEvents,
  type MovementSequence,
} from "./movement-model.js";

function seq(sequenceId: string, tokens: string[]): MovementSequence {
  return { sequenceId, tokens };
}

// A small synthetic "open editor → type → save" motor routine, repeated with a
// couple of variants so the model has statistics to learn.
const OPEN = movementActionToken("device", "tapped editor-icon");
const TYPE = movementActionToken("device", "typed into document");
const SAVE = movementActionToken("device", "triggered save-shortcut");
const CLOSE = movementActionToken("device", "tapped close-button");

const TRAIN_SEQUENCES: MovementSequence[] = [
  seq("t1", [OPEN, TYPE, SAVE, CLOSE]),
  seq("t2", [OPEN, TYPE, SAVE, CLOSE]),
  seq("t3", [OPEN, TYPE, TYPE, SAVE, CLOSE]),
];

describe("normalizeMovementLabel + tokenization", () => {
  it("produces stable, generalization-friendly tokens", () => {
    expect(normalizeMovementLabel("  Tapped  Submit Button! ")).toBe("tapped-submit-button");
    expect(movementActionToken("Device", "Tapped Submit")).toBe("action:device:tapped-submit");
  });

  it("keeps only action events by default and preserves order", () => {
    const tokens = tokenizeMovementEvents([
      { kind: "observation", ts: 1, trajectoryId: "x", source: "device", summary: "editor open" },
      { kind: "action", ts: 2, trajectoryId: "x", tool: "device", summary: "tapped editor-icon" },
      { kind: "transcript", ts: 3, messageId: "m", role: "user", content: "hi" },
      { kind: "action", ts: 4, trajectoryId: "x", tool: "device", summary: "typed into document" },
    ]);
    expect(tokens).toEqual([OPEN, TYPE]);
  });

  it("can include observation-context tokens when asked", () => {
    const tokens = tokenizeMovementEvents(
      [{ kind: "observation", ts: 1, trajectoryId: "x", source: "device", summary: "editor open" }],
      { include: "all" },
    );
    expect(tokens).toEqual(["obs:device:editor-open"]);
  });

  it("builds an ordered sequence from a trajectory's actions", () => {
    const trajectory: TrajectorySpan = {
      id: "traj-1",
      sessionId: "s1",
      createdAt: "2026-07-15T00:00:00.000Z",
      captureTier: "full",
      observations: [],
      actions: [
        { kind: "action", tool: "device", summary: "triggered save-shortcut", ts: 30 },
        { kind: "action", tool: "device", summary: "tapped editor-icon", ts: 10 },
        { kind: "action", tool: "device", summary: "typed into document", ts: 20 },
      ],
    };
    const built = buildMovementSequenceFromTrajectory(trajectory);
    expect(built).toEqual(seq("traj-1", [OPEN, TYPE, SAVE]));
  });
});

describe("NGramMovementBackend training + repeat", () => {
  it("learns to repeat the dominant recorded movement (deterministic top-1)", async () => {
    const backend = new NGramMovementBackend();
    const model = await backend.train(buildMovementDataset(TRAIN_SEQUENCES));

    // After OPEN, the recorded routine always types.
    expect(model.predict([OPEN]).token).toBe(TYPE);
    // After OPEN, TYPE, SAVE the routine always closes.
    expect(model.predict([OPEN, TYPE, SAVE]).token).toBe(CLOSE);
    // Exact context match — no backoff needed.
    expect(model.predict([OPEN, TYPE, SAVE]).backedOff).toBe(false);
  });

  it("is fully deterministic across repeated training runs", async () => {
    const backend = new NGramMovementBackend();
    const a = (await backend.train(buildMovementDataset(TRAIN_SEQUENCES))).serialize();
    const b = (await backend.train(buildMovementDataset(TRAIN_SEQUENCES))).serialize();
    expect(a).toEqual(b);
  });

  it("returns ranked candidates with probabilities", async () => {
    const backend = new NGramMovementBackend();
    const model = await backend.train(buildMovementDataset(TRAIN_SEQUENCES), { order: 1 });
    // After a single TYPE (order 1): sometimes TYPE again (t3), sometimes SAVE.
    const prediction = model.predict([TYPE], { topK: 5 });
    const total = prediction.candidates.reduce((sum, candidate) => sum + candidate.probability, 0);
    expect(total).toBeCloseTo(1, 5);
    expect(prediction.candidates.map((candidate) => candidate.token).sort()).toEqual([SAVE, TYPE].sort());
  });
});

describe("generalization via stupid-backoff", () => {
  it("predicts a related-but-unseen context by falling back to a shorter suffix", async () => {
    const backend = new NGramMovementBackend();
    const model = await backend.train(buildMovementDataset(TRAIN_SEQUENCES));

    // This exact 3-token context never appeared in training, but its suffix
    // [TYPE, SAVE] did → the model still predicts the sensible next movement.
    const novelContext = [CLOSE, TYPE, SAVE];
    const prediction = model.predict(novelContext);
    expect(prediction.token).toBe(CLOSE);
    expect(prediction.backedOff).toBe(true);
    expect(prediction.matchedOrder).toBeLessThan(Math.min(novelContext.length, DEFAULT_MOVEMENT_ORDER));
  });

  it("falls all the way back to the unigram distribution for a fully novel context", async () => {
    const backend = new NGramMovementBackend();
    const model = await backend.train(buildMovementDataset(TRAIN_SEQUENCES));
    const prediction = model.predict(["action:device:unknown-gesture"]);
    // Most frequent token overall is TYPE (appears 4×).
    expect(prediction.token).toBe(TYPE);
    expect(prediction.matchedOrder).toBe(0);
    expect(prediction.backedOff).toBe(true);
  });

  it("returns an empty prediction from an untrained model", async () => {
    const backend = new NGramMovementBackend();
    const model = await backend.train(buildMovementDataset([]));
    expect(model.predict([OPEN])).toEqual({
      token: undefined,
      candidates: [],
      matchedOrder: 0,
      backedOff: true,
    });
  });
});

describe("serialize / load round-trip", () => {
  it("reloads to an identical predictor", async () => {
    const backend = new NGramMovementBackend();
    const model = await backend.train(buildMovementDataset(TRAIN_SEQUENCES));
    const artifact = model.serialize();

    // Artifact must be JSON-serializable (pluggable backend seam).
    const roundTripped = backend.load(JSON.parse(JSON.stringify(artifact)));
    expect(roundTripped.serialize()).toEqual(artifact);

    for (const context of [[OPEN], [OPEN, TYPE], [OPEN, TYPE, SAVE], [CLOSE, TYPE, SAVE]]) {
      expect(roundTripped.predict(context)).toEqual(model.predict(context));
    }
  });
});

describe("evaluateMovementModel", () => {
  it("scores perfectly on held-out copies of the trained routine", async () => {
    const backend = new NGramMovementBackend();
    const model = await backend.train(buildMovementDataset(TRAIN_SEQUENCES));
    const result = evaluateMovementModel(model, [seq("held", [OPEN, TYPE, SAVE, CLOSE])], {
      skipFirstToken: true,
    });
    expect(result.predictionCount).toBe(3);
    expect(result.top1Accuracy).toBe(1);
    expect(result.topKAccuracy).toBe(1);
    expect(result.backoffCount).toBe(0);
  });

  it("reports lower fidelity and backoff on a divergent held-out routine", async () => {
    const backend = new NGramMovementBackend();
    const model = await backend.train(buildMovementDataset(TRAIN_SEQUENCES));
    const divergent = seq("held", [CLOSE, OPEN, "action:device:unknown-gesture", SAVE]);
    const result = evaluateMovementModel(model, [divergent], { skipFirstToken: true });
    expect(result.predictionCount).toBe(3);
    expect(result.top1Accuracy).toBeLessThan(1);
    expect(result.backoffCount).toBeGreaterThan(0);
    expect(result.backoffRate).toBeGreaterThan(0);
  });

  it("returns zeroed metrics for empty held-out input", async () => {
    const backend = new NGramMovementBackend();
    const model = await backend.train(buildMovementDataset(TRAIN_SEQUENCES));
    const result = evaluateMovementModel(model, []);
    expect(result.predictionCount).toBe(0);
    expect(result.top1Accuracy).toBe(0);
    expect(result.backoffRate).toBe(0);
  });
});

describe("buildMovementDatasetFromManifest", () => {
  it("derives one training sequence per reviewed replay bundle", () => {
    const manifest: ReviewedExportManifest = {
      version: 1,
      createdAt: "2026-07-15T00:00:00.000Z",
      reviewedBy: "reviewer",
      purpose: "movement-training",
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
          sessionId: "s1",
          trajectoryIds: ["traj-1"],
          eventCount: 3,
          events: [
            { kind: "observation", ts: 1, trajectoryId: "traj-1", source: "device", summary: "editor open" },
            { kind: "action", ts: 2, trajectoryId: "traj-1", tool: "device", summary: "tapped editor-icon" },
            { kind: "action", ts: 3, trajectoryId: "traj-1", tool: "device", summary: "typed into document" },
          ],
        },
      ],
    };
    const dataset = buildMovementDatasetFromManifest(manifest);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]).toEqual(seq("traj-1", [OPEN, TYPE]));
  });
});
