import { describe, expect, it } from "vitest";
import { buildReplayManifest } from "../capture/replay.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import {
  buildMovementDatasetFromExport,
  buildMovementDatasetFromReplays,
  buildMovementDatasetFromTrajectories,
  tokenizeReplayEvent,
  trainMovementModelFromExport,
} from "./movement-dataset.js";
import { NGramMovementModelBackend, evaluateNextTokenAccuracy } from "./movement-model.js";
import { generateSyntheticReplayStream } from "./synthetic-movements.js";

describe("movement dataset tokenization", () => {
  it("tokenizes actions and observations into coarse, shareable symbols", () => {
    expect(
      tokenizeReplayEvent({ kind: "action", ts: 1, trajectoryId: "t", tool: "device", summary: "tapped Sign In" }),
    ).toBe("act:device:tapped-sign");
    expect(
      tokenizeReplayEvent({ kind: "observation", ts: 1, trajectoryId: "t", source: "os", summary: "opened Dashboard!" }),
    ).toBe("obs:os:opened-dashboard");
  });

  it("excludes observations and transcript unless opted in", () => {
    expect(
      tokenizeReplayEvent(
        { kind: "observation", ts: 1, trajectoryId: "t", source: "os", summary: "x" },
        { includeObservations: false },
      ),
    ).toBeUndefined();
    expect(
      tokenizeReplayEvent({ kind: "transcript", ts: 1, messageId: "m", role: "user", content: "hi" }),
    ).toBeUndefined();
    expect(
      tokenizeReplayEvent({ kind: "transcript", ts: 1, messageId: "m", role: "user", content: "hi" }, { includeTranscript: true }),
    ).toBe("msg:user");
  });

  it("builds a sequence from trajectory spans ordered by timestamp", () => {
    const trajectory = buildTrajectorySpan({
      id: "traj-1",
      sessionId: "s1",
      observations: [{ kind: "observation", source: "os", summary: "focused window", ts: 10 }],
      actions: [
        { kind: "action", tool: "device", summary: "typed body", ts: 30 },
        { kind: "action", tool: "device", summary: "tapped compose", ts: 20 },
      ],
    });
    const [sequence] = buildMovementDatasetFromTrajectories([trajectory]);
    expect(sequence?.tokens).toEqual([
      "obs:os:focused-window",
      "act:device:tapped-compose",
      "act:device:typed-body",
    ]);
  });

  it("round-trips a replay manifest into a trainable dataset", () => {
    const manifest = buildReplayManifest({
      sessionId: "s1",
      transcript: [],
      trajectories: [
        buildTrajectorySpan({
          id: "traj-1",
          sessionId: "s1",
          observations: [{ kind: "observation", source: "os", summary: "focused mail", ts: 1 }],
          actions: [{ kind: "action", tool: "device", summary: "tapped send", ts: 2 }],
        }),
      ],
    });
    const dataset = buildMovementDatasetFromReplays([manifest]);
    expect(dataset).toHaveLength(1);
    expect(dataset[0]?.tokens).toContain("act:device:tapped-send");
  });

  it("learns synthetic flows and generalizes to held-out instances", async () => {
    // Disjoint train / eval streams (different seeds) of the same flow library.
    const trainStream = generateSyntheticReplayStream({ seed: 7, count: 40 });
    const evalStream = generateSyntheticReplayStream({ seed: 999, count: 20 });

    const trainDataset = buildMovementDatasetFromReplays(trainStream);
    const evalDataset = buildMovementDatasetFromReplays(evalStream);

    const { model } = await trainMovementModelFromExport({ replays: [] }, {}); // sanity: empty export trains
    expect(model.stats.sequenceCount).toBe(0);

    const trained = await new NGramMovementModelBackend().train({ sequences: trainDataset, order: 4 });

    // Held-out instances are new recordings of the same flows -> should replay
    // near-perfectly through back-off/generalization.
    const result = evaluateNextTokenAccuracy(trained, evalDataset);
    expect(result.predictions).toBeGreaterThan(0);
    expect(result.accuracy).toBeGreaterThan(0.9);
  });

  it("trains end-to-end from an exported manifest's replays", async () => {
    const stream = generateSyntheticReplayStream({ seed: 3, count: 5 });
    const replays = stream.map((manifest) => ({
      sessionId: manifest.sessionId,
      trajectoryIds: manifest.trajectoryIds,
      eventCount: manifest.eventCount,
      events: manifest.events,
    }));
    const dataset = buildMovementDatasetFromExport({ replays });
    expect(dataset.length).toBe(5);

    const { model, dataset: trainedOn } = await trainMovementModelFromExport({ replays });
    expect(trainedOn.length).toBe(5);
    expect(model.stats.vocabularySize).toBeGreaterThan(0);
  });
});
