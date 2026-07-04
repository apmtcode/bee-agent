import { describe, expect, it } from "vitest";
import { buildReplayManifest } from "../capture/replay.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import {
  MarkovMovementBackend,
  evaluateReplayFidelity,
  tokenizeStep,
} from "./movement-model.js";
import {
  buildMovementDatasetFromReplays,
  buildMovementDatasetFromTrajectories,
  generateSyntheticMovementDataset,
  movementSequenceFromTrajectory,
} from "./movement-dataset.js";

describe("movement dataset builders", () => {
  it("derives movement steps from a replay manifest's action events", () => {
    const replay = buildReplayManifest({
      sessionId: "sess-1",
      transcript: [],
      trajectories: [
        buildTrajectorySpan({
          id: "traj-1",
          sessionId: "sess-1",
          actions: [
            { kind: "action", tool: "device", summary: "tapped submit-button", ts: 2 },
            { kind: "action", tool: "device", summary: "swiped down", ts: 3 },
          ],
        }),
      ],
    });

    const dataset = buildMovementDatasetFromReplays([replay]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]!.steps).toEqual([
      { tool: "device", action: "tapped", target: "submit-button" },
      { tool: "device", action: "swiped", target: "down" },
    ]);
  });

  it("prefers structured metadata when building from a trajectory", () => {
    const trajectory = buildTrajectorySpan({
      id: "traj-2",
      sessionId: "sess-2",
      actions: [
        {
          kind: "action",
          tool: "device",
          summary: "swiped up",
          ts: 5,
          metadata: { gesture: "swipe", direction: "up", target: "feed" },
        },
      ],
    });

    const sequence = movementSequenceFromTrajectory(trajectory);
    expect(sequence.steps[0]).toEqual({ tool: "device", action: "swipe", target: "feed", direction: "up" });
  });

  it("orders trajectory actions by timestamp", () => {
    const trajectory = buildTrajectorySpan({
      id: "traj-3",
      sessionId: "sess-3",
      actions: [
        { kind: "action", tool: "mouse", summary: "clicked second", ts: 20 },
        { kind: "action", tool: "mouse", summary: "clicked first", ts: 10 },
      ],
    });
    const dataset = buildMovementDatasetFromTrajectories([trajectory]);
    expect(dataset.sequences[0]!.steps.map((s) => s.target)).toEqual(["first", "second"]);
  });
});

describe("synthetic movement generator", () => {
  it("is deterministic for a given seed", () => {
    const a = generateSyntheticMovementDataset({ sequenceCount: 6, seed: 42 });
    const b = generateSyntheticMovementDataset({ sequenceCount: 6, seed: 42 });
    expect(a).toEqual(b);
    expect(a.sequences).toHaveLength(6);
  });

  it("drives a full capture->dataset->train->replay round-trip in the cloud", async () => {
    // Train on synthetic sequences, then measure replay fidelity on the same
    // flows (repetition) — the pipeline works end-to-end with no OS input.
    const dataset = generateSyntheticMovementDataset({ sequenceCount: 12, seed: 7 });
    const model = await new MarkovMovementBackend().train(dataset, { order: 3 });

    // De-duplicate flows for the held-out eval (synthetic flows repeat).
    const unique = new Map(dataset.sequences.map((s) => [s.steps.map(tokenizeStep).join(","), s]));
    const report = evaluateReplayFidelity(model, [...unique.values()], { seedSteps: 1 });
    expect(report.stepAccuracy).toBe(1);
    expect(report.exactSequenceRate).toBe(1);
  });
});
