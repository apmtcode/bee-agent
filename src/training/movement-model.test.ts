import { describe, expect, it } from "vitest";
import { buildTrajectorySpan, type TrajectoryAction } from "../capture/trajectory.js";
import type { ReplayManifest } from "../capture/replay.js";
import {
  MarkovMovementBackend,
  buildMovementDataset,
  evaluateNextTokenAccuracy,
  movementSequenceFromReplay,
  movementSequenceFromTrajectory,
  movementTokenFromAction,
  type MovementSequence,
} from "./movement-model.js";

function action(tool: string, summary: string, ts: number): TrajectoryAction {
  return { kind: "action", tool, summary, ts };
}

/**
 * Synthetic movement grammar: a "save file" workflow expressed as ordered UI
 * movements. Deterministic — no clock/RNG — so training + inference are
 * reproducible in the cloud without any real OS input.
 */
function saveWorkflow(index: number): MovementSequence {
  return {
    id: `save-${index}`,
    tokens: [
      "focus:window.editor",
      "click:menu.file",
      "click:menu.file.save",
      "type:dialog.filename",
      "click:button.confirm",
    ],
  };
}

describe("MarkovMovementBackend", () => {
  const backend = new MarkovMovementBackend();

  it("trains a serializable artifact from recorded sequences", async () => {
    const dataset = buildMovementDataset([saveWorkflow(0), saveWorkflow(1)]);
    const artifact = await backend.train(dataset, { order: 3 });

    expect(artifact.backend).toBe("markov");
    expect(artifact.order).toBe(3);
    expect(artifact.sequenceCount).toBe(2);
    expect(artifact.tokenCount).toBe(10);
    expect(artifact.vocabulary).toEqual([
      "click:button.confirm",
      "click:menu.file",
      "click:menu.file.save",
      "focus:window.editor",
      "type:dialog.filename",
    ]);
    // Artifact must survive a JSON round-trip (persisted next to manifests).
    expect(JSON.parse(JSON.stringify(artifact))).toEqual(artifact);
  });

  it("repeats a recorded movement exactly from its seed (objective 2c)", async () => {
    const dataset = buildMovementDataset([saveWorkflow(0), saveWorkflow(1)]);
    const artifact = await backend.train(dataset);

    const rollout = backend.generate(artifact, {
      seed: ["focus:window.editor"],
      stopToken: "click:button.confirm",
      maxSteps: 10,
    });

    expect(rollout).toEqual([
      "click:menu.file",
      "click:menu.file.save",
      "type:dialog.filename",
      "click:button.confirm",
    ]);
  });

  it("is deterministic across repeated inference", async () => {
    const artifact = await backend.train(buildMovementDataset([saveWorkflow(0)]));
    const first = backend.generate(artifact, { seed: ["focus:window.editor"], maxSteps: 6 });
    const second = backend.generate(artifact, { seed: ["focus:window.editor"], maxSteps: 6 });
    expect(first).toEqual(second);
  });

  it("predicts the argmax continuation with a confidence and backoff order", async () => {
    const artifact = await backend.train(buildMovementDataset([saveWorkflow(0), saveWorkflow(1)]));
    const prediction = backend.predictNext(artifact, ["click:menu.file"]);
    expect(prediction.token).toBe("click:menu.file.save");
    expect(prediction.confidence).toBeCloseTo(1);
    expect(prediction.backoffOrder).toBe(1);
  });

  it("generalizes to a new-but-related context via backoff (objective 2d)", async () => {
    // Train on the save workflow only.
    const artifact = await backend.train(buildMovementDataset([saveWorkflow(0), saveWorkflow(1)]));

    // Unseen full context: a different prefix ("focus:window.terminal") that
    // still ends in a learned sub-movement. The model has never seen this exact
    // 2-gram, so it must back off to the shorter shared suffix.
    const prediction = backend.predictNext(artifact, ["focus:window.terminal", "click:menu.file"]);
    expect(prediction.token).toBe("click:menu.file.save");
    expect(prediction.backoffOrder).toBeLessThan(2);
  });

  it("returns an empty rollout for an untrained model", async () => {
    const artifact = await backend.train(buildMovementDataset([]));
    expect(artifact.tokenCount).toBe(0);
    expect(backend.generate(artifact, { seed: ["anything"] })).toEqual([]);
    expect(backend.predictNext(artifact, ["anything"]).token).toBeUndefined();
  });
});

describe("movement dataset bridges", () => {
  it("derives an ordered movement sequence from a trajectory span", () => {
    const trajectory = buildTrajectorySpan({
      id: "traj-1",
      sessionId: "sess-1",
      actions: [
        action("mouse", "click:button.confirm", 3),
        action("mouse", "click:menu.file", 1),
        action("mouse", "click:menu.file.save", 2),
      ],
    });
    const sequence = movementSequenceFromTrajectory(trajectory);
    expect(sequence.id).toBe("traj-1");
    // Sorted by timestamp, then normalized to `tool:summary` tokens.
    expect(sequence.tokens).toEqual([
      "mouse:click:menu.file",
      "mouse:click:menu.file.save",
      "mouse:click:button.confirm",
    ]);
  });

  it("derives a movement sequence from a replay manifest's actions", () => {
    const manifest: ReplayManifest = {
      version: 1,
      sessionId: "sess-2",
      trajectoryIds: ["traj-2"],
      eventCount: 3,
      events: [
        { kind: "observation", ts: 1, trajectoryId: "traj-2", source: "screen", summary: "editor open" },
        { kind: "action", ts: 2, trajectoryId: "traj-2", tool: "kbd", summary: "type:hello" },
        { kind: "action", ts: 3, trajectoryId: "traj-2", tool: "mouse", summary: "click:save" },
      ],
    };
    const sequence = movementSequenceFromReplay(manifest);
    expect(sequence.tokens).toEqual(["kbd:type:hello", "mouse:click:save"]);
  });

  it("drops empty sequences when assembling a dataset", () => {
    const dataset = buildMovementDataset([
      { id: "a", tokens: [] },
      { id: "b", tokens: [movementTokenFromAction({ tool: "mouse", summary: "click" })] },
    ]);
    expect(dataset.sequences.map((sequence) => sequence.id)).toEqual(["b"]);
  });
});

describe("evaluateNextTokenAccuracy", () => {
  const backend = new MarkovMovementBackend();

  it("scores perfect recall on the training distribution", async () => {
    const dataset = buildMovementDataset([saveWorkflow(0), saveWorkflow(1)]);
    const artifact = await backend.train(dataset);
    const result = evaluateNextTokenAccuracy(backend, artifact, dataset.sequences);
    expect(result.predictions).toBe(8); // 4 transitions x 2 sequences
    expect(result.accuracy).toBe(1);
    expect(result.meanConfidence).toBeGreaterThan(0);
  });

  it("measures generalization to a held-out related sequence", async () => {
    const artifact = await backend.train(buildMovementDataset([saveWorkflow(0), saveWorkflow(1)]));
    // Held-out sequence with a novel first movement but a shared save tail.
    const heldOut: MovementSequence = {
      id: "held-out",
      tokens: [
        "focus:window.terminal",
        "click:menu.file",
        "click:menu.file.save",
        "type:dialog.filename",
        "click:button.confirm",
      ],
    };
    const result = evaluateNextTokenAccuracy(backend, artifact, [heldOut]);
    // The shared 4-token tail transfers; only the novel first transition can miss.
    expect(result.accuracy).toBeGreaterThanOrEqual(0.75);
    expect(result.backoffRate).toBeGreaterThan(0);
  });
});
