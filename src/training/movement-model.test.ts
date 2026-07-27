import { describe, expect, it } from "vitest";
import {
  MarkovMovementBackend,
  buildMovementDatasetFromReplay,
  buildMovementDatasetFromTrajectories,
  movementToken,
  type MovementDataset,
} from "./movement-model.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import { buildReplayManifest } from "../capture/replay.js";
import {
  DEFAULT_SYNTHETIC_WORKFLOWS,
  generateSyntheticMovementDataset,
} from "./synthetic-movements.js";

function mailDataset(): MovementDataset {
  return generateSyntheticMovementDataset({
    workflows: [DEFAULT_SYNTHETIC_WORKFLOWS[0]],
    repeats: 4,
  });
}

describe("MarkovMovementBackend", () => {
  it("trains a model and reports stats over the dataset", async () => {
    const backend = new MarkovMovementBackend();
    const model = await backend.train(mailDataset(), { order: 2 });

    expect(model.stats.backend).toBe("markov");
    expect(model.stats.sequenceCount).toBe(4);
    expect(model.stats.stepCount).toBe(4 * 5);
    expect(model.stats.distinctTokens).toBeGreaterThan(0);
    expect(model.stats.maxOrder).toBe(2);
  });

  it("reproduces a recorded movement from its seed (objective 2c)", async () => {
    const dataset = mailDataset();
    const backend = new MarkovMovementBackend();
    const model = await backend.train(dataset, { order: 2 });

    const sequence = dataset.sequences[0];
    const generated = model.generate(
      { context: sequence.context, history: sequence.steps.slice(0, 1) },
      sequence.steps.length - 1,
    );
    const generatedTokens = generated.map(movementToken);
    const expectedTokens = sequence.steps.slice(1).map(movementToken);
    expect(generatedTokens).toEqual(expectedTokens);
  });

  it("predicts the next movement and terminates deterministically", async () => {
    const dataset = mailDataset();
    const backend = new MarkovMovementBackend();
    const model = await backend.train(dataset, { order: 2 });
    const sequence = dataset.sequences[0];

    const prediction = model.predictNext({
      context: sequence.context,
      history: sequence.steps.slice(0, 1),
    });
    expect(prediction).toBeDefined();
    expect(movementToken(prediction!.step)).toBe(movementToken(sequence.steps[1]));
    expect(prediction!.confidence).toBeGreaterThan(0);

    // Two identical trainings must produce identical predictions (no RNG).
    const modelB = await backend.train(dataset, { order: 2 });
    const predictionB = modelB.predictNext({
      context: sequence.context,
      history: sequence.steps.slice(0, 1),
    });
    expect(movementToken(predictionB!.step)).toBe(movementToken(prediction!.step));
  });

  it("generalizes to a related but unseen prefix via back-off (objective 2d)", async () => {
    // Train on a workflow, then query with a novel history it never saw exactly
    // but that shares the app context — the model should still back off and
    // predict a plausible next movement rather than returning nothing.
    const dataset = mailDataset();
    const backend = new MarkovMovementBackend();
    const model = await backend.train(dataset, { order: 3 });

    const novelHistory = [
      { ts: 0, gesture: "scroll", direction: "up", summary: "unseen movement" },
    ];
    const prediction = model.predictNext({ context: "mail", history: novelHistory });
    expect(prediction).toBeDefined();
    expect(prediction!.generalized).toBe(true);
    expect(prediction!.order).toBeLessThan(1);
  });

  it("returns undefined when there is no learned context at all", async () => {
    const backend = new MarkovMovementBackend();
    const model = await backend.train({ version: 1, sequences: [] });
    expect(model.predictNext({ context: "nothing", history: [] })).toBeUndefined();
    expect(model.generate({ context: "nothing", history: [] }, 5)).toEqual([]);
  });
});

describe("dataset builders", () => {
  it("builds a movement dataset from trajectory spans with gesture metadata", () => {
    const trajectory = buildTrajectorySpan({
      id: "traj-1",
      sessionId: "session-1",
      observations: [
        { kind: "observation", source: "device", summary: "Mail active", ts: 1, metadata: { appName: "Mail" } },
      ],
      actions: [
        { kind: "action", tool: "device", summary: "tapped compose", ts: 3, metadata: { gesture: "tap", target: "compose" } },
        { kind: "action", tool: "device", summary: "typed body", ts: 2, metadata: { gesture: "type", target: "body" } },
      ],
    });

    const dataset = buildMovementDatasetFromTrajectories([trajectory]);
    expect(dataset.sequences).toHaveLength(1);
    const sequence = dataset.sequences[0];
    expect(sequence.context).toBe("Mail");
    // Steps are sorted by ts, so "type body" (ts 2) comes before "tap compose" (ts 3).
    expect(sequence.steps.map((step) => step.gesture)).toEqual(["type", "tap"]);
    expect(sequence.steps[0].target).toBe("body");
  });

  it("skips trajectories with no actions", () => {
    const trajectory = buildTrajectorySpan({ id: "empty", sessionId: "s" });
    expect(buildMovementDatasetFromTrajectories([trajectory]).sequences).toHaveLength(0);
  });

  it("builds a movement dataset from a replay manifest action timeline", () => {
    const trajectory = buildTrajectorySpan({
      id: "traj-2",
      sessionId: "session-2",
      actions: [
        { kind: "action", tool: "tap", summary: "tapped", ts: 5 },
        { kind: "action", tool: "swipe", summary: "swiped", ts: 6 },
      ],
    });
    const manifest = buildReplayManifest({ sessionId: "session-2", transcript: [], trajectories: [trajectory] });
    const dataset = buildMovementDatasetFromReplay(manifest);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0].steps.map((step) => step.gesture)).toEqual(["tap", "swipe"]);
  });
});
