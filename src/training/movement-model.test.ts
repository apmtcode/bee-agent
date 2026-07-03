import { describe, expect, it } from "vitest";
import type { ReplayTimelineEvent } from "../capture/replay.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import {
  MarkovMovementBackend,
  MOVEMENT_END_TOKEN,
  buildMovementDatasetFromReplays,
  buildMovementDatasetFromTrajectories,
  createMovementModelBackend,
  evaluateNextTokenAccuracy,
  listMovementModelBackends,
  loadMovementModel,
  registerMovementModelBackend,
  tokenizeReplayEvent,
  type MovementModelBackend,
  type MovementTrainingDataset,
} from "./movement-model.js";

const dataset: MovementTrainingDataset = {
  version: 1,
  sequences: [
    { id: "a", tokens: ["obs:screen", "act:mouse.move", "act:mouse.click", "act:key.type"] },
    { id: "b", tokens: ["obs:screen", "act:mouse.move", "act:mouse.click", "act:key.type"] },
    { id: "c", tokens: ["obs:screen", "act:mouse.move", "act:mouse.click", "act:key.enter"] },
  ],
};

describe("MarkovMovementBackend", () => {
  it("trains and deterministically predicts the majority next movement", async () => {
    const model = await new MarkovMovementBackend().train(dataset, { order: 2 });

    // After "mouse.move mouse.click" the majority follow-up is key.type (2 vs 1).
    const prediction = model.predictNext(["act:mouse.move", "act:mouse.click"]);
    expect(prediction?.token).toBe("act:key.type");
    expect(prediction?.orderUsed).toBe(2);
    expect(prediction?.probability).toBeGreaterThan(0.5);
  });

  it("is reproducible across runs", async () => {
    const backend = new MarkovMovementBackend();
    const first = (await backend.train(dataset)).toJSON();
    const second = (await backend.train(dataset)).toJSON();
    expect(first).toEqual(second);
  });

  it("generates a full recorded movement sequence from a seed and halts at the end token", async () => {
    const model = await new MarkovMovementBackend().train(dataset, { order: 3 });
    const rollout = model.generate({ seed: ["obs:screen"], maxTokens: 16 });
    expect(rollout).toEqual(["obs:screen", "act:mouse.move", "act:mouse.click", "act:key.type"]);
    expect(rollout).not.toContain(MOVEMENT_END_TOKEN);
  });

  it("generalizes to a novel context via back-off instead of returning nothing", async () => {
    const model = await new MarkovMovementBackend().train(dataset, { order: 2 });
    // This exact 2-token context was never observed, but the suffix "mouse.click" was.
    const prediction = model.predictNext(["act:never.seen", "act:mouse.click"]);
    expect(prediction).toBeDefined();
    expect(prediction?.token).toBe("act:key.type");
    expect(prediction?.orderUsed).toBeLessThan(2);
  });
});

describe("model persistence", () => {
  it("round-trips through serialization for inference without retraining", async () => {
    const model = await new MarkovMovementBackend().train(dataset, { order: 2 });
    const restored = loadMovementModel(model.toJSON());
    expect(restored.order).toBe(model.order);
    expect(restored.predictNext(["obs:screen"])?.token).toBe(model.predictNext(["obs:screen"])?.token);
    expect(restored.generate({ seed: ["obs:screen"] })).toEqual(model.generate({ seed: ["obs:screen"] }));
  });
});

describe("dataset builders", () => {
  it("tokenizes replay timeline events", () => {
    const events: ReplayTimelineEvent[] = [
      { kind: "observation", ts: 1, trajectoryId: "t1", source: "screen", summary: "" },
      { kind: "action", ts: 2, trajectoryId: "t1", tool: "mouse.click", summary: "" },
      { kind: "transcript", ts: 3, messageId: "m1", role: "assistant", content: "" },
    ];
    expect(events.map(tokenizeReplayEvent)).toEqual(["obs:screen", "act:mouse.click", "msg:assistant"]);

    const built = buildMovementDatasetFromReplays([{ trajectoryIds: ["t1"], events }]);
    expect(built.sequences[0]).toEqual({ id: "t1", tokens: ["obs:screen", "act:mouse.click", "msg:assistant"] });
  });

  it("builds a time-ordered dataset from trajectory spans", () => {
    const span = buildTrajectorySpan({
      id: "traj-1",
      sessionId: "sess-1",
      observations: [{ kind: "observation", source: "screen", summary: "", ts: 30 }],
      actions: [
        { kind: "action", tool: "mouse.click", summary: "", ts: 20 },
        { kind: "action", tool: "mouse.move", summary: "", ts: 10 },
      ],
    });
    const built = buildMovementDatasetFromTrajectories([span]);
    // Sorted by ts: move(10), click(20), screen(30).
    expect(built.sequences[0]?.tokens).toEqual(["act:mouse.move", "act:mouse.click", "obs:screen"]);
  });
});

describe("evaluateNextTokenAccuracy", () => {
  it("scores held-out related sequences above chance and is perfect on memorized ones", async () => {
    const model = await new MarkovMovementBackend().train(dataset, { order: 2 });

    const memorized = evaluateNextTokenAccuracy(model, [dataset.sequences[0]!]);
    expect(memorized.accuracy).toBe(1);

    const heldOut = evaluateNextTokenAccuracy(model, [
      { id: "held", tokens: ["obs:screen", "act:mouse.move", "act:mouse.click", "act:key.type"] },
    ]);
    expect(heldOut.total).toBeGreaterThan(0);
    expect(heldOut.accuracy).toBeGreaterThan(0.5);
  });

  it("returns zero accuracy for an empty eval set", async () => {
    const model = await new MarkovMovementBackend().train(dataset);
    expect(evaluateNextTokenAccuracy(model, [])).toEqual({ total: 0, correct: 0, accuracy: 0 });
  });
});

describe("backend registry", () => {
  it("creates the default backend and lists registered backends", () => {
    expect(createMovementModelBackend().id).toBe("markov-backoff");
    expect(listMovementModelBackends()).toContain("markov-backoff");
  });

  it("throws for an unknown backend id", () => {
    expect(() => createMovementModelBackend("does-not-exist")).toThrow(/Unknown movement model backend/);
  });

  it("supports plugging in a custom backend behind the seam", async () => {
    const custom: MovementModelBackend = {
      id: "custom-test",
      async train(data) {
        return new MarkovMovementBackend().train(data);
      },
    };
    registerMovementModelBackend("custom-test", () => custom);
    expect(listMovementModelBackends()).toContain("custom-test");
    expect(createMovementModelBackend("custom-test").id).toBe("custom-test");
  });
});
