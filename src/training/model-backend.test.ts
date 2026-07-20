import { describe, expect, it } from "vitest";
import {
  NgramMovementBackend,
  type MovementDataset,
  type MovementSequence,
  evaluateReplayFidelity,
  movementDatasetFromReplays,
  movementTokenOf,
} from "./model-backend.js";
import {
  DEFAULT_SYNTHETIC_INTENTS,
  generateHeldOutMovements,
  generateSyntheticMovementDataset,
} from "./synthetic-movements.js";

const trainingSet: MovementDataset = {
  sequences: [
    {
      id: "t1",
      events: [
        { kind: "observation", source: "browser:deploy", summary: "deploy page" },
        { kind: "action", tool: "browser", summary: "click deploy" },
        { kind: "observation", source: "terminal:prompt", summary: "shell ready" },
        { kind: "action", tool: "terminal", summary: "run tests" },
      ],
    },
    {
      id: "t2",
      events: [
        { kind: "observation", source: "browser:deploy", summary: "deploy page" },
        { kind: "action", tool: "browser", summary: "click deploy" },
      ],
    },
  ],
};

describe("NgramMovementBackend", () => {
  it("learns and repeats recorded movements (in-distribution replay)", async () => {
    const policy = await new NgramMovementBackend().train(trainingSet, { order: 2 });
    const report = evaluateReplayFidelity(policy, trainingSet.sequences);
    // Every recorded action should be reproduced exactly.
    expect(report.actionSteps).toBe(3);
    expect(report.accuracy).toBe(1);
    expect(report.toolAccuracy).toBe(1);
  });

  it("predicts the recorded action for a known observation", async () => {
    const policy = await new NgramMovementBackend().train(trainingSet, { order: 2 });
    const prediction = policy.predict([{ kind: "observation", source: "browser:deploy", summary: "deploy page" }]);
    expect(prediction).toMatchObject({ tool: "browser", summary: "click deploy" });
    expect(prediction?.confidence).toBe(1);
  });

  it("generalizes to a novel history via back-off", async () => {
    const policy = await new NgramMovementBackend().train(trainingSet, { order: 2 });
    // A context never seen at full order (preceded by an unfamiliar action),
    // but ending in the familiar "browser:deploy" observation.
    const prediction = policy.predict([
      { kind: "action", tool: "window", summary: "dismiss dialog" },
      { kind: "observation", source: "browser:deploy", summary: "deploy page" },
    ]);
    expect(prediction?.tool).toBe("browser");
    expect(prediction?.summary).toBe("click deploy");
    // Backed off below the full order because the 2-token context was unseen.
    expect(prediction?.matchedOrder).toBeLessThan(2);
  });

  it("falls back to the global prior when no context matches", async () => {
    const policy = await new NgramMovementBackend().train(trainingSet, { order: 2 });
    const prediction = policy.predict([{ kind: "observation", source: "totally:unknown" }]);
    // browser action occurs twice, terminal once -> global prior is browser.
    expect(prediction?.tool).toBe("browser");
    expect(prediction?.matchedOrder).toBe(0);
  });

  it("returns undefined when no action was ever observed", async () => {
    const policy = await new NgramMovementBackend().train({
      sequences: [{ id: "obs-only", events: [{ kind: "observation", source: "x" }] }],
    });
    expect(policy.predict([{ kind: "observation", source: "x" }])).toBeUndefined();
  });

  it("is deterministic and round-trips through serialize/restore", async () => {
    const backend = new NgramMovementBackend();
    const policy = await backend.train(trainingSet, { order: 2 });
    const serialized = policy.serialize();
    const restored = NgramMovementBackend.restore(serialized);

    expect(restored.stats).toEqual(policy.stats);
    const history: MovementSequence["events"] = [
      { kind: "observation", source: "browser:deploy", summary: "deploy page" },
    ];
    expect(restored.predict(history)).toEqual(policy.predict(history));
    // Re-serializing yields identical bytes (determinism).
    expect(restored.serialize()).toEqual(serialized);
  });

  it("reports meaningful stats", async () => {
    const policy = await new NgramMovementBackend().train(trainingSet, { order: 2 });
    expect(policy.stats).toMatchObject({
      backendId: "ngram-markov",
      order: 2,
      sequenceCount: 2,
      actionCount: 3,
      actionVocabulary: 2,
    });
    expect(policy.stats.contextCount).toBeGreaterThan(0);
  });
});

describe("synthetic movement generalization", () => {
  it("generalizes from synthetic training data to held-out related trajectories", async () => {
    const train = generateSyntheticMovementDataset({ seed: 7, sequenceCount: 40, noise: 0.3 });
    const heldOut = generateHeldOutMovements({ seed: 7, sequenceCount: 20, noise: 0.3 });

    const policy = await new NgramMovementBackend().train(train, { order: 2 });
    const report = evaluateReplayFidelity(policy, heldOut.sequences);

    expect(report.actionSteps).toBeGreaterThan(0);
    // The intent grammar is deterministic given the observation, so a policy
    // that learned it should nail nearly every held-out action.
    expect(report.accuracy).toBeGreaterThan(0.95);
  });

  it("covers every default intent tool in generated data", () => {
    const dataset = generateSyntheticMovementDataset({ seed: 3, sequenceCount: 60 });
    const tools = new Set(
      dataset.sequences.flatMap((s) => s.events.filter((e) => e.kind === "action").map((e) => movementTokenOf(e))),
    );
    for (const intent of DEFAULT_SYNTHETIC_INTENTS) {
      expect(tools.has(`act:${intent.tool}`)).toBe(true);
    }
  });

  it("produces byte-identical datasets for the same seed", () => {
    const a = generateSyntheticMovementDataset({ seed: 99, sequenceCount: 10, noise: 0.5 });
    const b = generateSyntheticMovementDataset({ seed: 99, sequenceCount: 10, noise: 0.5 });
    expect(a).toEqual(b);
  });
});

describe("movementDatasetFromReplays", () => {
  it("normalizes replay timeline events into movement sequences", () => {
    const dataset = movementDatasetFromReplays([
      {
        sessionId: "sess-1",
        events: [
          { kind: "transcript", ts: 1, messageId: "m1", role: "user", content: "deploy it" },
          { kind: "observation", ts: 2, trajectoryId: "tr1", source: "browser", summary: "deploy page" },
          { kind: "action", ts: 3, trajectoryId: "tr1", tool: "browser", summary: "click deploy" },
        ],
      },
    ]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]!.events).toEqual([
      { kind: "message", role: "user" },
      { kind: "observation", source: "browser", summary: "deploy page" },
      { kind: "action", tool: "browser", summary: "click deploy" },
    ]);
  });
});
