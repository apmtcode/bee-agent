import { describe, expect, it } from "vitest";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import type { TrajectoryAction } from "../capture/trajectory.js";
import {
  MOVEMENT_END_TOKEN,
  NgramMovementBackend,
  evaluateMovementFidelity,
  generateSyntheticMovementSequences,
  parseToken,
  sequenceFromSpan,
  stepFromAction,
  tokenizeStep,
  trainMovementModel,
  type MovementStep,
} from "./movement-model.js";

function action(tool: string, ts: number, metadata?: Record<string, unknown>): TrajectoryAction {
  return { kind: "action", tool, summary: `${tool}@${ts}`, ts, ...(metadata ? { metadata } : {}) };
}

describe("tokenizeStep / parseToken", () => {
  it("round-trips a structured step through its token", () => {
    const step: MovementStep = { tool: "device", gesture: "tap", target: "ComposeButton" };
    const token = tokenizeStep(step);
    expect(parseToken(token)).toEqual({ tool: "device", gesture: "tap", target: "composebutton" });
  });

  it("produces stable tokens for the same logical step", () => {
    expect(tokenizeStep({ tool: "device", gesture: "swipe", direction: "up" })).toBe(
      tokenizeStep({ tool: "device", gesture: "swipe", direction: "up" }),
    );
  });

  it("keeps tokens separator-safe", () => {
    const token = tokenizeStep({ tool: "a|b", gesture: "x" });
    expect(token.split("|")).toHaveLength(4);
    expect(parseToken(token).tool).toBe("a/b");
  });
});

describe("sequenceFromSpan", () => {
  it("extracts movement tokens ordered by timestamp", () => {
    const span = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [
        action("device", 30, { gesture: "type", target: "field" }),
        action("device", 10, { gesture: "tap", target: "button" }),
      ],
    });
    const sequence = sequenceFromSpan(span);
    expect(sequence).toEqual([
      tokenizeStep({ tool: "device", gesture: "tap", target: "button" }),
      tokenizeStep({ tool: "device", gesture: "type", target: "field" }),
    ]);
  });

  it("derives target from valueSummary when target is absent", () => {
    const step = stepFromAction(action("device", 1, { gesture: "type", valueSummary: "hello" }));
    expect(step).toEqual({ tool: "device", gesture: "type", target: "hello" });
  });
});

describe("NgramMovementBackend training + replay", () => {
  const a = tokenizeStep({ tool: "device", gesture: "tap", target: "menu" });
  const b = tokenizeStep({ tool: "device", gesture: "tap", target: "settings" });
  const c = tokenizeStep({ tool: "device", gesture: "tap", target: "save" });

  it("predicts the learned next movement", () => {
    const model = trainMovementModel([[a, b, c]], { order: 2 });
    const prediction = model.predict([a, b]);
    expect(prediction.token).toBe(c);
    expect(prediction.probability).toBeGreaterThan(0);
  });

  it("reproduces a recorded sequence via greedy rollout (objective 2d)", () => {
    const model = trainMovementModel([[a, b, c]], { order: 2 });
    const replayed = model.generate([a], 10);
    expect([a, ...replayed]).toEqual([a, b, c]);
  });

  it("terminates at the end sentinel rather than looping forever", () => {
    const model = trainMovementModel([[a, b]], { order: 2 });
    const replayed = model.generate([a], 100);
    expect(replayed).toEqual([b]);
    expect(replayed).not.toContain(MOVEMENT_END_TOKEN);
  });

  it("generalizes to a novel-but-related context via backoff (objective 2e)", () => {
    // "b" is always followed by "c"; "a" appears before "b" but the order-2
    // context [z, b] was never seen. Backoff to context [b] should still predict c.
    const z = tokenizeStep({ tool: "device", gesture: "shortcut", target: "spotlight" });
    const model = trainMovementModel([[a, b, c]], { order: 2 });
    const prediction = model.predict([z, b]);
    expect(prediction.token).toBe(c);
    expect(prediction.contextOrderUsed).toBe(1);
  });

  it("falls back to the unigram prior for a fully unseen context", () => {
    const model = trainMovementModel([[a, b, c]], { order: 2 });
    const unseen = tokenizeStep({ tool: "browser", gesture: "click" });
    const prediction = model.predict([unseen]);
    expect(prediction.contextOrderUsed).toBe(0);
    expect(prediction.distribution.length).toBeGreaterThan(0);
  });

  it("excludes the end sentinel from the reported vocabulary size", () => {
    const model = trainMovementModel([[a, b, c]], { order: 2 });
    expect(model.vocabularySize).toBe(3);
  });
});

describe("serialize / restore", () => {
  it("restores an identical model from a snapshot", () => {
    const a = tokenizeStep({ tool: "device", gesture: "tap", target: "x" });
    const b = tokenizeStep({ tool: "device", gesture: "tap", target: "y" });
    const backend = new NgramMovementBackend();
    const model = backend.train([[a, b]], { order: 2 });
    const restored = backend.restore(model.serialize());
    expect(restored.predict([a]).token).toBe(model.predict([a]).token);
    expect(restored.serialize()).toEqual(model.serialize());
  });
});

describe("synthetic generator + fidelity eval", () => {
  const login: MovementStep[] = [
    { tool: "device", gesture: "tap", target: "username" },
    { tool: "device", gesture: "type", target: "user" },
    { tool: "device", gesture: "tap", target: "password" },
    { tool: "device", gesture: "type", target: "secret" },
    { tool: "device", gesture: "tap", target: "submit" },
  ];

  it("is deterministic", () => {
    const spec = { workflows: [{ name: "login", steps: login }], count: 3 };
    expect(generateSyntheticMovementSequences(spec)).toEqual(
      generateSyntheticMovementSequences(spec),
    );
  });

  it("injects related variants on the configured cadence", () => {
    const sequences = generateSyntheticMovementSequences({
      workflows: [{ name: "login", steps: login }],
      count: 2,
      variantEvery: 2,
    });
    expect(sequences[0]).not.toEqual(sequences[1]);
    // Only the final step differs between the base and the variant.
    expect(sequences[1]!.slice(0, -1)).toEqual(sequences[0]!.slice(0, -1));
  });

  it("scores perfect fidelity when replaying its own training data", () => {
    const dataset = generateSyntheticMovementSequences({
      workflows: [{ name: "login", steps: login }],
      count: 4,
    });
    const model = trainMovementModel(dataset, { order: 2 });
    const report = evaluateMovementFidelity(model, dataset);
    expect(report.fullSequenceAccuracy).toBe(1);
    expect(report.nextStepAccuracy).toBe(1);
    expect(report.sequences).toBe(4);
  });

  it("generalizes to held-out related variants above chance", () => {
    const train = generateSyntheticMovementSequences({
      workflows: [{ name: "login", steps: login }],
      count: 4,
    });
    const heldOut = generateSyntheticMovementSequences({
      workflows: [{ name: "login", steps: login }],
      count: 1,
      variantEvery: 1, // every sequence is a variant -> novel final target
    });
    const model = trainMovementModel(train, { order: 2 });
    const report = evaluateMovementFidelity(model, heldOut);
    // The variant only diverges on its last step, so most next-step
    // predictions still land via backoff to learned context.
    expect(report.nextStepAccuracy).toBeGreaterThan(0.5);
  });
});
