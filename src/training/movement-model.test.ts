import { describe, expect, it } from "vitest";
import { buildTrajectorySpan, type TrajectorySpan } from "../capture/trajectory.js";
import {
  MarkovMovementBackend,
  deriveMovementDatasetFromTrajectories,
  evaluateReplayFidelity,
  restoreMovementModel,
  tokenKey,
  type MovementDataset,
  type MovementSequence,
} from "./movement-model.js";

function gestureTrajectory(params: {
  id: string;
  goal: string;
  reward?: number;
  steps: Array<{ gesture: string; target?: string; direction?: string }>;
}): TrajectorySpan {
  return buildTrajectorySpan({
    id: params.id,
    sessionId: `session-${params.id}`,
    captureTier: "app",
    observations: [{ kind: "observation", source: "device", summary: params.goal, ts: 0 }],
    actions: params.steps.map((step, index) => ({
      kind: "action",
      tool: "device",
      summary: `${step.gesture} ${step.target ?? step.direction ?? ""}`.trim(),
      ts: index + 1,
      metadata: {
        gesture: step.gesture,
        ...(step.target ? { target: step.target } : {}),
        ...(step.direction ? { direction: step.direction } : {}),
      },
    })),
    outcome: { status: "success", summary: params.goal, ...(params.reward !== undefined ? { reward: params.reward } : {}) },
  });
}

const trainingTrajectories: TrajectorySpan[] = [
  gestureTrajectory({
    id: "compose-email",
    goal: "compose email",
    reward: 1,
    steps: [{ gesture: "tap", target: "body" }, { gesture: "type", target: "body" }, { gesture: "tap", target: "send" }],
  }),
  gestureTrajectory({
    id: "reply-email",
    goal: "reply email",
    reward: 1,
    steps: [{ gesture: "tap", target: "thread" }, { gesture: "type", target: "body" }, { gesture: "tap", target: "send" }],
  }),
  gestureTrajectory({
    id: "compose-message",
    goal: "compose message",
    reward: 1,
    steps: [{ gesture: "tap", target: "body" }, { gesture: "type", target: "body" }, { gesture: "tap", target: "send" }],
  }),
];

async function trainOn(dataset: MovementDataset) {
  return await new MarkovMovementBackend().train(dataset);
}

describe("deriveMovementDatasetFromTrajectories", () => {
  it("maps trajectory actions to movement tokens with gesture metadata", () => {
    const dataset = deriveMovementDatasetFromTrajectories(trainingTrajectories);
    expect(dataset.sequences).toHaveLength(3);
    const compose = dataset.sequences.find((sequence) => sequence.id === "compose-email");
    expect(compose?.goal).toBe("compose email");
    expect(compose?.tokens.map((token) => token.gesture)).toEqual(["tap", "type", "tap"]);
    expect(compose?.tokens.at(-1)?.target).toBe("send");
    expect(compose?.reward).toBe(1);
  });

  it("skips trajectories with no actions", () => {
    const empty = buildTrajectorySpan({ id: "idle", sessionId: "s", actions: [] });
    const dataset = deriveMovementDatasetFromTrajectories([empty]);
    expect(dataset.sequences).toHaveLength(0);
  });
});

describe("MarkovMovementBackend training + replay (objective 2c)", () => {
  it("reports learning statistics", async () => {
    const model = await trainOn(deriveMovementDatasetFromTrajectories(trainingTrajectories));
    expect(model.backendId).toBe("markov-mock");
    expect(model.stats.sequenceCount).toBe(3);
    expect(model.stats.goalCount).toBe(3);
    expect(model.stats.vocabularySize).toBeGreaterThan(0);
  });

  it("reproduces a recorded movement exactly for a known goal", async () => {
    const model = await trainOn(deriveMovementDatasetFromTrajectories(trainingTrajectories));
    const result = model.generate({ goal: "reply email" });
    expect(result.mode).toBe("replay");
    expect(result.matchScore).toBe(1);
    expect(result.tokens.map((token) => `${token.gesture}:${token.target}`)).toEqual([
      "tap:thread",
      "type:body",
      "tap:send",
    ]);
  });

  it("prefers the higher-reward recording when a goal has duplicates", async () => {
    const dataset = deriveMovementDatasetFromTrajectories([
      gestureTrajectory({ id: "lo", goal: "open app", reward: 0, steps: [{ gesture: "tap", target: "wrong" }] }),
      gestureTrajectory({ id: "hi", goal: "open app", reward: 5, steps: [{ gesture: "tap", target: "icon" }] }),
    ]);
    const model = await trainOn(dataset);
    const result = model.generate({ goal: "open app", mode: "replay" });
    expect(result.tokens[0]?.target).toBe("icon");
  });
});

describe("MarkovMovementBackend generalization (objective 2d)", () => {
  it("composes a novel-but-related movement ending in the learned terminal action", async () => {
    const model = await trainOn(deriveMovementDatasetFromTrajectories(trainingTrajectories));
    // "reply message" was never recorded, but shares words with reply/compose + email/message.
    const result = model.generate({ goal: "reply message", mode: "generalize" });
    expect(result.mode).toBe("generalize");
    expect(result.tokens.length).toBeGreaterThan(0);
    // Every produced token comes from the learned vocabulary.
    const vocab = new Set(
      deriveMovementDatasetFromTrajectories(trainingTrajectories).sequences.flatMap((sequence) =>
        sequence.tokens.map((token) => tokenKey(token)),
      ),
    );
    for (const token of result.tokens) {
      expect(vocab.has(tokenKey(token))).toBe(true);
    }
    // Generalized movements still converge on the terminal "tap send" the model always learned.
    expect(result.tokens.at(-1)).toMatchObject({ gesture: "tap", target: "send" });
  });

  it("cold-starts from the global prior for an unrelated goal", async () => {
    const model = await trainOn(deriveMovementDatasetFromTrajectories(trainingTrajectories));
    const result = model.generate({ goal: "xyzzy plugh", mode: "generalize" });
    expect(result.matchScore).toBe(0);
    expect(result.tokens.length).toBeGreaterThan(0);
  });
});

describe("predictNext", () => {
  it("predicts the most likely next token given a prefix", async () => {
    const model = await trainOn(deriveMovementDatasetFromTrajectories(trainingTrajectories));
    const prediction = model.predictNext([
      { tool: "device", gesture: "tap", target: "body", summary: "tap body" },
      { tool: "device", gesture: "type", target: "body", summary: "type body" },
    ]);
    expect(prediction?.token).toMatchObject({ gesture: "tap", target: "send" });
    expect(prediction?.probability).toBeGreaterThan(0);
    expect(prediction?.probability).toBeLessThanOrEqual(1);
  });
});

describe("determinism + persistence", () => {
  it("produces identical output across repeated training runs", async () => {
    const dataset = deriveMovementDatasetFromTrajectories(trainingTrajectories);
    const a = await trainOn(dataset);
    const b = await trainOn(dataset);
    const request = { goal: "reply message", mode: "generalize" as const };
    expect(a.generate(request).tokens).toEqual(b.generate(request).tokens);
  });

  it("round-trips through a snapshot without changing behavior", async () => {
    const model = await trainOn(deriveMovementDatasetFromTrajectories(trainingTrajectories));
    const restored = restoreMovementModel(model.snapshot());
    expect(restored.stats).toEqual(model.stats);
    const request = { goal: "reply email" };
    expect(restored.generate(request).tokens).toEqual(model.generate(request).tokens);
  });
});

describe("evaluateReplayFidelity harness", () => {
  it("scores perfect replay fidelity on the training goals", async () => {
    const dataset = deriveMovementDatasetFromTrajectories(trainingTrajectories);
    const model = await trainOn(dataset);
    const report = evaluateReplayFidelity(model, dataset.sequences);
    expect(report.evaluated).toBe(3);
    expect(report.exactMatchRate).toBe(1);
    expect(report.meanTokenAccuracy).toBe(1);
  });

  it("reports partial token accuracy on held-out related goals", async () => {
    const model = await trainOn(deriveMovementDatasetFromTrajectories(trainingTrajectories));
    const heldOut: MovementSequence[] = [
      {
        id: "reply-message",
        goal: "reply message",
        tokens: [
          { tool: "device", gesture: "tap", target: "thread", summary: "tap thread" },
          { tool: "device", gesture: "type", target: "body", summary: "type body" },
          { tool: "device", gesture: "tap", target: "send", summary: "tap send" },
        ],
      },
    ];
    const report = evaluateReplayFidelity(model, heldOut, { mode: "generalize" });
    expect(report.evaluated).toBe(1);
    expect(report.meanTokenAccuracy).toBeGreaterThan(0);
    expect(report.meanTokenAccuracy).toBeLessThanOrEqual(1);
  });
});
