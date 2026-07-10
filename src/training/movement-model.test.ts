import { describe, expect, it } from "vitest";
import type { ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  MOVEMENT_END_TOKEN,
  MovementModelRegistry,
  NgramMovementModelBackend,
  buildMovementDataset,
  defaultMovementModelRegistry,
  evaluateMovementPolicy,
  extractMovementSequence,
  movementSequenceFromReplayEvents,
  movementToken,
  movementTokenTool,
  trainMovementModel,
  type MovementDataset,
  type MovementSequence,
} from "./movement-model.js";

/** Synthetic trajectory factory — stands in for real recorded OS movement. */
function syntheticTrajectory(id: string, actions: Array<{ tool: string; summary: string }>): TrajectorySpan {
  return {
    id,
    sessionId: `session-${id}`,
    createdAt: "2026-07-10T00:00:00.000Z",
    captureTier: "app",
    observations: [],
    actions: actions.map((action, index) => ({
      kind: "action",
      tool: action.tool,
      summary: action.summary,
      ts: (index + 1) * 10,
    })),
  };
}

function tokensOf(actions: Array<{ tool: string; summary: string }>): string[] {
  return actions.map((action) => movementToken(action.tool, action.summary));
}

describe("movement token normalization", () => {
  it("is stable across whitespace, case, and trailing punctuation", () => {
    expect(movementToken("Device", "Tapped  Submit.")).toBe(movementToken("device", "tapped submit"));
    expect(movementTokenTool(movementToken("device", "tapped submit"))).toBe("device");
  });
});

describe("sequence extraction", () => {
  it("orders trajectory actions chronologically by ts", () => {
    // Record actions with ts out of array order; extraction must sort by ts.
    const trajectory: TrajectorySpan = {
      id: "t1",
      sessionId: "s1",
      createdAt: "2026-07-10T00:00:00.000Z",
      captureTier: "app",
      observations: [],
      actions: [
        { kind: "action", tool: "device", summary: "tap submit", ts: 200 },
        { kind: "action", tool: "device", summary: "open app", ts: 100 },
      ],
    };
    const sequence = extractMovementSequence(trajectory);
    expect(sequence.tokens).toEqual([movementToken("device", "open app"), movementToken("device", "tap submit")]);
    expect(sequence.trajectoryId).toBe("t1");
  });

  it("extracts only action events from a replay timeline", () => {
    const events: ReplayTimelineEvent[] = [
      { kind: "transcript", ts: 1, messageId: "m1", role: "user", content: "hi" },
      { kind: "observation", ts: 2, trajectoryId: "t1", source: "device", summary: "app active" },
      { kind: "action", ts: 3, trajectoryId: "t1", tool: "device", summary: "tap submit" },
    ];
    const sequence = movementSequenceFromReplayEvents(events);
    expect(sequence.tokens).toEqual([movementToken("device", "tap submit")]);
    expect(sequence.trajectoryId).toBe("t1");
  });

  it("drops empty sequences when building a dataset", () => {
    const dataset = buildMovementDataset([
      syntheticTrajectory("empty", []),
      syntheticTrajectory("t1", [{ tool: "device", summary: "tap" }]),
    ]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]?.trajectoryId).toBe("t1");
  });
});

describe("ngram movement policy — learning", () => {
  const workflow = [
    { tool: "device", summary: "open mail" },
    { tool: "device", summary: "compose message" },
    { tool: "device", summary: "type body" },
    { tool: "device", summary: "tap send" },
  ];

  function trainedOnWorkflow() {
    const dataset = buildMovementDataset([
      syntheticTrajectory("a", workflow),
      syntheticTrajectory("b", workflow),
    ]);
    return trainMovementModel(dataset);
  }

  it("predicts the recorded next movement with full confidence", () => {
    const policy = trainedOnWorkflow();
    const tokens = tokensOf(workflow);
    const prediction = policy.predict([tokens[0]!]);
    expect(prediction?.token).toBe(tokens[1]);
    expect(prediction?.confidence).toBe(1);
    expect(prediction?.source).toBe("ngram");
  });

  it("regenerates the full recorded trajectory from its seed", () => {
    const policy = trainedOnWorkflow();
    const tokens = tokensOf(workflow);
    const generated = policy.generate([tokens[0]!]);
    expect(generated).toEqual(tokens.slice(1));
    // The end sentinel is consumed, never emitted.
    expect(generated).not.toContain(MOVEMENT_END_TOKEN);
  });

  it("learns to stop: predicts the end sentinel after the last movement", () => {
    const policy = trainedOnWorkflow();
    const tokens = tokensOf(workflow);
    const prediction = policy.predict(tokens);
    expect(prediction?.token).toBe(MOVEMENT_END_TOKEN);
  });
});

describe("ngram movement policy — generalization", () => {
  it("generalizes an unseen movement via its tool family", () => {
    // Every "type" gesture is followed by a "tap send"; the model should apply
    // that learned family behaviour to a brand-new, unseen "type" movement.
    const dataset: MovementDataset = {
      sequences: [
        { tokens: [movementToken("device", "type subject"), movementToken("device", "tap send")] },
        { tokens: [movementToken("device", "type note"), movementToken("device", "tap send")] },
      ],
    };
    const policy = trainMovementModel(dataset);
    const unseen = movementToken("device", "type a totally new body");
    const prediction = policy.predict([unseen]);
    expect(prediction?.source).toBe("family");
    expect(prediction?.token).toBe(movementToken("device", "tap send"));
  });

  it("falls back to the global prior when there is no family signal", () => {
    const dataset: MovementDataset = {
      sequences: [{ tokens: [movementToken("device", "tap a"), movementToken("device", "tap b")] }],
    };
    const policy = trainMovementModel(dataset);
    const prediction = policy.predict([movementToken("keyboard", "unknown key")]);
    expect(prediction?.source).toBe("prior");
    expect(prediction).toBeDefined();
  });

  it("suppresses low-confidence predictions when minConfidence is set", () => {
    const dataset: MovementDataset = {
      sequences: [
        { tokens: [movementToken("device", "start"), movementToken("device", "left")] },
        { tokens: [movementToken("device", "start"), movementToken("device", "right")] },
      ],
    };
    const policy = trainMovementModel(dataset);
    // After "start" the next movement is a 50/50 split → confidence 0.5.
    expect(policy.predict([movementToken("device", "start")])?.confidence).toBe(0.5);
    expect(policy.predict([movementToken("device", "start")], { minConfidence: 0.9 })).toBeUndefined();
  });
});

describe("serialization round-trip", () => {
  it("restores an identical policy via the registry", () => {
    const dataset = buildMovementDataset([
      syntheticTrajectory("a", [
        { tool: "device", summary: "open" },
        { tool: "device", summary: "confirm" },
      ]),
    ]);
    const policy = trainMovementModel(dataset);
    const snapshot = policy.serialize();
    expect(snapshot.backend).toBe("ngram");

    const registry = defaultMovementModelRegistry();
    const restored = registry.restore(snapshot);

    const context = [movementToken("device", "open")];
    expect(restored.predict(context)).toEqual(policy.predict(context));
    expect(restored.generate(context)).toEqual(policy.generate(context));
    // Snapshot is plain JSON — a genuinely replayable artifact.
    expect(JSON.parse(JSON.stringify(snapshot))).toEqual(snapshot);
  });
});

describe("pluggable registry", () => {
  it("lists and resolves the default ngram backend", () => {
    const registry = defaultMovementModelRegistry();
    expect(registry.list()).toContain("ngram");
    expect(registry.has("ngram")).toBe(true);
    expect(registry.get("ngram")).toBeInstanceOf(NgramMovementModelBackend);
  });

  it("throws a helpful error for an unknown backend", () => {
    const registry = new MovementModelRegistry();
    expect(() => registry.get("on-device-mlx")).toThrow(/unknown movement-model backend/);
  });

  it("supports a custom backend implementation", () => {
    const registry = defaultMovementModelRegistry();
    registry.register({
      name: "constant",
      train: () => ({
        backend: "constant",
        predict: () => ({ token: "fixed", confidence: 1, order: 0, source: "prior" as const }),
        generate: () => ["fixed"],
        serialize: () => ({
          version: 1,
          backend: "constant",
          order: 1,
          transitions: [],
          toolFamilies: [],
          tokenTools: [],
        }),
      }),
      restore: () => {
        throw new Error("not supported");
      },
    });
    const policy = registry.train("constant", { sequences: [] });
    expect(policy.predict([])?.token).toBe("fixed");
  });
});

describe("generalization eval harness", () => {
  it("scores perfect top-1 accuracy on memorized sequences", () => {
    const workflow: MovementSequence = {
      tokens: [movementToken("device", "a"), movementToken("device", "b"), movementToken("device", "c")],
    };
    const policy = trainMovementModel({ sequences: [workflow, workflow] });
    const result = evaluateMovementPolicy(policy, [workflow]);
    expect(result.topOneAccuracy).toBe(1);
    expect(result.correct).toBe(result.predictions);
    expect(result.meanConfidence).toBeGreaterThan(0);
  });

  it("measures generalization on held-out but related sequences", () => {
    const training = {
      sequences: [
        { tokens: [movementToken("device", "type x"), movementToken("device", "tap send")] },
        { tokens: [movementToken("device", "type y"), movementToken("device", "tap send")] },
        { tokens: [movementToken("device", "type z"), movementToken("device", "tap send")] },
      ],
    };
    const policy = trainMovementModel(training);
    const heldOut: MovementSequence[] = [
      { tokens: [movementToken("device", "type brand new"), movementToken("device", "tap send")] },
    ];
    const result = evaluateMovementPolicy(policy, heldOut);
    // Family generalization should recover "tap send" for the unseen "type" input.
    expect(result.topOneAccuracy).toBeGreaterThan(0);
    expect(result.predictions).toBe(3); // context "", 1 token, 2 tokens (incl. end)
  });
});
