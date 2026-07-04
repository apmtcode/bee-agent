import { describe, expect, it } from "vitest";
import { buildReplayManifest } from "../capture/replay.js";
import { buildTrajectorySpan, type TrajectoryAction, type TrajectorySpan } from "../capture/trajectory.js";
import {
  NgramMovementBackend,
  buildMovementDatasetFromReplays,
  buildMovementDatasetFromTrajectories,
  createDefaultMovementModelRegistry,
  evaluateMovementModel,
  movementTokenFromAction,
  movementTokenKey,
  movementTokenSignature,
  restoreMovementModel,
  type MovementDataset,
} from "./movement-model.js";

function action(tool: string, verb: string, target?: string, direction?: string, ts = 0): TrajectoryAction {
  return {
    kind: "action",
    tool,
    summary: `${verb} ${target ?? ""}`.trim(),
    ts,
    metadata: {
      ...(tool === "browser" ? { action: verb } : { gesture: verb }),
      ...(target ? { target } : {}),
      ...(direction ? { direction } : {}),
    },
  };
}

/** A synthetic "open → scroll down → tap" workflow on a given target family. */
function workflowTrajectory(id: string, item: string): TrajectorySpan {
  return buildTrajectorySpan({
    id,
    sessionId: `session-${id}`,
    actions: [
      action("device", "tap", `open-${item}`, undefined, 1),
      action("device", "scroll", undefined, "down", 2),
      action("device", "tap", `confirm-${item}`, undefined, 3),
    ],
  });
}

describe("movementTokenFromAction", () => {
  it("prefers structured metadata and normalizes verb aliases", () => {
    const token = movementTokenFromAction({
      tool: "device",
      summary: "tapped submit-button",
      metadata: { gesture: "tap", target: "submit-button" },
    });
    expect(token).toEqual({ tool: "device", verb: "tap", target: "submit-button" });
  });

  it("falls back to the summary's leading verb when metadata is absent", () => {
    const token = movementTokenFromAction({ tool: "browser", summary: "navigated to /inbox" });
    expect(token.verb).toBe("navigate");
    expect(token.tool).toBe("browser");
  });
});

describe("dataset builders", () => {
  it("orders trajectory actions by timestamp", () => {
    const trajectory = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [action("device", "tap", "b", undefined, 3), action("device", "scroll", undefined, "down", 1)],
    });
    const dataset = buildMovementDatasetFromTrajectories([trajectory]);
    expect(dataset.sequences[0]!.tokens.map((t) => t.verb)).toEqual(["scroll", "tap"]);
  });

  it("reconstructs sequences from a replay manifest", () => {
    const trajectory = workflowTrajectory("t1", "cart");
    const replay = buildReplayManifest({ sessionId: "s1", transcript: [], trajectories: [trajectory] });
    const dataset = buildMovementDatasetFromReplays([replay]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]!.tokens.map((t) => t.verb)).toEqual(["tap", "scroll", "tap"]);
  });
});

describe("NgramMovementBackend — repeat (objective 2c)", () => {
  it("reproduces a recorded movement sequence via greedy generation", () => {
    const trajectory = workflowTrajectory("t1", "cart");
    const dataset = buildMovementDatasetFromTrajectories([trajectory]);
    const model = new NgramMovementBackend().train(dataset);

    const replayed = model.generate([], trajectory.actions.length);
    expect(replayed.map(movementTokenKey)).toEqual(
      dataset.sequences[0]!.tokens.map(movementTokenKey),
    );
    // None of these should be generalized — they are exact recordings.
    const first = model.predictNext([]);
    expect(first?.generalized).toBe(false);
  });

  it("predicts the recorded continuation from a mid-sequence prefix", () => {
    const dataset = buildMovementDatasetFromTrajectories([workflowTrajectory("t1", "cart")]);
    const model = new NgramMovementBackend().train(dataset);
    const prefix = dataset.sequences[0]!.tokens.slice(0, 1);
    const prediction = model.predictNext(prefix);
    expect(prediction?.token.verb).toBe("scroll");
    expect(prediction?.token.direction).toBe("down");
    expect(prediction?.generalized).toBe(false);
  });
});

describe("NgramMovementBackend — generalize (objective 2d)", () => {
  it("transfers the learned workflow structure onto an unseen target", () => {
    // Train on several instances of the same workflow over different items.
    const dataset = buildMovementDatasetFromTrajectories([
      workflowTrajectory("t1", "cart"),
      workflowTrajectory("t2", "profile"),
      workflowTrajectory("t3", "settings"),
    ]);
    const model = new NgramMovementBackend().train(dataset);

    // A brand-new target never seen in training.
    const novelOpen = movementTokenFromAction({
      tool: "device",
      summary: "tap open-billing",
      metadata: { gesture: "tap", target: "open-billing" },
    });
    const prediction = model.predictNext([novelOpen]);
    expect(prediction).toBeDefined();
    // The concrete (tap, open-billing) context was never seen, so the model
    // must fall back to the abstract signature model — generalization.
    expect(prediction!.generalized).toBe(true);
    expect(prediction!.token.verb).toBe("scroll");
    expect(prediction!.token.direction).toBe("down");
  });

  it("scores generalization on a held-out related trajectory via the eval harness", () => {
    const train = buildMovementDatasetFromTrajectories([
      workflowTrajectory("t1", "cart"),
      workflowTrajectory("t2", "profile"),
      workflowTrajectory("t3", "settings"),
    ]);
    const model = new NgramMovementBackend().train(train);
    const heldOut: MovementDataset = buildMovementDatasetFromTrajectories([workflowTrajectory("t9", "billing")]);

    const result = evaluateMovementModel(model, heldOut);
    // The middle "scroll down" is structurally learnable on the novel target.
    expect(result.signatureMatches).toBeGreaterThan(0);
    expect(result.generalizedSignatureMatches).toBeGreaterThan(0);
    expect(result.signatureAccuracy).toBeGreaterThanOrEqual(result.exactAccuracy);
  });
});

describe("snapshot round-trip (inference format)", () => {
  it("restores an equivalent model from a snapshot", () => {
    const dataset = buildMovementDatasetFromTrajectories([
      workflowTrajectory("t1", "cart"),
      workflowTrajectory("t2", "profile"),
    ]);
    const model = new NgramMovementBackend().train(dataset);
    const restored = restoreMovementModel(model.snapshot());

    const seed = dataset.sequences[0]!.tokens.slice(0, 1);
    expect(restored.predictNext(seed)?.token).toEqual(model.predictNext(seed)?.token);
    expect(restored.generate([], 3).map(movementTokenKey)).toEqual(model.generate([], 3).map(movementTokenKey));
    // Snapshot must be JSON-serializable (the on-disk dataset format).
    expect(() => JSON.stringify(model.snapshot())).not.toThrow();
  });
});

describe("MovementModelRegistry (pluggable backend seam)", () => {
  it("resolves the default n-gram backend by name", () => {
    const registry = createDefaultMovementModelRegistry();
    expect(registry.list()).toContain("ngram-markov");
    const backend = registry.get("ngram-markov");
    expect(backend).toBeDefined();
    const model = backend!.train(buildMovementDatasetFromTrajectories([workflowTrajectory("t1", "cart")]));
    expect(model.backend).toBe("ngram-markov");
  });

  it("computes stable token keys and signatures", () => {
    const token = { tool: "device", verb: "scroll", direction: "down" } as const;
    expect(movementTokenKey(token)).toBe("device|scroll|*|down");
    expect(movementTokenSignature(token)).toBe("device|scroll|down");
  });
});
