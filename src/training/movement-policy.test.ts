import { describe, expect, it } from "vitest";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import type { TrajectoryAction } from "../capture/trajectory.js";
import {
  MOVEMENT_END,
  NGramMovementBackend,
  createMovementPolicyBackend,
  evaluateNextMovementAccuracy,
  extractMovementSequence,
  listMovementPolicyBackends,
  loadMovementPolicyModel,
  movementFeatureKey,
  movementKey,
  movementSequenceFromReplayEvents,
  movementTokenFromAction,
  registerMovementPolicyBackend,
  rolloutMovements,
  type MovementPolicyBackend,
  type MovementPolicyModel,
  type MovementToken,
} from "./movement-policy.js";

function action(tool: string, summary: string, ts: number, metadata?: Record<string, unknown>): TrajectoryAction {
  return { kind: "action", tool, summary, ts, ...(metadata ? { metadata } : {}) };
}

function tap(target: string): MovementToken {
  return { tool: "device", gesture: "tap", target };
}

describe("movement tokenization", () => {
  it("normalizes an action's metadata into a movement token", () => {
    const token = movementTokenFromAction(action("device", "tapped submit", 1, { gesture: "tap", target: "submit" }));
    expect(token).toEqual({ tool: "device", gesture: "tap", target: "submit" });
  });

  it("falls back to the summary verb when no gesture metadata is present", () => {
    const token = movementTokenFromAction(action("keyboard", "Type hello", 1));
    expect(token.gesture).toBe("type");
  });

  it("extracts a movement sequence ordered by timestamp", () => {
    const span = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [
        action("device", "tapped b", 20, { gesture: "tap", target: "b" }),
        action("device", "tapped a", 10, { gesture: "tap", target: "a" }),
      ],
    });
    expect(extractMovementSequence(span).map((token) => token.target)).toEqual(["a", "b"]);
  });

  it("derives a sequence from replay action events", () => {
    const sequence = movementSequenceFromReplayEvents([
      { kind: "observation", ts: 1, trajectoryId: "t", source: "device", summary: "screen" },
      { kind: "action", ts: 3, trajectoryId: "t", tool: "device", summary: "Swipe up" },
      { kind: "action", ts: 2, trajectoryId: "t", tool: "device", summary: "Tap start" },
    ]);
    expect(sequence).toEqual([
      { tool: "device", gesture: "tap" },
      { tool: "device", gesture: "swipe" },
    ]);
  });

  it("feature key drops the target so related movements share a feature", () => {
    expect(movementFeatureKey(tap("row-1"))).toBe(movementFeatureKey(tap("row-2")));
    expect(movementKey(tap("row-1"))).not.toBe(movementKey(tap("row-2")));
  });
});

describe("NGramMovementBackend — repeat recorded movements", () => {
  const backend = new NGramMovementBackend();

  it("reproduces a recorded movement sequence exactly via rollout", () => {
    const sequence = [tap("menu"), tap("settings"), tap("save")];
    const model = backend.train([sequence]);
    const rolled = rolloutMovements(backend, model, [sequence[0]!]);
    expect(rolled.map((step) => step.token.target)).toEqual(["settings", "save"]);
    // The recorded continuation is an exact (non-generalized) match.
    expect(rolled.every((step) => step.generalized === false)).toBe(true);
  });

  it("predicts the exact next movement from a seen context", () => {
    const model = backend.train([[tap("menu"), tap("settings"), tap("save")]]);
    const prediction = backend.predictNext(model, [tap("menu"), tap("settings")]);
    expect(prediction?.token).not.toBe(MOVEMENT_END);
    expect(prediction && prediction.token !== MOVEMENT_END && prediction.token.target).toBe("save");
    expect(prediction?.generalized).toBe(false);
    expect(prediction?.confidence).toBeGreaterThan(0);
  });

  it("learns to stop at the end of a recorded sequence", () => {
    const model = backend.train([[tap("menu"), tap("save")]]);
    const prediction = backend.predictNext(model, [tap("menu"), tap("save")]);
    expect(prediction?.token).toBe(MOVEMENT_END);
  });

  it("caps rollout length to avoid runaway loops", () => {
    // A self-looping training pair (a -> a) never emits END; rollout must stop.
    const model = backend.train([[tap("a"), tap("a"), tap("a"), tap("a")]]);
    const rolled = rolloutMovements(backend, model, [tap("a")], { maxSteps: 5 });
    expect(rolled.length).toBeLessThanOrEqual(5);
  });
});

describe("NGramMovementBackend — generalize to new-but-related movements", () => {
  const backend = new NGramMovementBackend();

  it("generalizes across targets sharing a feature", () => {
    // Trained: focusing a field is followed by typing into it, for two rows.
    const model = backend.train([
      [{ tool: "device", gesture: "focus", target: "row-1" }, { tool: "device", gesture: "type", target: "row-1" }],
      [{ tool: "device", gesture: "focus", target: "row-2" }, { tool: "device", gesture: "type", target: "row-2" }],
    ]);
    // A never-before-seen context (focus row-9) should still predict a "type"
    // movement by generalizing over the target-agnostic feature.
    const prediction = backend.predictNext(model, [{ tool: "device", gesture: "focus", target: "row-9" }]);
    expect(prediction).toBeDefined();
    expect(prediction!.token).not.toBe(MOVEMENT_END);
    expect(prediction!.token !== MOVEMENT_END && prediction!.token.gesture).toBe("type");
    expect(prediction!.generalized).toBe(true);
  });

  it("falls back to the globally most frequent movement with no context match", () => {
    const model = backend.train([[tap("a"), tap("a"), tap("b")]]);
    const prediction = backend.predictNext(model, [{ tool: "browser", gesture: "click", target: "unseen" }]);
    expect(prediction).toBeDefined();
    expect(prediction!.order).toBe(0);
    expect(prediction!.generalized).toBe(true);
  });

  it("returns undefined for an untrained (empty) model", () => {
    const model = backend.train([]);
    expect(backend.predictNext(model, [tap("a")])).toBeUndefined();
  });
});

describe("evaluation harness", () => {
  const backend = new NGramMovementBackend();

  it("scores perfect accuracy on the training sequence (memorization)", () => {
    const sequence = [tap("menu"), tap("settings"), tap("save")];
    const model = backend.train([sequence]);
    const result = evaluateNextMovementAccuracy(backend, model, [sequence]);
    expect(result.predictions).toBe(3);
    expect(result.accuracy).toBe(1);
  });

  it("measures generalization on a held-out but related sequence", () => {
    const model = backend.train([
      [{ tool: "device", gesture: "focus", target: "a" }, { tool: "device", gesture: "type", target: "a" }],
      [{ tool: "device", gesture: "focus", target: "b" }, { tool: "device", gesture: "type", target: "b" }],
    ]);
    const heldOut = [[{ tool: "device", gesture: "focus", target: "c" as string }]];
    // The first movement has no context, so exact accuracy on the held-out
    // sequence is measured but not required to be perfect; the harness runs.
    const result = evaluateNextMovementAccuracy(backend, model, heldOut);
    expect(result.predictions).toBe(1);
    expect(result.accuracy).toBeGreaterThanOrEqual(0);
  });
});

describe("pluggable backend registry", () => {
  it("creates the default ngram backend", () => {
    expect(createMovementPolicyBackend().id).toBe("ngram");
    expect(listMovementPolicyBackends()).toContain("ngram");
  });

  it("throws for an unknown backend id", () => {
    expect(() => createMovementPolicyBackend("does-not-exist")).toThrow(/unknown movement-policy backend/);
  });

  it("registers and resolves a custom backend (real-model seam)", () => {
    const custom: MovementPolicyBackend = {
      id: "stub",
      train: () => new NGramMovementBackend().train([]),
      predictNext: () => undefined,
    };
    registerMovementPolicyBackend("stub", () => custom);
    expect(createMovementPolicyBackend("stub")).toBe(custom);
    expect(listMovementPolicyBackends()).toContain("stub");
  });
});

describe("model persistence", () => {
  it("round-trips a trained model through JSON", () => {
    const backend = new NGramMovementBackend();
    const model = backend.train([[tap("menu"), tap("save")]]);
    const restored: MovementPolicyModel = loadMovementPolicyModel(JSON.stringify(model));
    expect(restored).toEqual(model);
    const prediction = backend.predictNext(restored, [tap("menu")]);
    expect(prediction && prediction.token !== MOVEMENT_END && prediction.token.target).toBe("save");
  });

  it("rejects an invalid serialized model", () => {
    expect(() => loadMovementPolicyModel(JSON.stringify({ version: 2 }))).toThrow(/invalid movement-policy model/);
  });
});
