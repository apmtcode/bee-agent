import { describe, expect, it } from "vitest";
import { buildTrajectorySpan, type TrajectoryAction } from "../capture/trajectory.js";
import { buildReplayManifest } from "../capture/replay.js";
import {
  createMovementBackend,
  evaluateMovementPolicy,
  isMovementEndToken,
  MOVEMENT_BACKENDS,
  movementDatasetFromReplayManifests,
  movementDatasetFromTrajectories,
  movementPolicyFromSnapshot,
  movementSequenceFromTrajectory,
  movementTokenFromAction,
  movementTokenKey,
  NGramMovementBackend,
  registerMovementBackend,
  type MovementDataset,
  type MovementSequence,
  type MovementToken,
} from "./movement-model.js";

function tok(channel: string, action: string, extra: Partial<MovementToken> = {}): MovementToken {
  return { channel, action, ...extra };
}

function seq(id: string, tokens: MovementToken[]): MovementSequence {
  return { id, tokens };
}

// A small, structured "login flow" grammar used to synthesize training data.
function loginFlow(id: string): MovementSequence {
  return seq(id, [
    tok("mouse", "click", { target: "username" }),
    tok("keyboard", "type", { target: "username" }),
    tok("mouse", "click", { target: "password" }),
    tok("keyboard", "type", { target: "password" }),
    tok("mouse", "click", { target: "submit" }),
  ]);
}

describe("movement-model n-gram backend", () => {
  it("reproduces a recorded movement sequence exactly from its first token", async () => {
    const dataset: MovementDataset = { sequences: [loginFlow("a"), loginFlow("b"), loginFlow("c")] };
    const policy = await new NGramMovementBackend().train(dataset);

    const seed = [tok("mouse", "click", { target: "username" })];
    const rollout = policy.rollout(seed, { maxSteps: 10 });

    const full = loginFlow("expected").tokens;
    expect([seed[0], ...rollout].map(movementTokenKey)).toEqual(full.map(movementTokenKey));
  });

  it("predicts the next movement with a probability and reports backoff depth", async () => {
    const dataset: MovementDataset = { sequences: [loginFlow("a"), loginFlow("b")] };
    const policy = await new NGramMovementBackend(3).train(dataset);

    const prediction = policy.predictNext([
      tok("mouse", "click", { target: "username" }),
      tok("keyboard", "type", { target: "username" }),
    ]);

    expect(prediction).toBeDefined();
    expect(prediction?.token).toMatchObject({ channel: "mouse", action: "click", target: "password" });
    expect(prediction?.probability).toBeGreaterThan(0);
    // Full order-2 context ("click username" -> "type username") is present in training.
    expect(prediction?.contextOrderUsed).toBe(2);
  });

  it("generalizes to an unseen context by backing off to a shorter one", async () => {
    // In every training flow, "click next" is followed by "click submit",
    // regardless of what preceded "click next".
    const dataset: MovementDataset = {
      sequences: [
        seq("x", [
          tok("keyboard", "type", { target: "user" }),
          tok("mouse", "click", { target: "next" }),
          tok("mouse", "click", { target: "submit" }),
        ]),
        seq("y", [
          tok("mouse", "tap", { target: "menu" }),
          tok("mouse", "click", { target: "next" }),
          tok("mouse", "click", { target: "submit" }),
        ]),
      ],
    };
    const policy = await new NGramMovementBackend(3).train(dataset);

    // A never-seen prefix ("scroll page") precedes "click next": the full
    // order-2 context is unseen, but the model backs off to the seen order-1
    // context ("click next") and still predicts the common continuation.
    const prediction = policy.predictNext([
      tok("mouse", "scroll", { target: "page" }),
      tok("mouse", "click", { target: "next" }),
    ]);
    expect(prediction).toBeDefined();
    expect(prediction?.token).toMatchObject({ target: "submit" });
    // It could not match the full unseen context, so backoff depth is reduced.
    expect(prediction?.contextOrderUsed).toBe(1);
  });

  it("terminates a rollout at the learned end of sequence", async () => {
    const dataset: MovementDataset = { sequences: [loginFlow("a"), loginFlow("b")] };
    const policy = await new NGramMovementBackend().train(dataset);

    const rollout = policy.rollout([], { maxSteps: 50 });
    // Model learned a 5-step flow; it should stop there, not run to maxSteps.
    expect(rollout.length).toBe(5);
    expect(rollout.every((token) => !isMovementEndToken(token))).toBe(true);
  });

  it("caps runaway rollouts at maxSteps", async () => {
    // A self-looping sequence with no clear end signal beyond the loop.
    const dataset: MovementDataset = {
      sequences: [seq("loop", [tok("mouse", "move"), tok("mouse", "move"), tok("mouse", "move")])],
    };
    const policy = await new NGramMovementBackend(2).train(dataset);
    const rollout = policy.rollout([tok("mouse", "move")], { maxSteps: 4 });
    expect(rollout.length).toBeLessThanOrEqual(4);
  });

  it("exposes a sorted vocabulary excluding sentinels", async () => {
    const dataset: MovementDataset = { sequences: [loginFlow("a")] };
    const policy = await new NGramMovementBackend().train(dataset);
    expect(policy.vocabulary).toContain(movementTokenKey(tok("mouse", "click", { target: "submit" })));
    expect(policy.vocabulary.every((key) => !key.includes("meta"))).toBe(true);
    expect([...policy.vocabulary]).toEqual([...policy.vocabulary].sort());
  });

  it("round-trips through a snapshot with identical predictions", async () => {
    const dataset: MovementDataset = { sequences: [loginFlow("a"), loginFlow("b")] };
    const policy = await new NGramMovementBackend().train(dataset);
    const snapshot = policy.snapshot();

    // Snapshot must survive JSON serialization (it is persisted as an artifact).
    const restored = movementPolicyFromSnapshot(JSON.parse(JSON.stringify(snapshot)));

    const context = [tok("mouse", "click", { target: "username" })];
    expect(restored.predictNext(context)).toEqual(policy.predictNext(context));
    expect(restored.rollout(context)).toEqual(policy.rollout(context));
    expect(restored.order).toBe(policy.order);
  });
});

describe("capture-pipeline bridges", () => {
  function deviceAction(ts: number, gesture: string, target: string, direction?: string): TrajectoryAction {
    return {
      kind: "action",
      tool: "device",
      summary: `${gesture} ${target}`,
      ts,
      metadata: { gesture, target, ...(direction ? { direction } : {}) },
    };
  }

  it("derives structured tokens from trajectory action metadata", () => {
    const action = deviceAction(10, "swipe", "gallery", "left");
    const token = movementTokenFromAction(action);
    expect(token).toEqual({ channel: "device", action: "swipe", direction: "left", target: "gallery" });
  });

  it("falls back to the summary verb when metadata lacks a gesture", () => {
    const action: TrajectoryAction = { kind: "action", tool: "browser", summary: "Clicked the Login button", ts: 1 };
    expect(movementTokenFromAction(action)).toEqual({ channel: "browser", action: "clicked" });
  });

  it("orders trajectory actions by timestamp into a sequence", () => {
    const trajectory = buildTrajectorySpan({
      id: "traj-1",
      sessionId: "s1",
      actions: [deviceAction(30, "tap", "c"), deviceAction(10, "tap", "a"), deviceAction(20, "tap", "b")],
    });
    const sequence = movementSequenceFromTrajectory(trajectory);
    expect(sequence.tokens.map((token) => token.target)).toEqual(["a", "b", "c"]);
  });

  it("builds a dataset from trajectories and drops empty ones", () => {
    const withActions = buildTrajectorySpan({ id: "t1", sessionId: "s", actions: [deviceAction(1, "tap", "a")] });
    const withoutActions = buildTrajectorySpan({ id: "t2", sessionId: "s" });
    const dataset = movementDatasetFromTrajectories([withActions, withoutActions]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0].id).toBe("t1");
  });

  it("builds a dataset from replay manifests", () => {
    const trajectory = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [deviceAction(1, "tap", "a"), deviceAction(2, "swipe", "b", "up")],
    });
    const manifest = buildReplayManifest({ sessionId: "s1", transcript: [], trajectories: [trajectory] });
    const dataset = movementDatasetFromReplayManifests([manifest]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0].tokens.map((token) => token.action)).toEqual(["tap", "swipe"]);
  });

  it("trains directly on a trajectory-derived dataset", async () => {
    const trajectories = [1, 2, 3].map((n) =>
      buildTrajectorySpan({
        id: `t${n}`,
        sessionId: "s",
        actions: [deviceAction(1, "tap", "menu"), deviceAction(2, "tap", "settings"), deviceAction(3, "tap", "save")],
      }),
    );
    const dataset = movementDatasetFromTrajectories(trajectories);
    const policy = await new NGramMovementBackend().train(dataset);
    const rollout = policy.rollout([tok("device", "tap", { target: "menu" })]);
    expect(rollout.map((token) => token.target)).toEqual(["settings", "save"]);
  });
});

describe("generalization evaluation harness", () => {
  it("scores perfect fidelity when held-out sequences follow a learned pattern", async () => {
    const train: MovementDataset = { sequences: [loginFlow("a"), loginFlow("b"), loginFlow("c")] };
    const policy = await new NGramMovementBackend().train(train);
    const heldOut = [loginFlow("held-1"), loginFlow("held-2")];

    const evaluation = evaluateMovementPolicy(policy, heldOut, { promptRatio: 0.4 });
    expect(evaluation.averageFidelity).toBe(1);
    expect(evaluation.exactSequences).toBe(2);
    expect(evaluation.perSequence[0].matched).toBe(evaluation.perSequence[0].expected.length);
  });

  it("reports partial fidelity for sequences that diverge from training", async () => {
    const train: MovementDataset = { sequences: [loginFlow("a"), loginFlow("b")] };
    const policy = await new NGramMovementBackend().train(train);
    // A held-out flow that shares the prefix but ends differently.
    const divergent = seq("d", [
      tok("mouse", "click", { target: "username" }),
      tok("keyboard", "type", { target: "username" }),
      tok("mouse", "click", { target: "cancel" }),
    ]);

    const evaluation = evaluateMovementPolicy(policy, [divergent], { promptRatio: 0.6 });
    expect(evaluation.averageFidelity).toBeGreaterThanOrEqual(0);
    expect(evaluation.averageFidelity).toBeLessThan(1);
  });

  it("ignores empty held-out sequences", async () => {
    const policy = await new NGramMovementBackend().train({ sequences: [loginFlow("a")] });
    const evaluation = evaluateMovementPolicy(policy, [seq("empty", [])]);
    expect(evaluation.perSequence).toHaveLength(0);
    expect(evaluation.averageFidelity).toBe(0);
  });
});

describe("pluggable backend registry", () => {
  it("creates the default ngram backend", () => {
    const backend = createMovementBackend();
    expect(backend.id).toBe("ngram");
  });

  it("throws for an unknown backend id", () => {
    expect(() => createMovementBackend("does-not-exist")).toThrow(/unknown movement backend/);
  });

  it("allows registering a custom backend", async () => {
    registerMovementBackend("passthrough", () => new NGramMovementBackend(1));
    expect(MOVEMENT_BACKENDS["passthrough"]).toBeDefined();
    const backend = createMovementBackend("passthrough");
    const policy = await backend.train({ sequences: [loginFlow("a")] });
    expect(policy.order).toBe(1);
    delete MOVEMENT_BACKENDS["passthrough"];
  });
});
