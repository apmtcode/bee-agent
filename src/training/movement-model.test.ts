import { describe, expect, it } from "vitest";
import type { ReplayManifest } from "../capture/replay.js";
import { buildReplayManifest } from "../capture/replay.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import {
  DeterministicMarkovMovementBackend,
  MOVEMENT_END_TOKEN,
  MovementModelTrainer,
  movementSequenceFromReplayManifest,
  movementSequenceFromTrajectory,
  movementTokenFromReplayEvent,
  type MovementSequence,
} from "./movement-model.js";

function seq(id: string, tokens: string[]): MovementSequence {
  return { id, tokens };
}

describe("DeterministicMarkovMovementBackend", () => {
  it("reproduces a recorded movement sequence exactly (objective #2c)", () => {
    const backend = new DeterministicMarkovMovementBackend();
    const recorded = seq("t1", ["open-app", "click-field", "type-name", "click-submit"]);
    const artifact = backend.train([recorded]);

    // From an empty seed the model should replay the full recorded trajectory.
    const replayed = backend.generate(artifact, []);
    expect(replayed).toEqual(recorded.tokens);
  });

  it("is fully deterministic: identical inputs yield identical artifacts", () => {
    const backend = new DeterministicMarkovMovementBackend();
    const examples = [seq("a", ["x", "y", "z"]), seq("b", ["x", "y", "w"])];
    const first = backend.train(examples);
    const second = backend.train(examples);
    expect(JSON.stringify(first)).toEqual(JSON.stringify(second));
  });

  it("breaks argmax ties on token order for reproducibility", () => {
    const backend = new DeterministicMarkovMovementBackend();
    // After context "shared", "close" and "save" each appear once -> tie.
    const artifact = backend.train([
      seq("a", ["shared", "save"]),
      seq("b", ["shared", "close"]),
    ]);
    const prediction = backend.predictNext(artifact, ["shared"]);
    expect(prediction.token).toBe("close"); // "close" < "save"
    expect(prediction.score).toBeCloseTo(0.5, 5);
    expect(prediction.candidates.map((c) => c.token)).toEqual(["close", "save"]);
  });

  it("generalizes to an unseen prefix via backoff to a shorter context (objective #2d)", () => {
    const backend = new DeterministicMarkovMovementBackend();
    const artifact = backend.train(
      [
        seq("a", ["open", "edit", "save"]),
        seq("b", ["open", "edit", "save"]),
        seq("c", ["launch", "edit", "close"]),
      ],
      { order: 2 },
    );

    // "restart edit" was never recorded, but "edit" -> "save" dominates order-1.
    const prediction = backend.predictNext(artifact, ["restart", "edit"]);
    expect(prediction.token).toBe("save");
    // Full order-2 context ["restart","edit"] is unseen, so it backed off.
    expect(prediction.contextOrder).toBeLessThan(2);
    expect(prediction.contextOrder).toBeGreaterThanOrEqual(0);
  });

  it("higher order reproduces more exactly; lower order generalizes more", () => {
    const backend = new DeterministicMarkovMovementBackend();
    const examples = [
      seq("a", ["a1", "shared", "a2"]),
      seq("b", ["b1", "shared", "b2"]),
    ];
    // order 1: context is only "shared", which is ambiguous (a2 vs b2 tie).
    const order1 = backend.train(examples, { order: 1 });
    const p1 = backend.predictNext(order1, ["a1", "shared"]);
    expect(p1.token).toBe("a2"); // "a2" < "b2" tie-break, ignores "a1"
    // order 2: context ["a1","shared"] uniquely determines "a2".
    const order2 = backend.train(examples, { order: 2 });
    const p2 = backend.predictNext(order2, ["a1", "shared"]);
    expect(p2.token).toBe("a2");
    expect(p2.contextOrder).toBe(2);
    expect(p2.score).toBeCloseTo(1, 5);
  });

  it("returns an empty prediction for a model with no data", () => {
    const backend = new DeterministicMarkovMovementBackend();
    const artifact = backend.train([]);
    const prediction = backend.predictNext(artifact, ["anything"]);
    expect(prediction.token).toBeUndefined();
    expect(prediction.contextOrder).toBe(-1);
    expect(prediction.candidates).toEqual([]);
    expect(backend.generate(artifact, [])).toEqual([]);
  });

  it("respects maxSteps and the end token when generating", () => {
    const backend = new DeterministicMarkovMovementBackend();
    const artifact = backend.train([seq("t", ["m1", "m2", "m3", "m4"])]);
    expect(backend.generate(artifact, [], { maxSteps: 2 })).toEqual(["m1", "m2"]);
    // The learned end token is never emitted into the output.
    const full = backend.generate(artifact, []);
    expect(full).not.toContain(MOVEMENT_END_TOKEN);
    expect(full).toEqual(["m1", "m2", "m3", "m4"]);
  });

  it("serializes to plain JSON and reloads without behavior change", () => {
    const backend = new DeterministicMarkovMovementBackend();
    const artifact = backend.train([seq("t", ["p", "q", "r"])]);
    const roundTripped = JSON.parse(JSON.stringify(artifact));
    expect(backend.generate(roundTripped, [])).toEqual(["p", "q", "r"]);
    expect(artifact.vocabulary).toEqual(["p", "q", "r"]);
    expect(artifact.exampleCount).toBe(1);
    expect(artifact.tokenCount).toBe(3);
  });
});

describe("MovementModelTrainer", () => {
  it("trains, replays, and reports high fidelity on the training distribution", () => {
    const trainer = new MovementModelTrainer();
    const recorded = seq("t1", ["focus-window", "drag-icon", "drop-icon", "confirm"]);
    const artifact = trainer.train([recorded]);

    expect(trainer.backendName).toBe("deterministic-markov");
    expect(trainer.replay(artifact)).toEqual(recorded.tokens);

    const fidelity = trainer.evaluate(artifact, [recorded]);
    expect(fidelity.sequences).toBe(1);
    expect(fidelity.tokens).toBe(4);
    expect(fidelity.accuracy).toBeCloseTo(1, 5);
    expect(fidelity.perSequence[0].id).toBe("t1");
  });

  it("replays from a seed prefix (partial continuation)", () => {
    const trainer = new MovementModelTrainer();
    const artifact = trainer.train([seq("t", ["s1", "s2", "s3", "s4"])], { order: 1 });
    expect(trainer.replay(artifact, ["s2"])).toEqual(["s2", "s3", "s4"]);
  });

  it("measures partial fidelity on held-out but related sequences", () => {
    const trainer = new MovementModelTrainer();
    const artifact = trainer.train(
      [seq("a", ["open", "edit", "save"]), seq("b", ["open", "edit", "save"])],
      { order: 2 },
    );
    // Related held-out sequence shares the "open"/"edit" openings but diverges.
    const heldOut = seq("h", ["open", "edit", "discard"]);
    const fidelity = trainer.evaluate(artifact, [heldOut]);
    expect(fidelity.tokens).toBe(3);
    // "open" and "edit" are predicted correctly; "discard" is not -> 2/3.
    expect(fidelity.correct).toBe(2);
    expect(fidelity.accuracy).toBeCloseTo(2 / 3, 5);
  });

  it("accepts a custom pluggable backend", () => {
    const artifact = {
      version: 1 as const,
      backend: "stub",
      order: 1,
      vocabulary: [],
      transitions: [],
      exampleCount: 0,
      tokenCount: 0,
    };
    const trainer = new MovementModelTrainer({
      backend: {
        name: "stub",
        train: () => artifact,
        predictNext: () => ({ token: "fixed", score: 1, contextOrder: 0, candidates: [] }),
        generate: () => ["fixed", "fixed"],
      },
    });
    expect(trainer.backendName).toBe("stub");
    expect(trainer.train([])).toBe(artifact);
    expect(trainer.replay(artifact, ["seed"])).toEqual(["seed", "fixed", "fixed"]);
  });
});

describe("movement tokenization bridges", () => {
  it("tokenizes a trajectory's observations and actions in timestamp order", () => {
    const trajectory = buildTrajectorySpan({
      id: "traj-1",
      sessionId: "sess-1",
      observations: [{ kind: "observation", source: "OS", summary: "Focused  Editor", ts: 10 }],
      actions: [{ kind: "action", tool: "Device", summary: "Tapped Submit", ts: 20 }],
    });
    const sequence = movementSequenceFromTrajectory(trajectory);
    expect(sequence.id).toBe("traj-1");
    expect(sequence.tokens).toEqual(["obs:os:focused editor", "action:device:tapped submit"]);
  });

  it("drops transcript events and preserves timeline order from a replay manifest", () => {
    const trajectory = buildTrajectorySpan({
      id: "traj-2",
      sessionId: "sess-2",
      observations: [{ kind: "observation", source: "device", summary: "app active", ts: 5 }],
      actions: [{ kind: "action", tool: "device", summary: "swiped up", ts: 6 }],
    });
    const manifest: ReplayManifest = buildReplayManifest({
      sessionId: "sess-2",
      transcript: [
        { id: "m1", message: { role: "user", content: "hi", timestamp: 1 } },
      ],
      trajectories: [trajectory],
    });
    const sequence = movementSequenceFromReplayManifest(manifest);
    expect(sequence.id).toBe("sess-2");
    expect(sequence.tokens).toEqual(["obs:device:app active", "action:device:swiped up"]);
  });

  it("maps replay event kinds to canonical tokens (transcript -> undefined)", () => {
    expect(
      movementTokenFromReplayEvent({ kind: "action", ts: 1, trajectoryId: "t", tool: "Mouse", summary: "Click OK" }),
    ).toBe("action:mouse:click ok");
    expect(
      movementTokenFromReplayEvent({ kind: "observation", ts: 1, trajectoryId: "t", source: "UI", summary: "Panel Open" }),
    ).toBe("obs:ui:panel open");
    expect(
      movementTokenFromReplayEvent({ kind: "transcript", ts: 1, messageId: "m", role: "user", content: "hi" }),
    ).toBeUndefined();
  });

  it("feeds a synthetic recording end-to-end into the trainer", () => {
    const trajectory = buildTrajectorySpan({
      id: "traj-e2e",
      sessionId: "sess-e2e",
      observations: [{ kind: "observation", source: "os", summary: "opened editor", ts: 1 }],
      actions: [
        { kind: "action", tool: "keyboard", summary: "typed hello", ts: 2 },
        { kind: "action", tool: "keyboard", summary: "pressed save", ts: 3 },
      ],
    });
    const sequence = movementSequenceFromTrajectory(trajectory);
    const trainer = new MovementModelTrainer();
    const artifact = trainer.train([sequence]);
    expect(trainer.replay(artifact)).toEqual(sequence.tokens);
  });
});
