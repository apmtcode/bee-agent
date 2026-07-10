import { describe, expect, it } from "vitest";
import {
  MarkovMovementBackend,
  MOVEMENT_END,
  movementSequenceFromReplayEvents,
  movementSequenceFromTrajectory,
  movementTokenFromAction,
  type MovementDataset,
} from "./movement-policy.js";
import {
  DEFAULT_MOVEMENT_TEMPLATES,
  evaluateMovementPolicy,
  generateSyntheticMovementDataset,
  splitMovementDataset,
} from "./synthetic.js";
import type { ReplayTimelineEvent } from "../../capture/replay.js";
import type { TrajectorySpan } from "../../capture/trajectory.js";

const backend = new MarkovMovementBackend();

function dataset(sequences: string[][]): MovementDataset {
  return {
    version: 1,
    sequences: sequences.map((tokens, index) => ({ id: `seq-${index}`, tokens })),
  };
}

describe("MarkovMovementBackend training + replay (objective 2c)", () => {
  it("repeats a recorded movement exactly by rolling out from its first move", () => {
    const model = backend.train(dataset([["x:click:one", "y:type:two", "z:click:three"]]), { order: 3 });
    const rolled = backend.generate(model, { seed: ["x:click:one"] });
    expect(rolled).toEqual(["x:click:one", "y:type:two", "z:click:three"]);
    // Rollout stops at the END sentinel rather than emitting it.
    expect(rolled).not.toContain(MOVEMENT_END);
  });

  it("ranks next-move candidates deterministically by probability then token", () => {
    const model = backend.train(dataset([["a", "b"], ["a", "c"]]), { order: 2 });
    const predictions = backend.predictNext(model, ["a"]);
    expect(predictions.map((p) => p.token)).toEqual(["b", "c"]);
    expect(predictions[0]?.probability).toBeCloseTo(0.5, 5);
    expect(predictions.every((p) => p.backoffOrder === 1)).toBe(true);
  });

  it("records vocabulary without the END sentinel", () => {
    const model = backend.train(dataset([["a", "b"]]), { order: 2 });
    expect(model.vocabulary).toEqual(["a", "b"]);
    expect(model.vocabulary).not.toContain(MOVEMENT_END);
  });
});

describe("generalization via backoff (objective 2d)", () => {
  it("predicts a related next move for an unseen high-order context by backing off", () => {
    const model = backend.train(dataset([["p", "m", "n", "done"], ["q", "m", "n", "done"]]), { order: 3 });
    // "z m n" was never seen at order 3, but the "m n" suffix always precedes "done".
    const [best] = backend.predictNext(model, ["z", "m", "n"]);
    expect(best?.token).toBe("done");
    expect(best?.backoffOrder).toBe(2);
  });

  it("always yields a candidate for a known vocabulary via unigram fallback", () => {
    const model = backend.train(dataset([["only:one:move"]]), { order: 3 });
    const predictions = backend.predictNext(model, ["totally", "unseen", "context"]);
    expect(predictions.length).toBeGreaterThan(0);
    expect(predictions[0]?.backoffOrder).toBe(0);
  });
});

describe("movement token derivation", () => {
  it("builds a stable token from an action's tool/gesture/target", () => {
    expect(
      movementTokenFromAction({
        tool: "device",
        summary: "tapped submit",
        metadata: { gesture: "tap", target: "Submit Button" },
      }),
    ).toBe("device:tap:submit-button");
  });

  it("falls back to a slugged summary when metadata is sparse", () => {
    expect(movementTokenFromAction({ tool: "mouse", summary: "Click Save!", metadata: {} })).toBe(
      "mouse:act:click-save",
    );
  });

  it("derives an ordered sequence from a trajectory's actions", () => {
    const trajectory: TrajectorySpan = {
      id: "t1",
      sessionId: "s1",
      createdAt: "2026-07-10T00:00:00.000Z",
      captureTier: "app",
      observations: [],
      actions: [
        { kind: "action", tool: "mouse", summary: "click send", ts: 20, metadata: { gesture: "click", target: "send" } },
        { kind: "action", tool: "app", summary: "focus box", ts: 10, metadata: { gesture: "focus", target: "box" } },
      ],
    };
    const sequence = movementSequenceFromTrajectory(trajectory);
    expect(sequence.tokens).toEqual(["app:focus:box", "mouse:click:send"]);
  });

  it("derives a sequence from replay action events only", () => {
    const events: ReplayTimelineEvent[] = [
      { kind: "observation", ts: 1, trajectoryId: "t", source: "os", summary: "window" },
      { kind: "action", ts: 3, trajectoryId: "t", tool: "keyboard", summary: "type hi" },
      { kind: "action", ts: 2, trajectoryId: "t", tool: "mouse", summary: "click ok" },
    ];
    const sequence = movementSequenceFromReplayEvents("r1", events);
    expect(sequence.tokens).toEqual(["mouse:act:click-ok", "keyboard:act:type-hi"]);
  });
});

describe("synthetic dataset generator", () => {
  it("is deterministic for a fixed seed", () => {
    const a = generateSyntheticMovementDataset({ seed: 42 });
    const b = generateSyntheticMovementDataset({ seed: 42 });
    expect(a).toEqual(b);
  });

  it("emits instancesPerTemplate sequences per template", () => {
    const generated = generateSyntheticMovementDataset({ seed: 7, instancesPerTemplate: 4 });
    expect(generated.sequences).toHaveLength(DEFAULT_MOVEMENT_TEMPLATES.length * 4);
    for (const sequence of generated.sequences) {
      expect(sequence.tokens.length).toBeGreaterThan(0);
    }
  });

  it("splits deterministically into disjoint train / held-out partitions", () => {
    const generated = generateSyntheticMovementDataset({ seed: 3, instancesPerTemplate: 6 });
    const { train, heldOut } = splitMovementDataset(generated, 0.3);
    expect(train.sequences.length + heldOut.sequences.length).toBe(generated.sequences.length);
    expect(heldOut.sequences.length).toBeGreaterThan(0);
    const trainIds = new Set(train.sequences.map((s) => s.id));
    expect(heldOut.sequences.some((s) => trainIds.has(s.id))).toBe(false);
  });
});

describe("generalization eval harness", () => {
  it("scores deterministic replay + prediction fidelity", () => {
    const model = backend.train(dataset([["a", "b", "c"], ["a", "b", "c"]]), { order: 3 });
    const heldOut = dataset([["a", "b", "c"]]);
    const evaluation = evaluateMovementPolicy(backend, model, heldOut);
    expect(evaluation.sequenceCount).toBe(1);
    expect(evaluation.tokenCount).toBe(3);
    // i=0 falls back to unigram (misses); i=1,i=2 predict from context (hit).
    expect(evaluation.nextTokenAccuracy).toBeCloseTo(2 / 3, 5);
    expect(evaluation.exactReplayRate).toBe(1);
    expect(evaluation.backoffRate).toBe(0);
  });

  it("learns real structure from synthetic data (held-out accuracy beats chance)", () => {
    const generated = generateSyntheticMovementDataset({ seed: 11, instancesPerTemplate: 10 });
    const { train, heldOut } = splitMovementDataset(generated, 0.3);
    const model = backend.train(train, { order: 3 });
    const evaluation = evaluateMovementPolicy(backend, model, heldOut);
    // Each step offers 2 related tokens, so chance ≈ 0.5; a learned policy that
    // conditions on the prefix should clear that comfortably.
    expect(evaluation.nextTokenAccuracy).toBeGreaterThan(0.5);
    expect(evaluation.exactReplayRate).toBeGreaterThan(0);
  });
});
