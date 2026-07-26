import { describe, expect, it } from "vitest";
import {
  NgramMovementBackend,
  rolloutMovement,
  sequenceFromReplayManifest,
  sequenceFromTrajectory,
  type MovementEvent,
  type MovementSequence,
} from "./movement-model.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import type { ReplayManifest } from "../capture/replay.js";

function obs(source: string, summary: string): MovementEvent {
  return { kind: "observation", source, summary };
}

function act(tool: string, summary: string): MovementEvent {
  return { kind: "action", tool, summary };
}

/** Synthetic "open → edit → save file X" recordings — a related family. */
function fileEditSequence(id: string, file: string): MovementSequence {
  return {
    id,
    events: [
      obs("editor", `viewing ${file}`),
      act("open", file),
      act("edit", file),
      act("save", file),
    ],
  };
}

describe("NgramMovementBackend", () => {
  it("repeats a recorded movement exactly from its own prefix", () => {
    const backend = new NgramMovementBackend();
    const policy = backend.train([fileEditSequence("t1", "alpha.ts")], { order: 3 });

    // Given the recorded prefix, it reproduces the recorded next action.
    const afterOpen = policy.predict([obs("editor", "viewing alpha.ts"), act("open", "alpha.ts")]);
    expect(afterOpen?.action).toEqual(act("edit", "alpha.ts"));
    expect(afterOpen?.source).toBe("exact");
    expect(afterOpen?.confidence).toBe(1);

    const afterEdit = policy.predict([act("open", "alpha.ts"), act("edit", "alpha.ts")]);
    expect(afterEdit?.action).toEqual(act("save", "alpha.ts"));
    expect(afterEdit?.source).toBe("exact");
  });

  it("generalizes a learned pattern to a new-but-related instance", () => {
    const backend = new NgramMovementBackend();
    const policy = backend.train(
      [fileEditSequence("t1", "alpha.ts"), fileEditSequence("t2", "beta.ts")],
      { order: 3 },
    );

    // "gamma.ts" was never recorded, but the open→edit→save *pattern* was.
    const prediction = policy.predict([act("open", "gamma.ts"), act("edit", "gamma.ts")]);
    expect(prediction?.source).toBe("generalized");
    // It correctly predicts the "save" tool (generalization of the pattern).
    expect(prediction?.action.tool).toBe("save");
  });

  it("backs off to shorter context when the full window is unseen", () => {
    const backend = new NgramMovementBackend();
    const policy = backend.train([fileEditSequence("t1", "alpha.ts")], { order: 3 });

    // A novel observation prefix that never preceded "open" in training, but the
    // single-event context (open) → edit still matches at order 1.
    const prediction = policy.predict([obs("terminal", "unseen context"), act("open", "alpha.ts")]);
    expect(prediction?.action).toEqual(act("edit", "alpha.ts"));
    expect(prediction?.matchedOrder).toBe(1);
  });

  it("falls back to the prior action when no context matches", () => {
    const backend = new NgramMovementBackend();
    const policy = backend.train(
      [
        { id: "t1", events: [act("save", "x"), act("save", "x"), act("open", "y")] },
      ],
      { order: 2 },
    );

    const prediction = policy.predict([obs("nothing", "matches this at all")]);
    expect(prediction?.source).toBe("prior");
    // "save x" is the most frequent concrete action overall (2 vs 1).
    expect(prediction?.action).toEqual(act("save", "x"));
  });

  it("returns undefined when nothing was learned", () => {
    const backend = new NgramMovementBackend();
    const policy = backend.train([{ id: "empty", events: [obs("x", "y")] }]);
    expect(policy.predict([obs("x", "y")])).toBeUndefined();
  });

  it("is deterministic across independent trainings (tie-break stable)", () => {
    const backend = new NgramMovementBackend();
    const data: MovementSequence[] = [
      { id: "t1", events: [obs("s", "ctx"), act("zeta", "1")] },
      { id: "t2", events: [obs("s", "ctx"), act("alpha", "1")] },
    ];
    const a = backend.train(data).predict([obs("s", "ctx")]);
    const b = backend.train(data).predict([obs("s", "ctx")]);
    expect(a).toEqual(b);
    // Equal counts (1 each) tie-break lexicographically on action key → "alpha".
    expect(a?.action.tool).toBe("alpha");
  });

  it("reports model stats", () => {
    const backend = new NgramMovementBackend();
    const policy = backend.train([fileEditSequence("t1", "alpha.ts")], { order: 2 });
    const stats = policy.stats;
    expect(stats.backendId).toBe("ngram-mock");
    expect(stats.order).toBe(2);
    expect(stats.sequenceCount).toBe(1);
    expect(stats.actionCount).toBe(3);
    expect(stats.specificContextCount).toBeGreaterThan(0);
  });
});

describe("rolloutMovement", () => {
  it("reproduces a full recorded movement from a seed observation", () => {
    const backend = new NgramMovementBackend();
    const policy = backend.train([fileEditSequence("t1", "alpha.ts")], { order: 3 });

    const steps = rolloutMovement(policy, [obs("editor", "viewing alpha.ts")], { maxSteps: 5 });
    expect(steps.map((step) => step.action)).toEqual([
      act("open", "alpha.ts"),
      act("edit", "alpha.ts"),
      act("save", "alpha.ts"),
    ]);
  });

  it("honors maxSteps and minConfidence", () => {
    const backend = new NgramMovementBackend();
    const policy = backend.train([fileEditSequence("t1", "alpha.ts")], { order: 3 });

    const capped = rolloutMovement(policy, [obs("editor", "viewing alpha.ts")], { maxSteps: 1 });
    expect(capped).toHaveLength(1);
    expect(capped[0].action).toEqual(act("open", "alpha.ts"));

    // A confidence floor above 1 stops the rollout immediately.
    const stopped = rolloutMovement(policy, [obs("editor", "viewing alpha.ts")], { minConfidence: 1.1 });
    expect(stopped).toHaveLength(0);
  });
});

describe("sequence adapters", () => {
  it("derives a time-ordered movement sequence from a trajectory span", () => {
    const trajectory = buildTrajectorySpan({
      id: "traj-1",
      sessionId: "sess-1",
      observations: [{ kind: "observation", source: "editor", summary: "viewing", ts: 10 }],
      actions: [
        { kind: "action", tool: "save", summary: "file", ts: 30 },
        { kind: "action", tool: "open", summary: "file", ts: 20 },
      ],
    });
    const sequence = sequenceFromTrajectory(trajectory);
    expect(sequence.id).toBe("traj-1");
    expect(sequence.events).toEqual([
      obs("editor", "viewing"),
      act("open", "file"),
      act("save", "file"),
    ]);
  });

  it("derives a movement sequence from a replay manifest, dropping transcript events", () => {
    const manifest: ReplayManifest = {
      version: 1,
      sessionId: "sess-1",
      trajectoryIds: ["traj-1"],
      eventCount: 3,
      events: [
        { kind: "transcript", ts: 5, messageId: "m1", role: "user", content: "hi" },
        { kind: "observation", ts: 10, trajectoryId: "traj-1", source: "editor", summary: "viewing" },
        { kind: "action", ts: 20, trajectoryId: "traj-1", tool: "open", summary: "file" },
      ],
    };
    const sequence = sequenceFromReplayManifest(manifest);
    expect(sequence.events).toEqual([obs("editor", "viewing"), act("open", "file")]);
  });

  it("trains end-to-end on trajectory-derived sequences", () => {
    const backend = new NgramMovementBackend();
    const trajectory = buildTrajectorySpan({
      id: "traj-1",
      sessionId: "sess-1",
      observations: [{ kind: "observation", source: "editor", summary: "viewing", ts: 10 }],
      actions: [
        { kind: "action", tool: "open", summary: "file", ts: 20 },
        { kind: "action", tool: "edit", summary: "file", ts: 30 },
      ],
    });
    const policy = backend.train([sequenceFromTrajectory(trajectory)]);
    const prediction = policy.predict([obs("editor", "viewing"), act("open", "file")]);
    expect(prediction?.action).toEqual(act("edit", "file"));
  });
});
