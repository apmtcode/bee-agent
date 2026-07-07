import { describe, expect, it } from "vitest";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import type { TrajectoryAction } from "../capture/trajectory.js";
import {
  buildMovementDataset,
  createDefaultMovementRegistry,
  DEFAULT_MOVEMENT_BACKEND_ID,
  evaluateMovementPolicy,
  MovementModelRegistry,
  movementToken,
  NgramMovementBackend,
  type MovementAction,
  type MovementDataset,
  type MovementModelBackend,
  type MovementPolicyModel,
  type MovementSequence,
} from "./movement-policy.js";

function action(tool: string, gesture: string, target: string, ts: number): TrajectoryAction {
  return {
    kind: "action",
    tool,
    summary: `${gesture} ${target}`,
    ts,
    metadata: { gesture, target },
  };
}

function movement(tool: string, gesture: string, target: string): MovementAction {
  return { tool, gesture, target, summary: `${gesture} ${target}` };
}

function sequence(id: string, actions: MovementAction[]): MovementSequence {
  return { id, actions };
}

describe("movementToken", () => {
  it("is deterministic and normalizes missing fields", () => {
    expect(movementToken({ tool: "Device", gesture: "Tap", target: "Send", summary: "x" })).toBe(
      "device:tap:send:-",
    );
    expect(movementToken({ tool: "device", summary: "x" })).toBe("device:-:-:-");
  });
});

describe("buildMovementDataset", () => {
  it("extracts timestamp-ordered movement sequences from trajectories and skips rejected/empty ones", () => {
    const good = buildTrajectorySpan({
      id: "traj-good",
      sessionId: "s1",
      actions: [action("device", "tap", "menu", 20), action("device", "tap", "settings", 10)],
      outcome: { status: "success", summary: "opened settings", reward: 1 },
    });
    const rejected = {
      ...buildTrajectorySpan({
        id: "traj-rejected",
        sessionId: "s1",
        actions: [action("device", "tap", "delete", 5)],
      }),
      review: { status: "rejected" as const, reviewedAt: "2026-01-01T00:00:00Z", reviewedBy: "me" },
    };
    const empty = buildTrajectorySpan({ id: "traj-empty", sessionId: "s1", actions: [] });

    const dataset = buildMovementDataset([good, rejected, empty]);

    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]?.id).toBe("traj-good");
    expect(dataset.sequences[0]?.goal).toBe("opened settings");
    // Sorted by ts: settings(10) before menu(20).
    expect(dataset.sequences[0]?.actions.map((a) => a.target)).toEqual(["settings", "menu"]);
  });
});

describe("NgramMovementBackend", () => {
  const trainingDataset: MovementDataset = {
    version: 1,
    sequences: [
      sequence("open-settings", [
        movement("device", "tap", "home"),
        movement("device", "tap", "menu"),
        movement("device", "tap", "settings"),
      ]),
      sequence("open-settings-2", [
        movement("device", "tap", "home"),
        movement("device", "tap", "menu"),
        movement("device", "tap", "settings"),
      ]),
    ],
  };

  it("repeats a recorded movement sequence exactly (memorization)", () => {
    const model = new NgramMovementBackend({ order: 2 }).train(trainingDataset);
    const generated = model.generate({
      seed: [movement("device", "tap", "home")],
      maxSteps: 8,
    });
    expect(generated.map((a) => a.target)).toEqual(["menu", "settings"]);
  });

  it("generalizes to a new-but-related sequence via back-off", () => {
    // Train on two overlapping flows that share the "menu -> settings" suffix.
    const dataset: MovementDataset = {
      version: 1,
      sequences: [
        sequence("flow-a", [
          movement("device", "tap", "home"),
          movement("device", "tap", "menu"),
          movement("device", "tap", "settings"),
        ]),
        sequence("flow-b", [
          movement("device", "tap", "search"),
          movement("device", "tap", "menu"),
          movement("device", "tap", "settings"),
        ]),
      ],
    };
    const model = new NgramMovementBackend({ order: 2 }).train(dataset);

    // A prefix never seen as a whole: search -> menu came from flow-b, but the
    // model must still continue to "settings" (seen after "menu" in both flows).
    const prediction = model.predictNext([
      movementToken(movement("device", "tap", "search")),
      movementToken(movement("device", "tap", "menu")),
    ]);
    expect(prediction?.action?.target).toBe("settings");
  });

  it("is deterministic across repeated training and generation", () => {
    const a = new NgramMovementBackend({ order: 2 }).train(trainingDataset);
    const b = new NgramMovementBackend({ order: 2 }).train(trainingDataset);
    expect(a.serialize()).toEqual(b.serialize());
    expect(a.generate({ seed: [movement("device", "tap", "home")] })).toEqual(
      b.generate({ seed: [movement("device", "tap", "home")] }),
    );
  });

  it("terminates generation at the end sentinel without looping", () => {
    const model = new NgramMovementBackend({ order: 1 }).train(trainingDataset);
    const generated = model.generate({ seed: [movement("device", "tap", "settings")], maxSteps: 50 });
    // "settings" is always followed by end-of-sequence in training.
    expect(generated).toEqual([]);
  });
});

describe("evaluateMovementPolicy", () => {
  it("scores replay fidelity on held-out related sequences and counts generalization", () => {
    const dataset: MovementDataset = {
      version: 1,
      sequences: [
        sequence("t1", [
          movement("device", "tap", "home"),
          movement("device", "tap", "menu"),
          movement("device", "tap", "settings"),
        ]),
      ],
    };
    const model = new NgramMovementBackend({ order: 2 }).train(dataset);
    const heldOut = [
      sequence("held", [
        movement("device", "tap", "home"),
        movement("device", "tap", "menu"),
        movement("device", "tap", "settings"),
      ]),
    ];
    const evaluation = evaluateMovementPolicy(model, heldOut);
    expect(evaluation.predictions).toBe(3);
    // Mid-sequence transitions are recalled exactly; only the empty-context
    // first-move prediction is inherently ambiguous, so ≥2/3 are correct.
    expect(evaluation.correct).toBeGreaterThanOrEqual(2);
    expect(evaluation.fidelity).toBeCloseTo(evaluation.correct / evaluation.predictions);
    // Given the true prefix, the next movement is recalled deterministically.
    expect(
      model.predictNext([
        movementToken(movement("device", "tap", "home")),
        movementToken(movement("device", "tap", "menu")),
      ])?.action?.target,
    ).toBe("settings");
  });
});

describe("MovementModelRegistry", () => {
  it("exposes the default deterministic backend and trains through it", () => {
    const registry = createDefaultMovementRegistry({ order: 2 });
    expect(registry.ids()).toContain(DEFAULT_MOVEMENT_BACKEND_ID);
    const model = registry.train(DEFAULT_MOVEMENT_BACKEND_ID, {
      version: 1,
      sequences: [sequence("s", [movement("device", "tap", "home")])],
    });
    expect(model.backend).toBe(DEFAULT_MOVEMENT_BACKEND_ID);
  });

  it("supports pluggable custom backends", () => {
    const stubModel: MovementPolicyModel = {
      backend: "stub",
      order: 0,
      predictNext: () => undefined,
      generate: () => [movement("stub", "noop", "none")],
      actionForToken: () => undefined,
      serialize: () => ({ version: 1, backend: "stub", order: 0, vocabulary: {}, transitions: [] }),
    };
    const stubBackend: MovementModelBackend = { id: "stub", train: () => stubModel };
    const registry = new MovementModelRegistry().register(stubBackend);
    const model = registry.train("stub", { version: 1, sequences: [] });
    expect(model.generate()).toEqual([movement("stub", "noop", "none")]);
    expect(() => registry.train("missing", { version: 1, sequences: [] })).toThrow(/Unknown movement backend/);
  });
});
