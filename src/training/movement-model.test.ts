import { describe, expect, it } from "vitest";
import { buildTrajectorySpan, type TrajectoryAction } from "../capture/trajectory.js";
import {
  MarkovMovementBackend,
  buildMovementDataset,
  deriveMovementSequence,
  generateSyntheticMovementDataset,
  movementStepToken,
  type MovementTrainingDataset,
} from "./movement-model.js";

function action(tool: string, summary: string, ts: number, metadata?: Record<string, unknown>): TrajectoryAction {
  return { kind: "action", tool, summary, ts, ...(metadata ? { metadata } : {}) };
}

describe("deriveMovementSequence", () => {
  it("orders actions by ts and computes relative timing", () => {
    const trajectory = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [
        action("mouse", "tap", 300, { gesture: "tap", target: "submit", x: 40, y: 80 }),
        action("mouse", "move", 100, { gesture: "move", x: 10, y: 20 }),
        action("mouse", "move", 200, { gesture: "move", x: 30, y: 60 }),
      ],
    });

    const sequence = deriveMovementSequence(trajectory);

    expect(sequence.steps.map((step) => step.coordinate)).toEqual([
      { x: 10, y: 20 },
      { x: 30, y: 60 },
      { x: 40, y: 80 },
    ]);
    expect(sequence.steps[0].dtMs).toBeUndefined();
    expect(sequence.steps[1].dtMs).toBe(100);
    expect(sequence.steps[2].dtMs).toBe(100);
    expect(sequence.steps[2].target).toBe("submit");
  });
});

describe("MarkovMovementBackend", () => {
  const backend = new MarkovMovementBackend();

  function toTrajectory(id: string, steps: Array<{ gesture: string; target?: string; x: number; y: number }>) {
    return buildTrajectorySpan({
      id,
      sessionId: "s",
      actions: steps.map((step, index) =>
        action("mouse", step.gesture, index * 100, {
          gesture: step.gesture,
          x: step.x,
          y: step.y,
          ...(step.target ? { target: step.target } : {}),
        }),
      ),
    });
  }

  const dataset: MovementTrainingDataset = buildMovementDataset([
    toTrajectory("a", [
      { gesture: "move", x: 0, y: 0 },
      { gesture: "move", x: 50, y: 50 },
      { gesture: "tap", target: "submit", x: 100, y: 100 },
    ]),
    toTrajectory("b", [
      { gesture: "move", x: 2, y: 2 },
      { gesture: "move", x: 48, y: 52 },
      { gesture: "tap", target: "submit", x: 98, y: 102 },
    ]),
  ]);

  it("training is deterministic — same dataset yields identical models", () => {
    const first = backend.train(dataset);
    const second = backend.train(dataset);
    expect(JSON.stringify(first)).toEqual(JSON.stringify(second));
    expect(first.sequenceCount).toBe(2);
    expect(first.stepCount).toBe(6);
    expect(first.tokens).toContain("mouse/move");
    expect(first.tokens).toContain("mouse/tap");
  });

  it("reproduces the learned movement structure (move → move → tap)", () => {
    const model = backend.train(dataset);
    const prediction = backend.predict(model);
    expect(prediction.tokens).toEqual(["mouse/move", "mouse/move", "mouse/tap"]);
    expect(prediction.terminatedBy).toBe("terminal");
    expect(prediction.steps.at(-1)?.gesture).toBe("tap");
    expect(prediction.steps.at(-1)?.target).toBe("submit");
  });

  it("prediction is deterministic", () => {
    const model = backend.train(dataset);
    expect(JSON.stringify(backend.predict(model))).toEqual(JSON.stringify(backend.predict(model)));
  });

  it("generalizes: retargets to an unseen UI target", () => {
    const model = backend.train(dataset);
    const prediction = backend.predict(model, { targetOverride: "cancel-button" });
    expect(prediction.steps.every((step) => step.target === "cancel-button")).toBe(true);
  });

  it("generalizes: interpolates coordinates toward a new goal it never trained on", () => {
    const model = backend.train(dataset);
    const prediction = backend.predict(model, { goalCoordinate: { x: 400, y: 300 } });

    const start = prediction.steps[0].coordinate!;
    const end = prediction.steps.at(-1)!.coordinate!;
    // First step anchored at the learned start; last step reaches the new goal.
    expect(start.x).toBeCloseTo(1, 5);
    expect(start.y).toBeCloseTo(1, 5);
    expect(end).toEqual({ x: 400, y: 300 });
    // Monotonic progression toward the goal (generalized straight-line movement).
    const xs = prediction.steps.map((step) => step.coordinate!.x);
    expect(xs).toEqual([...xs].sort((a, b) => a - b));
  });

  it("respects an explicit maxSteps bound", () => {
    // A self-looping dataset would otherwise run forever without the bound.
    const looping = buildMovementDataset([
      toTrajectory("loop", [
        { gesture: "scroll", x: 0, y: 0 },
        { gesture: "scroll", x: 0, y: 10 },
        { gesture: "scroll", x: 0, y: 20 },
      ]),
    ]);
    const model = backend.train(looping);
    const prediction = backend.predict(model, { maxSteps: 2 });
    expect(prediction.steps.length).toBeLessThanOrEqual(2);
  });
});

describe("generateSyntheticMovementDataset", () => {
  it("is reproducible for a given seed and validates the full round-trip", () => {
    const first = generateSyntheticMovementDataset({ sequenceCount: 8, seed: 42 });
    const second = generateSyntheticMovementDataset({ sequenceCount: 8, seed: 42 });
    expect(JSON.stringify(first)).toEqual(JSON.stringify(second));
    expect(first.sequences).toHaveLength(8);

    const backend = new MarkovMovementBackend();
    const model = backend.train(first);
    const prediction = backend.predict(model);
    // The synthetic pattern is always move → move → tap.
    expect(prediction.tokens[0]).toBe("mouse/move");
    expect(prediction.tokens.at(-1)).toBe("mouse/tap");
    expect(model.stepCount).toBe(24);
  });

  it("different seeds produce different datasets", () => {
    const a = generateSyntheticMovementDataset({ sequenceCount: 4, seed: 1 });
    const b = generateSyntheticMovementDataset({ sequenceCount: 4, seed: 2 });
    expect(JSON.stringify(a)).not.toEqual(JSON.stringify(b));
  });
});

describe("movementStepToken", () => {
  it("includes direction when present", () => {
    expect(movementStepToken({ tool: "mouse", gesture: "scroll", direction: "down" })).toBe("mouse/scroll:down");
    expect(movementStepToken({ tool: "mouse", gesture: "tap" })).toBe("mouse/tap");
  });
});
