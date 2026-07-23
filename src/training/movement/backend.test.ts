import { describe, expect, it } from "vitest";
import { MockMovementModelBackend } from "./backend.js";
import { buildMovementDataset } from "./dataset.js";
import { pointerPath } from "./event-schema.js";
import { evaluateGeneralization } from "./eval.js";
import { generateClickGesture, generateGestureBatch } from "./synthetic-stream.js";

describe("MockMovementModelBackend", () => {
  it("trains an artifact capturing the learned profile", async () => {
    const dataset = buildMovementDataset(
      generateGestureBatch({
        targets: [
          { x: 100, y: 20 },
          { x: 40, y: 200 },
          { x: 300, y: 150 },
        ],
        seed: 5,
      }),
    );
    const backend = new MockMovementModelBackend();
    const artifact = await backend.train(dataset);
    expect(artifact.backend).toBe("mock-profile-v1");
    expect(artifact.exampleCount).toBe(3);
    expect(artifact.knots.length).toBeGreaterThan(1);
    expect(artifact.knots[0]).toMatchObject({ t: 0 });
    expect(artifact.knots.at(-1)).toMatchObject({ t: 1 });
  });

  it("reproduces a recorded gesture (replay fidelity)", async () => {
    const recorded = generateClickGesture({ id: "r", start: { x: 0, y: 0 }, target: { x: 120, y: 80 }, steps: 12 });
    const dataset = buildMovementDataset([recorded]);
    const backend = new MockMovementModelBackend();
    const artifact = await backend.train(dataset);

    const inferred = await backend.infer(artifact, { start: { x: 0, y: 0 }, target: { x: 120, y: 80 } });
    const recordedMoves = recorded.events.filter((event) => event.kind === "pointer-move");
    const inferredMoves = inferred.events.filter((event) => event.kind === "pointer-move");
    // Reproduces the recorded resolution and lands exactly on the target.
    expect(inferredMoves).toHaveLength(recordedMoves.length);
    expect(pointerPath(inferred).at(-1)).toEqual({ x: 120, y: 80 });
    // Follows the same ease-in-out profile point-for-point.
    for (let i = 0; i < recordedMoves.length; i += 1) {
      const r = recordedMoves[i] as { x: number; y: number };
      const inf = inferredMoves[i] as { x: number; y: number };
      expect(Math.hypot(r.x - inf.x, r.y - inf.y)).toBeLessThan(1);
    }
  });

  it("generalizes to unseen targets with low landing + path error", async () => {
    // Train on a set of targets, hold out a different, related set.
    const trainTargets = [
      { x: 100, y: 20 },
      { x: 200, y: 60 },
      { x: 300, y: 120 },
      { x: 150, y: 200 },
    ];
    const heldOutTargets = [
      { x: 250, y: 90 },
      { x: 120, y: 150 },
    ];
    const start = { x: 0, y: 0 };
    const trainSet = generateGestureBatch({ targets: trainTargets, start, seed: 11, idPrefix: "train" });
    const heldOut = generateGestureBatch({ targets: heldOutTargets, start, seed: 99, idPrefix: "held" });

    const backend = new MockMovementModelBackend();
    const artifact = await backend.train(buildMovementDataset(trainSet));

    const result = await evaluateGeneralization(backend, artifact, heldOut, { tolerance: 1 });
    expect(result.count).toBe(2);
    // Lands on every unseen target and stays close to the reference path.
    expect(result.hitRate).toBe(1);
    expect(result.meanFinalError).toBeLessThan(1);
    expect(result.meanPathError).toBeLessThan(6);
  });

  it("specializes gestures for labels it has seen", async () => {
    const labeled = generateClickGesture({
      id: "labeled",
      label: "click:save-button",
      start: { x: 0, y: 0 },
      target: { x: 80, y: 40 },
      steps: 10,
    });
    const backend = new MockMovementModelBackend();
    const artifact = await backend.train(buildMovementDataset([labeled]));
    expect(Object.keys(artifact.labels)).toContain("click:save-button");

    const inferred = await backend.infer(artifact, {
      label: "click:save-button",
      start: { x: 0, y: 0 },
      target: { x: 80, y: 40 },
    });
    expect(pointerPath(inferred).at(-1)).toEqual({ x: 80, y: 40 });
  });
});
