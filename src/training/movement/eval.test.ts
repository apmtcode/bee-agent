import { describe, expect, it } from "vitest";
import { euclidean, evaluateGeneralization, resamplePath } from "./eval.js";
import { MockMovementModelBackend } from "./backend.js";
import { buildMovementDataset } from "./dataset.js";
import { generateGestureBatch } from "./synthetic-stream.js";

describe("eval", () => {
  it("computes euclidean distance", () => {
    expect(euclidean({ x: 0, y: 0 }, { x: 3, y: 4 })).toBe(5);
  });

  it("resamples a path to a fixed point count", () => {
    const resampled = resamplePath([
      { x: 0, y: 0 },
      { x: 10, y: 0 },
    ], 3);
    expect(resampled).toHaveLength(3);
    expect(resampled[1]).toEqual({ x: 5, y: 0 });
  });

  it("resamples degenerate paths safely", () => {
    expect(resamplePath([], 4)).toEqual([]);
    expect(resamplePath([{ x: 2, y: 2 }], 3)).toEqual([
      { x: 2, y: 2 },
      { x: 2, y: 2 },
      { x: 2, y: 2 },
    ]);
  });

  it("returns an empty result for an empty held-out set", async () => {
    const backend = new MockMovementModelBackend();
    const artifact = await backend.train(
      buildMovementDataset(generateGestureBatch({ targets: [{ x: 10, y: 10 }], seed: 1 })),
    );
    const result = await evaluateGeneralization(backend, artifact, []);
    expect(result).toMatchObject({ count: 0, hitRate: 0, meanFinalError: 0 });
  });

  it("reports per-trajectory fidelity for held-out gestures", async () => {
    const backend = new MockMovementModelBackend();
    const trainSet = generateGestureBatch({
      targets: [
        { x: 100, y: 30 },
        { x: 220, y: 90 },
      ],
      seed: 4,
    });
    const artifact = await backend.train(buildMovementDataset(trainSet));
    const heldOut = generateGestureBatch({ targets: [{ x: 160, y: 60 }], seed: 77, idPrefix: "held" });
    const result = await evaluateGeneralization(backend, artifact, heldOut, { tolerance: 1 });
    expect(result.perTrajectory).toHaveLength(1);
    expect(result.perTrajectory[0]!.hit).toBe(true);
    expect(result.perTrajectory[0]!.finalError).toBeLessThan(1);
  });
});
