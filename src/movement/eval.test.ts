import { describe, expect, it } from "vitest";
import type { Point } from "./events.js";
import { SimilarityTransformBackend, type MovementContext } from "./model.js";
import { generateDragDemonstration, generateSyntheticDataset } from "./synthetic.js";
import {
  evaluateGeneralization,
  pathRmse,
  resamplePath,
  type MovementEvalCase,
} from "./eval.js";

describe("resamplePath", () => {
  it("resamples a straight line to evenly spaced points", () => {
    const out = resamplePath([{ x: 0, y: 0 }, { x: 10, y: 0 }], 3);
    expect(out).toEqual([
      { x: 0, y: 0 },
      { x: 5, y: 0 },
      { x: 10, y: 0 },
    ]);
  });

  it("handles a single-point path", () => {
    expect(resamplePath([{ x: 2, y: 3 }], 2)).toEqual([
      { x: 2, y: 3 },
      { x: 2, y: 3 },
    ]);
  });
});

describe("pathRmse", () => {
  it("is zero for identical paths and positive for divergent ones", () => {
    const a = [{ x: 0, y: 0 }, { x: 10, y: 10 }];
    expect(pathRmse(a, a)).toBeCloseTo(0);
    expect(pathRmse(a, [{ x: 0, y: 0 }, { x: 10, y: 0 }])).toBeGreaterThan(0);
  });
});

describe("generalization eval", () => {
  function groundTruthCase(start: Point, end: Point): MovementEvalCase {
    const demo = generateDragDemonstration({ id: "gt", intent: "drag", start, end, steps: 8 });
    const context: MovementContext = { intent: "drag", start, end };
    return { context, expected: demo.events };
  }

  it("a model trained on synthetic drags generalizes to held-out targets with low error", async () => {
    const model = await new SimilarityTransformBackend().train(
      generateSyntheticDataset({ intent: "drag", count: 8, seed: 21 }),
    );
    const heldOut: MovementEvalCase[] = [
      groundTruthCase({ x: 100, y: 100 }, { x: 700, y: 250 }),
      groundTruthCase({ x: 50, y: 600 }, { x: 900, y: 80 }),
      groundTruthCase({ x: 300, y: 300 }, { x: 320, y: 760 }),
    ];

    const summary = evaluateGeneralization(model, heldOut);
    expect(summary.evaluated).toBe(3);
    // Line drags are exactly recoverable by a similarity transform.
    expect(summary.maxEndpointError).toBeLessThan(1e-6);
    expect(summary.maxPathRmse).toBeLessThan(1e-6);
  });

  it("reports empty results for unknown intents without throwing", async () => {
    const model = await new SimilarityTransformBackend().train(
      generateSyntheticDataset({ intent: "drag", count: 2, seed: 3 }),
    );
    const summary = evaluateGeneralization(model, [
      { context: { intent: "teleport", start: { x: 0, y: 0 }, end: { x: 1, y: 1 } }, expected: [] },
    ]);
    expect(summary.evaluated).toBe(0);
    expect(summary.results[0]!.empty).toBe(true);
  });
});
