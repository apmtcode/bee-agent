import { describe, expect, it } from "vitest";
import { euclideanDistance, pointerPath, type Point } from "./events.js";
import {
  SimilarityTransformBackend,
  retargetEvents,
  similarityTransform,
} from "./model.js";
import { generateDragDemonstration, generateSyntheticDataset } from "./synthetic.js";
import { buildMovementDataset } from "./events.js";

describe("similarityTransform", () => {
  it("maps the correspondence endpoints exactly", () => {
    const f = similarityTransform({ x: 0, y: 0 }, { x: 10, y: 0 }, { x: 5, y: 5 }, { x: 5, y: 25 });
    expect(f({ x: 0, y: 0 })).toEqual({ x: 5, y: 5 });
    const mappedEnd = f({ x: 10, y: 0 });
    expect(mappedEnd.x).toBeCloseTo(5);
    expect(mappedEnd.y).toBeCloseTo(25);
  });

  it("is the identity when source and target frames match", () => {
    const f = similarityTransform({ x: 1, y: 2 }, { x: 3, y: 8 }, { x: 1, y: 2 }, { x: 3, y: 8 });
    const p = { x: 7, y: -4 };
    const out = f(p);
    expect(out.x).toBeCloseTo(p.x);
    expect(out.y).toBeCloseTo(p.y);
  });

  it("falls back to translation for a degenerate source vector", () => {
    const f = similarityTransform({ x: 2, y: 2 }, { x: 2, y: 2 }, { x: 10, y: 10 }, { x: 99, y: 99 });
    expect(f({ x: 5, y: 5 })).toEqual({ x: 13, y: 13 });
  });

  it("preserves shape under rotation/scale (retargets an arc bow)", () => {
    const demo = generateDragDemonstration({
      id: "arc",
      intent: "drag",
      start: { x: 0, y: 0 },
      end: { x: 100, y: 0 },
      shape: "arc",
      arcHeight: 20,
      steps: 8,
    });
    // Rotate the frame 90° and double its length.
    const f = similarityTransform({ x: 0, y: 0 }, { x: 100, y: 0 }, { x: 0, y: 0 }, { x: 0, y: 200 });
    const retargeted = retargetEvents(demo.events, f);
    const path = pointerPath(retargeted);
    // Midpoint should sit near (±40, 100): bow of 20 scaled ×2 off the rotated axis.
    const mid = path[Math.floor(path.length / 2)]!;
    expect(Math.abs(mid.y)).toBeGreaterThan(50);
    expect(Math.abs(mid.x)).toBeGreaterThan(20);
  });
});

describe("SimilarityTransformBackend", () => {
  it("reproduces a recorded movement exactly when asked for its own frame", async () => {
    const demo = generateDragDemonstration({
      id: "d",
      intent: "drag",
      start: { x: 10, y: 20 },
      end: { x: 200, y: 300 },
      steps: 6,
    });
    const model = await new SimilarityTransformBackend().train(buildMovementDataset([demo]));
    const predicted = model.predict({ intent: "drag", start: { x: 10, y: 20 }, end: { x: 200, y: 300 } });
    const predictedPath = pointerPath(predicted);
    const expectedPath = pointerPath(demo.events);
    expect(predictedPath.length).toBe(expectedPath.length);
    for (let i = 0; i < expectedPath.length; i += 1) {
      expect(euclideanDistance(predictedPath[i]!, expectedPath[i]!)).toBeLessThan(1e-6);
    }
  });

  it("generalizes to a new, unseen target by landing on the requested endpoint", async () => {
    const dataset = generateSyntheticDataset({ intent: "drag", count: 5, seed: 7 });
    const model = await new SimilarityTransformBackend().train(dataset);
    const target = { intent: "drag", start: { x: 400, y: 400 } as Point, end: { x: 900, y: 120 } as Point };
    const predicted = model.predict(target);
    const path = pointerPath(predicted);
    expect(path.length).toBeGreaterThan(0);
    // Lands on the requested endpoint, and starts at the requested start.
    expect(euclideanDistance(path[0]!, target.start)).toBeLessThan(1e-6);
    expect(euclideanDistance(path[path.length - 1]!, target.end)).toBeLessThan(1e-6);
  });

  it("returns no prediction for an unknown intent", async () => {
    const model = await new SimilarityTransformBackend().train(generateSyntheticDataset({ intent: "drag" }));
    expect(model.predict({ intent: "fly", start: { x: 0, y: 0 }, end: { x: 1, y: 1 } })).toEqual([]);
    expect(model.intents).toEqual(["drag"]);
  });
});
