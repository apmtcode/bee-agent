import { describe, expect, it } from "vitest";
import {
  InProcessMovementModelBackend,
  type MovementDataset,
  type MovementSequence,
} from "./movement-model.js";
import {
  buildMovementDataset,
  evaluateMovementModel,
  generatePointerGesture,
  type MovementEvalCase,
} from "./movement-synthesis.js";

const START = { x: 0.1, y: 0.1 };

function trainClickModel(targets: { x: number; y: number }[]) {
  const sequences = generatePointerGesture({
    label: "point-and-click",
    start: START,
    targets,
    steps: 4,
    stepMs: 16,
    click: true,
  });
  const dataset = buildMovementDataset(sequences);
  const model = new InProcessMovementModelBackend().train(dataset);
  return { model, sequences };
}

describe("InProcessMovementModelBackend", () => {
  it("reproduces a recorded movement it was trained on (repeat)", () => {
    const { model, sequences } = trainClickModel([
      { x: 0.8, y: 0.2 },
      { x: 0.2, y: 0.9 },
      { x: 0.5, y: 0.5 },
    ]);

    const original = sequences[0]!;
    const predicted = model.predict({ label: "point-and-click", params: original.params });

    expect(predicted.events).toHaveLength(original.events.length);
    for (let index = 0; index < original.events.length; index += 1) {
      const a = predicted.events[index]!;
      const b = original.events[index]!;
      expect(a.type).toBe(b.type);
      if (typeof b.x === "number") {
        expect(a.x).toBeCloseTo(b.x, 4);
        expect(a.y).toBeCloseTo(b.y!, 4);
      }
      if (b.button) {
        expect(a.button).toBe(b.button);
      }
    }
  });

  it("generalizes to a held-out target (re-aims the gesture)", () => {
    // Train on a spread of targets, hold out a fresh interior target.
    const { model } = trainClickModel([
      { x: 0.8, y: 0.2 },
      { x: 0.2, y: 0.8 },
      { x: 0.9, y: 0.9 },
      { x: 0.15, y: 0.15 },
    ]);

    const heldOut = { x: 0.63, y: 0.41 };
    const predicted = model.predict({
      label: "point-and-click",
      params: { targetX: heldOut.x, targetY: heldOut.y },
    });

    const last = predicted.events[predicted.events.length - 1]!;
    expect(last.type).toBe("mouse-up");
    expect(last.x).toBeCloseTo(heldOut.x, 2);
    expect(last.y).toBeCloseTo(heldOut.y, 2);
  });

  it("extrapolates beyond the training envelope along the learned linear relation", () => {
    // Non-collinear training targets so the regression can separate targetX
    // from targetY, then query a point outside the training convex hull.
    const { model } = trainClickModel([
      { x: 0.3, y: 0.4 },
      { x: 0.4, y: 0.3 },
      { x: 0.5, y: 0.5 },
      { x: 0.35, y: 0.55 },
    ]);

    const predicted = model.predict({
      label: "point-and-click",
      params: { targetX: 0.9, targetY: 0.1 },
    });
    const last = predicted.events[predicted.events.length - 1]!;
    expect(last.x).toBeCloseTo(0.9, 2);
    expect(last.y).toBeCloseTo(0.1, 2);
  });

  it("clamps synthesized pointer coordinates to the unit square", () => {
    const { model } = trainClickModel([
      { x: 0.3, y: 0.3 },
      { x: 0.5, y: 0.5 },
    ]);
    const predicted = model.predict({
      label: "point-and-click",
      params: { targetX: 5, targetY: -5 },
    });
    for (const event of predicted.events) {
      if (typeof event.x === "number") {
        expect(event.x).toBeGreaterThanOrEqual(0);
        expect(event.x).toBeLessThanOrEqual(1);
        expect(event.y!).toBeGreaterThanOrEqual(0);
        expect(event.y!).toBeLessThanOrEqual(1);
      }
    }
  });

  it("survives a serialize → restore round-trip", () => {
    const { model } = trainClickModel([
      { x: 0.8, y: 0.2 },
      { x: 0.2, y: 0.8 },
      { x: 0.5, y: 0.5 },
    ]);
    const backend = new InProcessMovementModelBackend();
    const restored = backend.restore(model.serialize());

    const params = { targetX: 0.6, targetY: 0.35 };
    const before = model.predict({ label: "point-and-click", params });
    const after = restored.predict({ label: "point-and-click", params });
    expect(after.events).toEqual(before.events);
    expect(restored.labels()).toEqual(model.labels());
  });

  it("throws for an unknown label", () => {
    const { model } = trainClickModel([{ x: 0.5, y: 0.5 }]);
    expect(() => model.predict({ label: "nope" })).toThrow(/no template/);
  });

  it("learns multiple distinct labels in one dataset", () => {
    const click = generatePointerGesture({
      label: "click",
      start: START,
      targets: [{ x: 0.8, y: 0.2 }, { x: 0.2, y: 0.8 }],
      click: true,
    });
    const hover = generatePointerGesture({
      label: "hover",
      start: START,
      targets: [{ x: 0.6, y: 0.6 }, { x: 0.3, y: 0.9 }],
      click: false,
    });
    const dataset: MovementDataset = buildMovementDataset([...click, ...hover]);
    const model = new InProcessMovementModelBackend().train(dataset);

    expect(model.labels()).toEqual(["click", "hover"]);
    expect(model.predict({ label: "click", params: { targetX: 0.5, targetY: 0.5 } }).events.at(-1)!.type).toBe(
      "mouse-up",
    );
    expect(model.predict({ label: "hover", params: { targetX: 0.5, targetY: 0.5 } }).events.at(-1)!.type).toBe(
      "mouse-move",
    );
  });
});

describe("evaluateMovementModel", () => {
  it("reports low endpoint error and a high pass rate on held-out targets", () => {
    const { model } = trainClickModel([
      { x: 0.8, y: 0.2 },
      { x: 0.2, y: 0.8 },
      { x: 0.9, y: 0.9 },
      { x: 0.15, y: 0.15 },
      { x: 0.5, y: 0.5 },
    ]);

    const heldOutTargets = [
      { x: 0.63, y: 0.41 },
      { x: 0.35, y: 0.72 },
      { x: 0.77, y: 0.55 },
    ];
    const expected = generatePointerGesture({
      label: "point-and-click",
      start: START,
      targets: heldOutTargets,
      steps: 4,
      stepMs: 16,
      click: true,
    });
    const cases: MovementEvalCase[] = expected.map((sequence) => ({
      label: "point-and-click",
      params: sequence.params!,
      expected: sequence,
    }));

    const result = evaluateMovementModel(model, cases);
    expect(result.cases).toBe(3);
    expect(result.meanEndpointError).toBeLessThan(0.01);
    expect(result.maxPointerError).toBeLessThan(0.02);
    expect(result.passRate).toBe(1);
  });

  it("returns a neutral result for an empty eval set", () => {
    const { model } = trainClickModel([{ x: 0.5, y: 0.5 }]);
    const result = evaluateMovementModel(model, []);
    expect(result).toEqual({
      cases: 0,
      meanPointerError: 0,
      maxPointerError: 0,
      meanEndpointError: 0,
      passRate: 1,
    });
  });
});

describe("generatePointerGesture", () => {
  it("produces deterministic, parametric demonstrations", () => {
    const first: MovementSequence[] = generatePointerGesture({
      label: "g",
      start: START,
      targets: [{ x: 0.7, y: 0.4 }],
      steps: 2,
      stepMs: 10,
    });
    const second = generatePointerGesture({
      label: "g",
      start: START,
      targets: [{ x: 0.7, y: 0.4 }],
      steps: 2,
      stepMs: 10,
    });
    expect(first).toEqual(second);
    expect(first[0]!.events).toHaveLength(3); // steps + 1
    expect(first[0]!.events.at(-1)).toMatchObject({ x: 0.7, y: 0.4, t: 20 });
    expect(first[0]!.params).toEqual({ targetX: 0.7, targetY: 0.4 });
  });
});
