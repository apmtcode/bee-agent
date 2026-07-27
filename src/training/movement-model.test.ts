import { describe, expect, it } from "vitest";
import {
  MarkovMovementBackend,
  MovementBackendRegistry,
  createDefaultMovementBackendRegistry,
  evaluateNextStepAccuracy,
  type MovementSequence,
} from "./movement-model.js";
import { generateSyntheticMovements } from "./synthetic-movements.js";

function seq(id: string, actions: string[]): MovementSequence {
  return { id, steps: actions.map((action) => ({ action })) };
}

describe("MarkovMovementBackend", () => {
  it("learns and reproduces a recorded movement sequence", () => {
    const dataset = [
      seq("a", ["mouse.move", "mouse.down", "mouse.move", "mouse.up"]),
      seq("b", ["mouse.move", "mouse.down", "mouse.move", "mouse.up"]),
    ];
    const model = new MarkovMovementBackend().train(dataset, { order: 2 });

    const rollout = model.generate([{ action: "mouse.move" }], 3);
    expect(rollout.map((step) => step.action)).toEqual(["mouse.down", "mouse.move", "mouse.up"]);
  });

  it("predicts the next action with a full-order context hit", () => {
    const model = new MarkovMovementBackend().train(
      [seq("a", ["window.focus", "key.press", "key.press"])],
      { order: 2 },
    );
    const prediction = model.predictNext([{ action: "window.focus" }, { action: "key.press" }]);
    expect(prediction?.action).toBe("key.press");
    expect(prediction?.matchedContext).toBe(2);
    expect(prediction?.probability).toBeCloseTo(1, 5);
  });

  it("backs off to a shorter suffix when the full context is unseen", () => {
    // Trained only on this bigram context; a novel long context should still
    // resolve via the "key.press" -> "key.up" unigram/bigram suffix.
    const model = new MarkovMovementBackend().train(
      [seq("a", ["key.down", "key.press", "key.up"])],
      { order: 2 },
    );
    const prediction = model.predictNext([
      { action: "window.focus" },
      { action: "mouse.click" },
      { action: "key.press" },
    ]);
    expect(prediction?.action).toBe("key.up");
    expect(prediction?.matchedContext).toBeLessThan(2);
  });

  it("returns undefined when untrained", () => {
    const model = new MarkovMovementBackend().train([], { order: 2 });
    expect(model.predictNext([{ action: "mouse.move" }])).toBeUndefined();
    expect(model.generate([{ action: "mouse.move" }], 5)).toEqual([]);
  });

  it("builds a numeric-mean / string-mode prototype for generated steps", () => {
    const dataset: MovementSequence[] = [
      {
        id: "a",
        steps: [
          { action: "start" },
          { action: "mouse.move", params: { x: 10, y: 20, button: "left" } },
        ],
      },
      {
        id: "b",
        steps: [
          { action: "start" },
          { action: "mouse.move", params: { x: 30, y: 40, button: "left" } },
        ],
      },
    ];
    const model = new MarkovMovementBackend().train(dataset, { order: 1 });
    const [generated] = model.generate([{ action: "start" }], 1);
    expect(generated.action).toBe("mouse.move");
    expect(generated.params).toEqual({ x: 20, y: 30, button: "left" });
  });

  it("round-trips through serialize/load with identical predictions", () => {
    const dataset = [seq("a", ["a", "b", "c", "a", "b", "c"])];
    const backend = new MarkovMovementBackend();
    const model = backend.train(dataset, { order: 2 });
    const restored = backend.load(model.serialize());

    const context = [{ action: "a" }, { action: "b" }];
    expect(restored.predictNext(context)).toEqual(model.predictNext(context));
    expect(restored.vocabulary()).toEqual(model.vocabulary());
    expect(restored.serialize()).toEqual(model.serialize());
  });

  it("is deterministic across identical trainings", () => {
    const dataset = generateSyntheticMovements({ seed: 7, sequenceCount: 20 });
    const a = new MarkovMovementBackend().train(dataset, { order: 2 }).serialize();
    const b = new MarkovMovementBackend().train(dataset, { order: 2 }).serialize();
    expect(a).toEqual(b);
  });
});

describe("evaluateNextStepAccuracy (generalization harness)", () => {
  it("generalizes to held-out but related synthetic movements", () => {
    const train = generateSyntheticMovements({ seed: 1, sequenceCount: 120 });
    // Same grammar, different seed => structurally related, parameter-different.
    const heldOut = generateSyntheticMovements({ seed: 999, sequenceCount: 40 });

    const model = new MarkovMovementBackend().train(train, { order: 2 });
    const result = evaluateNextStepAccuracy(model, heldOut);

    expect(result.sampleCount).toBeGreaterThan(0);
    // Some contexts are genuinely ambiguous (the same suffix continues
    // differently across tasks), so top-1 cannot reach 1.0 — but the model
    // still vastly beats the random baseline (~1/vocabulary ≈ 0.12) on related
    // held-out sequences it never saw, which is the generalization claim.
    expect(result.accuracy).toBeGreaterThan(0.85);
    expect(result.accuracy).toBeGreaterThan(model.vocabulary().length > 0 ? 1 / model.vocabulary().length : 0);
    expect(result.unpredicted).toBe(0);
  });

  it("reports zero-sample eval safely on empty held-out data", () => {
    const model = new MarkovMovementBackend().train(
      generateSyntheticMovements({ seed: 3, sequenceCount: 5 }),
      { order: 2 },
    );
    const result = evaluateNextStepAccuracy(model, []);
    expect(result).toEqual({ sampleCount: 0, correct: 0, accuracy: 0, backoffRate: 0, unpredicted: 0 });
  });
});

describe("MovementBackendRegistry", () => {
  it("resolves the default markov backend and rejects unknown ids", () => {
    const registry = createDefaultMovementBackendRegistry();
    expect(registry.list()).toContain("markov");
    expect(registry.has("markov")).toBe(true);
    expect(registry.resolve("markov").id).toBe("markov");
    expect(() => registry.resolve("nope")).toThrow(/unknown movement backend: nope/);
  });

  it("supports registering an alternate (pluggable) backend", () => {
    const registry = new MovementBackendRegistry();
    const backend = new MarkovMovementBackend();
    registry.register({ ...backend, id: "custom", train: backend.train.bind(backend), load: backend.load.bind(backend) });
    expect(registry.list()).toEqual(["custom"]);
    expect(registry.resolve("custom").train([], { order: 1 }).backendId).toBeDefined();
  });
});
