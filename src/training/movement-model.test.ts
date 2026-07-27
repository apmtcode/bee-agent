import { describe, expect, it } from "vitest";
import {
  SequenceMovementBackend,
  SequenceMovementModel,
  createMovementBackend,
  evaluateMovementModel,
  movementToken,
  type MovementSequence,
  type MovementStep,
} from "./movement-model.js";
import { generateSyntheticDataset } from "./synthetic-events.js";

function seq(id: string, context: string, steps: MovementStep[]): MovementSequence {
  return { id, context, steps };
}

const WORKFLOW: MovementStep[] = [
  { actor: "window", action: "focus", target: "editor" },
  { actor: "mouse", action: "click", target: "file:tree" },
  { actor: "keyboard", action: "shortcut", value: "cmd+p" },
  { actor: "keyboard", action: "type", target: "field:quick-open", value: "readme" },
  { actor: "keyboard", action: "shortcut", value: "enter" },
];

describe("SequenceMovementBackend — repeat (exact recall)", () => {
  it("reproduces a recorded movement exactly from its prefix", () => {
    const model = createMovementBackend().train({
      sequences: [seq("a", "editor", WORKFLOW), seq("b", "editor", WORKFLOW)],
    });

    // Generate from just the first step; it should reconstruct the whole workflow.
    const result = model.generate("editor", WORKFLOW.slice(0, 1), 16);
    expect(result.steps.map(movementToken)).toEqual(WORKFLOW.map(movementToken));
    expect(result.terminated).toBe(true);
  });

  it("predicts the exact next step with recall provenance and full confidence", () => {
    const model = createMovementBackend().train({ sequences: [seq("a", "editor", WORKFLOW)] });
    const prediction = model.predictNext("editor", WORKFLOW.slice(0, 2));
    expect(prediction.step && movementToken(prediction.step)).toBe(movementToken(WORKFLOW[2]!));
    expect(prediction.provenance).toBe("recall");
    expect(prediction.confidence).toBe(1);
    expect(prediction.end).toBe(false);
  });

  it("predicts END at the tail of a recorded sequence", () => {
    const model = createMovementBackend().train({ sequences: [seq("a", "editor", WORKFLOW)] });
    const prediction = model.predictNext("editor", WORKFLOW);
    expect(prediction.end).toBe(true);
  });
});

describe("SequenceMovementBackend — generalization", () => {
  it("back-off predicts a plausible continuation for a related but novel prefix", () => {
    // Two workflows that share the "keyboard shortcut cmd+p" -> "type quick-open" bigram.
    const shared: MovementStep = { actor: "keyboard", action: "shortcut", value: "cmd+p" };
    const typeStep: MovementStep = { actor: "keyboard", action: "type", target: "field:quick-open", value: "x" };
    const model = createMovementBackend().train({
      sequences: [
        seq("a", "editor", [{ actor: "window", action: "focus", target: "editor" }, shared, typeStep]),
        seq("b", "editor", [{ actor: "mouse", action: "click", target: "menu" }, shared, typeStep]),
      ],
    });

    // A novel prefix ending in the shared bigram: model should still know what follows.
    const prediction = model.predictNext("editor", [
      { actor: "gesture", action: "swipe", direction: "up" },
      shared,
    ]);
    expect(prediction.step && movementToken(prediction.step)).toBe(movementToken(typeStep));
  });

  it("primes a first movement from the bare context label (cold start = generalization)", () => {
    const model = createMovementBackend().train({ sequences: [seq("a", "editor", WORKFLOW)] });
    const prediction = model.predictNext("editor", []);
    expect(prediction.step && movementToken(prediction.step)).toBe(movementToken(WORKFLOW[0]!));
    expect(prediction.provenance).toBe("generalization");
  });
});

describe("determinism", () => {
  it("produces identical models and predictions across runs", () => {
    const dataset = { sequences: [seq("a", "editor", WORKFLOW), seq("b", "editor", WORKFLOW)] };
    const a = new SequenceMovementBackend().train(dataset).serialize();
    const b = new SequenceMovementBackend().train(dataset).serialize();
    expect(a).toEqual(b);
  });

  it("breaks argmax ties deterministically by token order", () => {
    // Context "editor" primes two equally-frequent first steps; the lexicographically
    // smaller token must always win.
    const s1: MovementStep = { actor: "keyboard", action: "type", value: "a" };
    const s2: MovementStep = { actor: "window", action: "focus" };
    const model = createMovementBackend().train({
      sequences: [seq("a", "editor", [s1]), seq("b", "editor", [s2])],
    });
    const first = model.predictNext("editor", []);
    const expected = movementToken(s1) < movementToken(s2) ? s1 : s2;
    expect(first.step && movementToken(first.step)).toBe(movementToken(expected));
  });
});

describe("serialization", () => {
  it("round-trips through JSON with identical behaviour", () => {
    const model = createMovementBackend().train({ sequences: [seq("a", "editor", WORKFLOW)] });
    const restored = SequenceMovementModel.load(JSON.parse(JSON.stringify(model.serialize())));
    const original = model.generate("editor", WORKFLOW.slice(0, 1), 16);
    const replayed = restored.generate("editor", WORKFLOW.slice(0, 1), 16);
    expect(replayed.steps.map(movementToken)).toEqual(original.steps.map(movementToken));
  });

  it("rejects an unknown serialized model kind", () => {
    expect(() => SequenceMovementModel.load({ kind: "mystery" } as never)).toThrow(/unsupported/);
  });
});

describe("evaluateMovementModel — generalization harness", () => {
  it("scores high on held-out sequences from the same workflows", () => {
    const { train, heldOut } = generateSyntheticDataset({
      seed: 7,
      trainPerWorkflow: 12,
      heldOutPerWorkflow: 4,
      variation: 0.15,
    });
    const model = createMovementBackend().train({ sequences: train });
    const report = evaluateMovementModel(model, heldOut);
    expect(report.sequences).toBe(heldOut.length);
    expect(report.steps).toBeGreaterThan(0);
    // The model has learned the workflows; held-out next-step accuracy should be strong.
    expect(report.stepAccuracy).toBeGreaterThan(0.8);
    expect(report.stepAccuracy).toBeLessThanOrEqual(1);
  });

  it("returns zeroed metrics for an empty held-out set", () => {
    const model = createMovementBackend().train({ sequences: [seq("a", "editor", WORKFLOW)] });
    const report = evaluateMovementModel(model, []);
    expect(report).toMatchObject({ sequences: 0, steps: 0, stepAccuracy: 0, sequenceAccuracy: 0 });
  });
});

describe("generate — step cap", () => {
  it("never exceeds the requested budget beyond the prefix", () => {
    // A pathological self-looping dataset that never emits END within budget.
    const loop: MovementStep = { actor: "mouse", action: "move", direction: "down" };
    const model = createMovementBackend().train({
      sequences: [seq("a", "x", [loop, loop, loop, loop, loop, loop])],
    });
    const result = model.generate("x", [loop], 3);
    expect(result.steps.length).toBeLessThanOrEqual(1 + 3);
  });
});
