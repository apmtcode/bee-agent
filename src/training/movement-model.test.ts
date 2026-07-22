import { describe, expect, it } from "vitest";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import {
  MarkovMovementBackend,
  defaultSyntheticWorkflows,
  evaluateMovementModel,
  generateSyntheticMovementDataset,
  loadMovementModel,
  tokenizeTrajectory,
  type MovementDataset,
  type MovementSequence,
} from "./movement-model.js";

function seq(id: string, steps: MovementSequence["steps"]): MovementSequence {
  return { id, steps };
}

describe("MarkovMovementBackend", () => {
  it("repeats a recorded movement exactly from an exact context", () => {
    const dataset: MovementDataset = {
      sequences: [
        seq("login", [
          { gesture: "tap", target: "Email field" },
          { gesture: "type", target: "Email field", valueSummary: "typed email" },
          { gesture: "tap", target: "Password field" },
          { gesture: "tap", target: "Sign in button" },
        ]),
      ],
    };
    const model = new MarkovMovementBackend().train(dataset, { order: 2 });

    const generated = model.generate(dataset.sequences[0]!.steps.slice(0, 1), 3);
    expect(generated).toEqual(dataset.sequences[0]!.steps.slice(1));

    const prediction = model.predictNext(dataset.sequences[0]!.steps.slice(0, 2));
    expect(prediction.backoff).toBe("exact");
    expect(prediction.step).toEqual({ gesture: "tap", target: "Password field" });
  });

  it("is deterministic: same dataset yields identical serialized weights", () => {
    const dataset = generateSyntheticMovementDataset({ templates: defaultSyntheticWorkflows() });
    const a = new MarkovMovementBackend().train(dataset).serialize();
    const b = new MarkovMovementBackend().train(dataset).serialize();
    expect(a).toEqual(b);
  });

  it("generalizes to a new but related movement via gesture-class backoff", () => {
    // Train on form-submit workflows that all end in "tap {submit}" after "type".
    const training: MovementDataset = {
      sequences: [
        seq("f1", [
          { gesture: "tap", target: "Email field" },
          { gesture: "type", target: "Email field", valueSummary: "x" },
          { gesture: "tap", target: "Sign in button" },
        ]),
        seq("f2", [
          { gesture: "tap", target: "Search box" },
          { gesture: "type", target: "Search box", valueSummary: "y" },
          { gesture: "tap", target: "Search button" },
        ]),
      ],
    };
    const model = new MarkovMovementBackend().train(training, { order: 2 });

    // Held-out context uses brand-new targets never seen in training.
    const context = [
      { gesture: "tap", target: "Phone field" },
      { gesture: "type", target: "Phone field", valueSummary: "z" },
    ];
    const prediction = model.predictNext(context);
    expect(prediction.backoff).toBe("class");
    // The exact target is unknown, but the gesture must generalize correctly.
    expect(prediction.step?.gesture).toBe("tap");
  });

  it("falls back to unigram when no context matches, and to none when empty", () => {
    const model = new MarkovMovementBackend().train({
      sequences: [seq("s", [{ gesture: "scroll", direction: "down" }, { gesture: "scroll", direction: "down" }])],
    });
    const prediction = model.predictNext([{ gesture: "pinch", target: "Nowhere" }]);
    expect(prediction.backoff).toBe("unigram");
    expect(prediction.step?.gesture).toBe("scroll");

    const empty = new MarkovMovementBackend().train({ sequences: [] });
    const emptyPrediction = empty.predictNext([{ gesture: "tap" }]);
    expect(emptyPrediction.backoff).toBe("none");
    expect(emptyPrediction.step).toBeUndefined();
    expect(empty.generate([{ gesture: "tap" }], 3)).toEqual([]);
  });

  it("round-trips through serialize/loadMovementModel", () => {
    const dataset = generateSyntheticMovementDataset({ templates: defaultSyntheticWorkflows() });
    const model = new MarkovMovementBackend().train(dataset, { order: 2 });
    const restored = loadMovementModel(model.serialize());

    for (const sequence of dataset.sequences) {
      for (let i = 1; i < sequence.steps.length; i += 1) {
        const context = sequence.steps.slice(0, i);
        expect(restored.predictNext(context)).toEqual(model.predictNext(context));
      }
    }
    expect(restored.serialize()).toEqual(model.serialize());
  });
});

describe("tokenizeTrajectory", () => {
  it("maps recorded device actions into movement steps using gesture metadata", () => {
    const span = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [
        {
          kind: "action",
          tool: "device",
          summary: "tapped Save button",
          ts: 1,
          metadata: { gesture: "tap", target: "Save button" },
        },
        {
          kind: "action",
          tool: "device",
          summary: "scrolled down",
          ts: 2,
          metadata: { gesture: "scroll", direction: "down" },
        },
      ],
    });
    const sequence = tokenizeTrajectory(span);
    expect(sequence.id).toBe("t1");
    expect(sequence.steps[0]).toMatchObject({ gesture: "tap", target: "Save button" });
    expect(sequence.steps[1]).toMatchObject({ gesture: "scroll", direction: "down" });
  });

  it("falls back to the tool name when no gesture metadata is present", () => {
    const span = buildTrajectorySpan({
      id: "t2",
      sessionId: "s1",
      actions: [{ kind: "action", tool: "keyboard", summary: "pressed enter", ts: 1 }],
    });
    const sequence = tokenizeTrajectory(span);
    expect(sequence.steps[0]?.gesture).toBe("keyboard");
  });
});

describe("evaluateMovementModel", () => {
  it("reports perfect fidelity when evaluated on its own training data", () => {
    const dataset = generateSyntheticMovementDataset({ templates: defaultSyntheticWorkflows() });
    const model = new MarkovMovementBackend().train(dataset, { order: 2 });
    const result = evaluateMovementModel(model, dataset);

    expect(result.predictions).toBeGreaterThan(0);
    expect(result.exactAccuracy).toBe(1);
    expect(result.gestureAccuracy).toBe(1);
    expect(result.fullyReproduced).toBe(dataset.sequences.length);
  });

  it("generalizes: high gesture accuracy on held-out variants even when exact target differs", () => {
    // Train on two variants, hold out a third with unseen targets but same skeleton.
    const template = defaultSyntheticWorkflows()[0]!;
    const train = generateSyntheticMovementDataset({
      templates: [{ ...template, variants: template.variants.slice(0, 2) }],
    });
    const heldOut = generateSyntheticMovementDataset({
      templates: [
        {
          ...template,
          variants: [{ field: "Brand new field", submit: "Brand new button" }],
        },
      ],
    });

    const model = new MarkovMovementBackend().train(train, { order: 2 });
    const result = evaluateMovementModel(model, heldOut);

    // Gesture-level generalization should be strong even though the exact targets
    // (and thus exact-match) are unseen.
    expect(result.gestureAccuracy).toBeGreaterThanOrEqual(0.5);
    expect(result.gestureAccuracy).toBeGreaterThan(result.exactAccuracy);
    expect(result.backoffBreakdown.class).toBeGreaterThan(0);
  });
});
