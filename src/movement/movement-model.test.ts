import { describe, expect, it } from "vitest";
import {
  MarkovMovementBackend,
  deserializeMovementModel,
  evaluateReplayFidelity,
  tokenizeStep,
  type MovementDataset,
  type MovementStep,
} from "./movement-model.js";

function step(tool: string, action: string, target?: string): MovementStep {
  return target ? { tool, action, target } : { tool, action };
}

const flowA: MovementStep[] = [
  step("mouse", "click", "app-launcher"),
  step("mouse", "click", "search-box"),
  step("keyboard", "type", "search-box"),
  step("keyboard", "shortcut", "enter"),
  step("mouse", "click", "first-result"),
];

describe("MarkovMovementBackend", () => {
  it("repeats a recorded movement sequence exactly (objective 2c)", async () => {
    const dataset: MovementDataset = { sequences: [{ id: "s1", steps: flowA }] };
    const model = await new MarkovMovementBackend().train(dataset, { order: 2 });

    const generated = model.generate([flowA[0]!]);
    expect(generated.map(tokenizeStep)).toEqual(flowA.slice(1).map(tokenizeStep));
  });

  it("terminates generation at end-of-sequence rather than looping forever", async () => {
    const dataset: MovementDataset = { sequences: [{ id: "s1", steps: flowA }] };
    const model = await new MarkovMovementBackend().train(dataset, { order: 2 });

    const generated = model.generate([flowA[0]!], { maxSteps: 100 });
    // Exactly the recorded tail — it stops at EOS, well under the cap.
    expect(generated).toHaveLength(flowA.length - 1);
  });

  it("generalizes to a novel-but-related prefix via backoff (objective 2d)", async () => {
    // Two flows that share the "click search-box -> type" transition.
    const flowB: MovementStep[] = [
      step("mouse", "click", "search-box"),
      step("keyboard", "type", "search-box"),
      step("mouse", "click", "second-result"),
    ];
    const dataset: MovementDataset = {
      sequences: [
        { id: "a", steps: flowA },
        { id: "b", steps: flowB },
      ],
    };
    const model = await new MarkovMovementBackend().train(dataset, { order: 2 });

    // A prefix whose full order-2 context was never seen, but whose last step
    // ("click search-box") was. Backoff should still predict a valid next step.
    const novelPrefix = [step("device", "tap", "home"), step("mouse", "click", "search-box")];
    const prediction = model.predictNext(novelPrefix);
    expect(prediction).toBeDefined();
    expect(prediction!.backoff).toBe(true);
    expect(prediction!.step).toEqual(step("keyboard", "type", "search-box"));
    // The predicted step is drawn from the learned vocabulary, not invented.
    expect(model.vocabulary).toContain(tokenizeStep(prediction!.step));
  });

  it("predicts nothing for an empty model", async () => {
    const model = await new MarkovMovementBackend().train({ sequences: [] });
    expect(model.predictNext([step("mouse", "click", "x")])).toBeUndefined();
    expect(model.generate([step("mouse", "click", "x")])).toEqual([]);
  });

  it("round-trips through serialization without behaviour change", async () => {
    const dataset: MovementDataset = { sequences: [{ id: "s1", steps: flowA }] };
    const model = await new MarkovMovementBackend().train(dataset, { order: 2 });
    const serialized = model.toJSON();

    // JSON-safe and stable.
    expect(() => JSON.stringify(serialized)).not.toThrow();
    expect(Object.keys(serialized.transitions).length).toBeGreaterThan(0);
    expect(serialized.vocabulary[tokenizeStep(flowA[0]!)]).toEqual(flowA[0]);

    const restored = deserializeMovementModel(serialized);
    expect(restored.generate([flowA[0]!]).map(tokenizeStep)).toEqual(
      model.generate([flowA[0]!]).map(tokenizeStep),
    );
    expect(restored.toJSON()).toEqual(serialized);
  });

  it("is deterministic across retrains (no clock/RNG)", async () => {
    const dataset: MovementDataset = { sequences: [{ id: "s1", steps: flowA }] };
    const first = await new MarkovMovementBackend().train(dataset, { order: 2 });
    const second = await new MarkovMovementBackend().train(dataset, { order: 2 });
    expect(first.toJSON()).toEqual(second.toJSON());
  });
});

describe("evaluateReplayFidelity", () => {
  it("scores perfect fidelity on recorded sequences", async () => {
    const dataset: MovementDataset = { sequences: [{ id: "s1", steps: flowA }] };
    const model = await new MarkovMovementBackend().train(dataset, { order: 2 });

    const report = evaluateReplayFidelity(model, dataset.sequences, { seedSteps: 1 });
    expect(report.stepAccuracy).toBe(1);
    expect(report.exactSequenceRate).toBe(1);
    expect(report.perSequence[0]!.exact).toBe(true);
  });

  it("reports degraded fidelity on unrelated held-out sequences", async () => {
    const dataset: MovementDataset = { sequences: [{ id: "s1", steps: flowA }] };
    const model = await new MarkovMovementBackend().train(dataset, { order: 2 });

    const heldOut = [
      {
        id: "unrelated",
        steps: [step("gamepad", "press", "a"), step("gamepad", "press", "b"), step("gamepad", "press", "x")],
      },
    ];
    const report = evaluateReplayFidelity(model, heldOut, { seedSteps: 1 });
    expect(report.stepAccuracy).toBeLessThan(1);
    expect(report.exactSequenceRate).toBe(0);
  });
});
