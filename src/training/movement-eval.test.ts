import { describe, expect, it } from "vitest";
import { MarkovMovementBackend, type MovementDataset } from "./movement-model.js";
import { evaluateMovementModel } from "./movement-eval.js";

function dataset(sequences: string[][]): MovementDataset {
  return {
    version: 1,
    sequences: sequences.map((tokens, index) => ({ id: `seq-${index}`, tokens })),
  };
}

describe("evaluateMovementModel", () => {
  it("scores perfect fidelity when replaying a memorized sequence", () => {
    const backend = new MarkovMovementBackend();
    const train = dataset([["a", "b", "c", "d"]]);
    const model = backend.train(train, { order: 2 });

    const result = evaluateMovementModel(model, train);
    expect(result.nextTokenAccuracy).toBe(1);
    expect(result.replayFidelity).toBe(1);
    expect(result.predictions).toBe(4);
    expect(result.correct).toBe(4);
  });

  it("measures partial fidelity on held-out but related sequences", () => {
    const backend = new MarkovMovementBackend();
    const model = backend.train(
      dataset([
        ["open", "search", "select", "confirm"],
        ["open", "search", "select", "confirm"],
      ]),
      { order: 2 },
    );

    // Held-out sequence shares the learned tail but starts differently.
    const heldOut = dataset([["launch", "search", "select", "confirm"]]);
    const result = evaluateMovementModel(model, heldOut);

    expect(result.nextTokenAccuracy).toBeGreaterThan(0);
    expect(result.nextTokenAccuracy).toBeLessThanOrEqual(1);
    expect(result.sequences).toBe(1);
  });

  it("reports perfect scores for an empty dataset", () => {
    const backend = new MarkovMovementBackend();
    const model = backend.train(dataset([["a"]]), { order: 1 });

    const result = evaluateMovementModel(model, dataset([]));
    expect(result.nextTokenAccuracy).toBe(1);
    expect(result.replayFidelity).toBe(1);
    expect(result.predictions).toBe(0);
  });
});
