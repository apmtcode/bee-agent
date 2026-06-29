import { describe, expect, it } from "vitest";
import { evaluateMovementModel } from "./movement-eval.js";
import {
  NGramMovementBackend,
  type MovementDataset,
  type MovementSequence,
  type MovementToken,
} from "./movement-model.js";

function tok(tool: string, summary: string): MovementToken {
  return { tool, summary };
}

function seq(trajectoryId: string, tokens: MovementToken[]): MovementSequence {
  return { trajectoryId, tokens };
}

describe("evaluateMovementModel", () => {
  it("scores perfect fidelity when held-out matches a learned sequence", () => {
    const tokens = [tok("ui", "focus"), tok("ui", "open menu"), tok("ui", "select save"), tok("ui", "confirm")];
    const dataset: MovementDataset = { version: 1, sequences: [seq("train", tokens)] };
    const model = new NGramMovementBackend(2).train(dataset);

    const report = evaluateMovementModel(model, [seq("heldout", tokens)], { promptLength: 1 });
    expect(report.nextToken.accuracy).toBe(1);
    expect(report.rollout.fidelity).toBe(1);
    expect(report.rollout.perSequence[0].matched).toBe(3);
  });

  it("generalizes: high next-token accuracy on a novel recombination of learned sub-movements", () => {
    // Train on two flows that share the "open menu → select save" sub-movement.
    const dataset: MovementDataset = {
      version: 1,
      sequences: [
        seq("a", [tok("ui", "focus"), tok("ui", "open menu"), tok("ui", "select save")]),
        seq("b", [tok("ui", "scroll"), tok("ui", "open menu"), tok("ui", "select save")]),
      ],
    };
    const model = new NGramMovementBackend(2).train(dataset);

    // Held-out flow primes with an unseen first move but reuses the shared tail.
    const heldOut = seq("c", [tok("ui", "drag"), tok("ui", "open menu"), tok("ui", "select save")]);
    const report = evaluateMovementModel(model, [heldOut], { promptLength: 1 });

    // The "open menu → select save" transition is learned, so it generalizes to
    // the held-out flow despite the never-seen "drag" prefix.
    const tail = model.predict([tok("ui", "drag"), tok("ui", "open menu")]);
    expect(tail?.token).toEqual(tok("ui", "select save"));
    expect(report.nextToken.correct).toBeGreaterThanOrEqual(1);
  });

  it("skips sequences shorter than the prompt length for rollout but still scores next-token", () => {
    const dataset: MovementDataset = { version: 1, sequences: [seq("train", [tok("a", "1"), tok("b", "2")])] };
    const model = new NGramMovementBackend(2).train(dataset);

    const report = evaluateMovementModel(model, [seq("short", [tok("a", "1")])], { promptLength: 2 });
    expect(report.rollout.perSequence).toHaveLength(0);
    expect(report.rollout.expected).toBe(0);
    expect(report.rollout.fidelity).toBe(0);
    // A length-1 held-out sequence yields no next-token predictions either.
    expect(report.nextToken.predictions).toBe(0);
  });

  it("defaults the rollout prompt length to the model order", () => {
    const tokens = [tok("a", "1"), tok("a", "2"), tok("a", "3"), tok("a", "4")];
    const dataset: MovementDataset = { version: 1, sequences: [seq("train", tokens)] };
    const model = new NGramMovementBackend(2).train(dataset);
    const report = evaluateMovementModel(model, [seq("heldout", tokens)]);
    expect(report.rollout.perSequence[0].promptLength).toBe(2);
  });
});
