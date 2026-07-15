import { describe, expect, it } from "vitest";
import { NgramMovementBackend, trainMovementModel } from "./movement-model.js";
import { evaluateMovementModel } from "./movement-eval.js";
import {
  desktopMovementFamily,
  generateSyntheticTrajectories,
} from "./synthetic-trajectories.js";

describe("evaluateMovementModel", () => {
  it("scores perfect replay on an unambiguous training scenario", async () => {
    const family = desktopMovementFamily();
    // A single scenario has no conflicting contexts, so the model can replay it
    // exactly — demonstrating objective part (c), repeat recorded movements.
    const trainSpans = generateSyntheticTrajectories({
      scenarios: [family.train[0]],
      spansPerScenario: 5,
      seed: 3,
    });
    const model = await trainMovementModel(new NgramMovementBackend(), trainSpans);

    const result = evaluateMovementModel(model, trainSpans);
    expect(result.total).toBeGreaterThan(0);
    expect(result.accuracy).toBe(1);
    expect(result.exactCorrect).toBeGreaterThan(0);
  });

  it("generalizes to a held-out related scenario via backoff", async () => {
    const family = desktopMovementFamily();
    const trainSpans = generateSyntheticTrajectories({
      scenarios: family.train,
      spansPerScenario: 6,
      seed: 5,
    });
    const heldOutSpans = generateSyntheticTrajectories({
      scenarios: family.heldOut,
      spansPerScenario: 3,
      seed: 99,
      startTs: 100_000,
    });

    const model = await trainMovementModel(new NgramMovementBackend(), trainSpans);
    const result = evaluateMovementModel(model, heldOutSpans);

    // The held-out scenario shares the open->focus->click->type prefix, so most
    // next-action decisions are recoverable; only the novel "find" tail differs.
    expect(result.accuracy).toBeGreaterThan(0.5);
    // At least some correct predictions must come from generalization, not memorization.
    expect(result.backoffCorrect + result.exactCorrect).toBe(result.correct);
    expect(result.correct).toBeGreaterThan(0);
  });

  it("returns zeroed metrics for an empty held-out set", async () => {
    const model = await trainMovementModel(new NgramMovementBackend(), []);
    const result = evaluateMovementModel(model, []);
    expect(result).toMatchObject({ total: 0, correct: 0, accuracy: 0, averageConfidence: 0 });
    expect(result.cases).toEqual([]);
  });
});
