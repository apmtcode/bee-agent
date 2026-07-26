import { describe, expect, it } from "vitest";
import {
  evaluateNextActionPrediction,
  evaluateReplayFidelity,
  structuralSignature,
} from "./generalization-eval.js";
import { MarkovMovementBackend, buildMovementDataset } from "./movement-model.js";
import {
  FORM_FILL_GRAMMAR,
  createSeededRng,
  generateSyntheticTrajectories,
  withNovelTargets,
} from "./synthetic-movements.js";

describe("generalization eval (end-to-end movement pipeline)", () => {
  it("reproduces training movements with near-perfect next-action prediction", () => {
    const trajectories = generateSyntheticTrajectories({
      grammar: FORM_FILL_GRAMMAR,
      count: 30,
      rng: createSeededRng(101),
    });
    const dataset = buildMovementDataset(trajectories);
    const model = new MarkovMovementBackend().train(dataset, { order: 3 });

    const result = evaluateNextActionPrediction(model, dataset.sequences);
    expect(result.structuralAccuracy).toBe(1);
    // targets are memorized well on the training distribution
    expect(result.exactAccuracy).toBeGreaterThan(0.5);
  });

  it("generalizes movement structure to a held-out set with novel targets", () => {
    // Train on the original targets.
    const train = generateSyntheticTrajectories({
      grammar: FORM_FILL_GRAMMAR,
      count: 40,
      rng: createSeededRng(202),
    });
    const model = new MarkovMovementBackend().train(buildMovementDataset(train), { order: 3 });

    // Evaluate on structurally identical movements against targets never seen.
    const heldOut = generateSyntheticTrajectories({
      grammar: withNovelTargets(FORM_FILL_GRAMMAR),
      count: 20,
      rng: createSeededRng(303),
    });
    const result = evaluateNextActionPrediction(model, buildMovementDataset(heldOut).sequences);

    // Structure transfers perfectly even though every target is new...
    expect(result.structuralAccuracy).toBe(1);
    // ...and it did so by leaning on abstraction backoff (the generalization path).
    expect(result.abstractionRate).toBeGreaterThan(0);
    // Concrete targets do NOT transfer (honest signal: novel targets are unknown).
    expect(result.exactAccuracy).toBe(0);
  });

  it("measures replay fidelity via free rollout", () => {
    const trajectories = generateSyntheticTrajectories({
      grammar: FORM_FILL_GRAMMAR,
      count: 25,
      rng: createSeededRng(404),
    });
    const dataset = buildMovementDataset(trajectories);
    const model = new MarkovMovementBackend().train(dataset, { order: 4 });

    const fidelity = evaluateReplayFidelity(model, dataset.sequences);
    expect(fidelity.sequenceCount).toBe(25);
    // The dominant memorized rollout should structurally match most references.
    expect(fidelity.averageStructuralOverlap).toBeGreaterThan(0.7);
  });

  it("reports a readable structural signature", () => {
    const [trajectory] = generateSyntheticTrajectories({
      grammar: FORM_FILL_GRAMMAR,
      count: 1,
      rng: createSeededRng(5),
    });
    const dataset = buildMovementDataset([trajectory]);
    const signature = structuralSignature(dataset.sequences[0].tokens);
    expect(signature).toContain("->");
    expect(signature.startsWith("device")).toBe(true);
  });
});
