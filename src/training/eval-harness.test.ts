import { describe, expect, it } from "vitest";
import { DeterministicMarkovBackend, type MovementDataset, type MovementSample } from "./model-backend.js";
import { evaluateMovementModel } from "./eval-harness.js";

function seq(sourceSessionId: string, ...tokens: string[]): MovementSample {
  return { sourceSessionId, tokens };
}

const trainingDataset: MovementDataset = {
  samples: [
    seq("train-1", "obs:app", "act:focus", "act:click", "act:type", "act:submit"),
    seq("train-2", "obs:app", "act:focus", "act:click", "act:type", "act:submit"),
  ],
};

describe("evaluateMovementModel", () => {
  it("reports perfect fidelity when held-out data matches the trained sequence", async () => {
    const model = await new DeterministicMarkovBackend(2).train(trainingDataset);
    const report = evaluateMovementModel(model, [
      seq("holdout-1", "obs:app", "act:focus", "act:click", "act:type", "act:submit"),
    ]);
    expect(report.backendId).toBe("deterministic-markov");
    expect(report.nextTokenAccuracy).toBe(1);
    expect(report.replayFidelity).toBe(1);
    expect(report.evaluatedTransitions).toBe(4);
    // With a fully-recognized sequence and order-2 seed, predictions ride the
    // full-order context.
    expect(report.backoffProfile.fullOrderShare).toBeGreaterThan(0);
  });

  it("measures generalization on a related-but-unseen sequence", async () => {
    const model = await new DeterministicMarkovBackend(2).train(trainingDataset);
    // Same action dynamics, different leading observation → forces backoff.
    const report = evaluateMovementModel(model, [
      seq("holdout-2", "obs:OTHER", "act:focus", "act:click", "act:type", "act:submit"),
    ]);
    // The model still predicts most next moves correctly, but via shorter
    // contexts — so we expect real accuracy plus a nonzero backoff share.
    expect(report.nextTokenAccuracy).toBeGreaterThan(0.5);
    expect(report.backoffProfile.backoffShare).toBeGreaterThan(0);
  });

  it("scores lower on unrelated held-out movements", async () => {
    const model = await new DeterministicMarkovBackend(2).train(trainingDataset);
    const report = evaluateMovementModel(model, [
      seq("holdout-3", "act:unrelated", "act:mystery", "act:noise"),
    ]);
    expect(report.nextTokenAccuracy).toBeLessThan(1);
  });

  it("handles empty and single-token samples without dividing by zero", async () => {
    const model = await new DeterministicMarkovBackend(2).train(trainingDataset);
    const report = evaluateMovementModel(model, [
      { sourceSessionId: "empty", tokens: [] },
      { sourceSessionId: "single", tokens: ["act:focus"] },
    ]);
    expect(report.evaluatedTransitions).toBe(0);
    expect(report.nextTokenAccuracy).toBe(0);
    expect(report.replayFidelity).toBe(0);
    expect(report.sampleCount).toBe(2);
  });

  it("respects a custom seed length for rollout scoring", async () => {
    const model = await new DeterministicMarkovBackend(2).train(trainingDataset);
    const report = evaluateMovementModel(
      model,
      [seq("holdout-4", "obs:app", "act:focus", "act:click", "act:type", "act:submit")],
      { seedLength: 4 },
    );
    // Seed of 4 leaves a single-token continuation to reproduce.
    expect(report.perSample[0]?.rolloutLength).toBe(1);
    expect(report.replayFidelity).toBe(1);
  });
});
