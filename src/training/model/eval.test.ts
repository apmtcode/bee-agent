import { describe, expect, it } from "vitest";
import { MockMovementBackend, type MovementDataset } from "./movement-model.js";
import {
  evaluateGeneralization,
  evaluateNextEvent,
  evaluateRolloutFidelity,
} from "./eval.js";
import {
  BUILTIN_MOVEMENT_TEMPLATES,
  generateSyntheticSequence,
} from "./synthetic.js";

/**
 * Train on some seeds of each template, evaluate on held-out seeds of the SAME
 * templates. The held-out sequences are structurally identical (same movement
 * shapes) but generated from unseen seeds — a clean test of generalization to
 * "new but related" movements.
 */
function split() {
  const train = BUILTIN_MOVEMENT_TEMPLATES.flatMap((template) =>
    [1, 2, 3, 4].map((seed) => generateSyntheticSequence(template, seed)),
  );
  const heldOut = BUILTIN_MOVEMENT_TEMPLATES.flatMap((template) =>
    [101, 102].map((seed) => generateSyntheticSequence(template, seed)),
  );
  return { train, heldOut };
}

describe("generalization eval harness", () => {
  it("predicts held-out next events well above chance", async () => {
    const { train, heldOut } = split();
    const dataset: MovementDataset = { version: 1, sequences: train };
    const model = await new MockMovementBackend().train(dataset);
    const result = evaluateNextEvent(model, heldOut);
    expect(result.predictions).toBeGreaterThan(0);
    // The templates are deterministic in structure, so token-level next-event
    // accuracy on held-out seeds should be high.
    expect(result.accuracy).toBeGreaterThan(0.8);
  });

  it("reproduces held-out movement shapes with high rollout fidelity", async () => {
    const { train, heldOut } = split();
    const model = await new MockMovementBackend().train({ version: 1, sequences: train });
    // Seed with 2 real events so the shared "focus {app}" prefix is disambiguated
    // to a single template, then free-run the rest.
    const result = evaluateRolloutFidelity(model, heldOut, { seedLength: 2 });
    expect(result.sequences).toBe(heldOut.length);
    expect(result.meanFidelity).toBeGreaterThan(0.8);
  });

  it("returns a combined report", async () => {
    const { train, heldOut } = split();
    const model = await new MockMovementBackend().train({ version: 1, sequences: train });
    const report = evaluateGeneralization(model, heldOut);
    expect(report.nextEvent.accuracy).toBeGreaterThan(0);
    expect(report.rollout.meanFidelity).toBeGreaterThan(0);
  });

  it("reports zero cleanly on an empty held-out set", async () => {
    const model = await new MockMovementBackend().train({ version: 1, sequences: [] });
    const report = evaluateGeneralization(model, []);
    expect(report.nextEvent.accuracy).toBe(0);
    expect(report.rollout.meanFidelity).toBe(0);
    expect(report.rollout.exactMatchRate).toBe(0);
  });
});
