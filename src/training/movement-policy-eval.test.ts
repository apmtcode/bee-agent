import { describe, expect, it } from "vitest";
import { NgramMovementPolicyBackend, type MovementExample } from "./movement-policy.js";
import {
  evaluateMovementPolicy,
  generateSyntheticMovementExamples,
} from "./movement-policy-eval.js";

const backend = new NgramMovementPolicyBackend();

function example(id: string, tokens: string[]): MovementExample {
  return { trajectoryId: id, tokens };
}

describe("evaluateMovementPolicy", () => {
  it("scores a memorized trajectory as perfect replay", () => {
    const tokens = ["focus::a", "click::b", "type::c", "click::save"];
    const model = backend.train([example("t1", tokens)], { maxOrder: 3 });

    const result = evaluateMovementPolicy(model, [example("t1", tokens)]);
    expect(result.trajectoryCount).toBe(1);
    expect(result.top1Accuracy).toBe(1);
    expect(result.topKAccuracy).toBe(1);
    expect(result.exactReplayRate).toBe(1);
    expect(result.predictionCount).toBe(tokens.length); // includes the end sentinel transition
  });

  it("reports zero accuracy for an empty model", () => {
    const model = backend.train([], { maxOrder: 2 });
    const result = evaluateMovementPolicy(model, [example("t1", ["focus::a", "click::b"])]);
    expect(result.top1Accuracy).toBe(0);
    expect(result.exactReplayRate).toBe(0);
  });

  it("generalizes from a training split to a disjoint held-out split", () => {
    const train = generateSyntheticMovementExamples({ seed: 1, trajectoryCount: 40 });
    const heldOut = generateSyntheticMovementExamples({ seed: 999, trajectoryCount: 20 });

    const model = backend.train(train, { maxOrder: 3 });
    const result = evaluateMovementPolicy(model, heldOut, { k: 3 });

    // The synthetic process is mostly coherent, so a backoff model predicts the
    // canonical next movement on held-out data well above chance (1/7 ≈ 0.14).
    expect(result.top1Accuracy).toBeGreaterThan(0.6);
    expect(result.topKAccuracy).toBeGreaterThanOrEqual(result.top1Accuracy);
    // Held-out full contexts are largely unseen, so correct hits come from
    // generalization (backed-off contexts), not exact full-context recall.
    expect(result.generalizationRate).toBeGreaterThan(0);
    expect(result.meanMatchedOrder).toBeGreaterThan(0);
  });
});

describe("generateSyntheticMovementExamples", () => {
  it("is deterministic for a given seed", () => {
    const a = generateSyntheticMovementExamples({ seed: 7, trajectoryCount: 5 });
    const b = generateSyntheticMovementExamples({ seed: 7, trajectoryCount: 5 });
    expect(a).toEqual(b);
  });

  it("produces different data for different seeds", () => {
    const a = generateSyntheticMovementExamples({ seed: 1, trajectoryCount: 5 });
    const b = generateSyntheticMovementExamples({ seed: 2, trajectoryCount: 5 });
    expect(a).not.toEqual(b);
  });

  it("respects the requested count, length bounds, and vocabulary", () => {
    const vocabulary = ["m::1", "m::2", "m::3"];
    const examples = generateSyntheticMovementExamples({
      seed: 3,
      trajectoryCount: 6,
      vocabulary,
      minLength: 2,
      maxLength: 4,
    });
    expect(examples).toHaveLength(6);
    for (const item of examples) {
      expect(item.tokens.length).toBeGreaterThanOrEqual(2);
      expect(item.tokens.length).toBeLessThanOrEqual(4);
      for (const token of item.tokens) {
        expect(vocabulary).toContain(token);
      }
    }
  });
});
