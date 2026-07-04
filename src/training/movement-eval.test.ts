import { describe, expect, it } from "vitest";
import {
  evaluateMovementPolicy,
  generateSyntheticMovementFamily,
  heldOutGeneralizationCases,
  stepSequenceFidelity,
} from "./movement-eval.js";
import { MockMovementPolicyBackend } from "./mock-policy-backend.js";

describe("generateSyntheticMovementFamily", () => {
  it("produces one deterministic trajectory per target", () => {
    const family = generateSyntheticMovementFamily({
      goalPrefix: "search notes for",
      appId: "notes",
      targets: ["invoice", "receipt", "contract"],
    });
    expect(family).toHaveLength(3);
    expect(family[0].goal).toBe("search notes for invoice");
    expect(family[0].steps.map((step) => step.gesture)).toEqual(["tap", "type"]);
    expect(family[0].steps[1].valueSummary).toBe("focus invoice");
    // regenerating yields identical ids (no RNG / clock)
    const again = generateSyntheticMovementFamily({
      goalPrefix: "search notes for",
      appId: "notes",
      targets: ["invoice", "receipt", "contract"],
    });
    expect(again.map((t) => t.id)).toEqual(family.map((t) => t.id));
  });
});

describe("stepSequenceFidelity", () => {
  it("scores ordered step matches", () => {
    const steps = [
      { gesture: "tap" as const, appId: "a", target: "x", ts: 0 },
      { gesture: "type" as const, appId: "a", target: "x", valueSummary: "v", ts: 1 },
    ];
    expect(stepSequenceFidelity(steps, steps)).toBe(1);
    expect(stepSequenceFidelity([steps[0]], steps)).toBe(0.5);
    expect(stepSequenceFidelity([], [])).toBe(1);
    expect(stepSequenceFidelity([steps[0]], [])).toBe(0);
  });
});

describe("held-out generalization eval", () => {
  it("reproduces held-out family members by re-parameterizing", () => {
    const family = generateSyntheticMovementFamily({
      goalPrefix: "search notes for",
      appId: "notes",
      targets: ["invoice", "receipt", "contract", "memo", "report"],
    });
    const { train, cases } = heldOutGeneralizationCases(family, { holdOutEvery: 2 });

    expect(train.length).toBeGreaterThan(0);
    expect(cases.length).toBeGreaterThan(0);
    // held-out members must NOT be in the training set
    const trainIds = new Set(train.map((t) => t.id));
    for (const evalCase of cases) {
      const heldOutValue = evalCase.expected[1]?.valueSummary;
      expect(train.some((t) => t.steps[1]?.valueSummary === heldOutValue)).toBe(false);
    }

    const model = new MockMovementPolicyBackend().fit(train);
    const report = evaluateMovementPolicy(model, cases);

    // A policy fit only on siblings should reproduce held-out movements exactly
    // by substituting the varying value — the generalization requirement.
    expect(report.meanFidelity).toBe(1);
    expect(report.exactMatchRate).toBe(1);
    expect(report.results.every((r) => r.generalized)).toBe(true);
    expect(trainIds.size).toBe(train.length);
  });

  it("reports degraded fidelity when the policy cannot generalize a field", () => {
    const family = generateSyntheticMovementFamily({
      goalPrefix: "scroll feed in",
      appId: "feed",
      targets: ["home", "trending"],
      gestures: ["scroll"],
    });
    // Ask for the held-out member but WITHOUT supplying the varying target,
    // so the nearest match cannot be re-parameterized to it.
    const model = new MockMovementPolicyBackend().fit([family[1]]);
    const report = evaluateMovementPolicy(model, [
      { context: { goal: family[0].goal, appId: "feed" }, expected: family[0].steps },
    ]);
    expect(report.exactMatchRate).toBe(0);
    expect(report.meanFidelity).toBeLessThan(1);
  });
});
