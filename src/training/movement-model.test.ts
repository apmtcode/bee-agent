import { describe, expect, it } from "vitest";
import {
  buildMovementDataset,
  evaluateMovementModel,
  MarkovMovementBackend,
  MovementModelRegistry,
  MOVEMENT_END,
  toolWithSummaryTokenizer,
  type MovementDataset,
} from "./movement-model.js";
import {
  defaultWorkflowTemplates,
  generateSyntheticTrajectories,
} from "../capture/synthetic.js";

function trainOn(dataset: MovementDataset, order: number) {
  const backend = new MarkovMovementBackend();
  const snapshot = backend.train(dataset, { order });
  return backend.load(snapshot);
}

describe("buildMovementDataset", () => {
  it("tokenizes actions in timestamp order and drops empty trajectories", () => {
    const trajectories = generateSyntheticTrajectories({ count: 3, seed: 7 });
    const dataset = buildMovementDataset(trajectories);
    expect(dataset.samples).toHaveLength(3);
    for (const sample of dataset.samples) {
      expect(sample.tokens.length).toBeGreaterThan(0);
    }
    // vocabulary is deduped + sorted
    expect(dataset.vocabulary).toEqual([...new Set(dataset.vocabulary)].sort());
  });

  it("supports a richer tokenizer that folds the summary", () => {
    const trajectories = generateSyntheticTrajectories({ count: 1, seed: 1 });
    const plain = buildMovementDataset(trajectories);
    const rich = buildMovementDataset(trajectories, { tokenizer: toolWithSummaryTokenizer });
    expect(rich.vocabulary.length).toBeGreaterThanOrEqual(plain.vocabulary.length);
    expect(rich.samples[0]!.tokens[0]).toContain(":");
  });
});

describe("MarkovMovementBackend — repeat fidelity", () => {
  it("reproduces recorded movements exactly when contexts are unambiguous", () => {
    // A single template: every context (including the START padding) maps to a
    // unique successor, so a bounded-order Markov achieves perfect recall.
    const templates = [{ name: "alpha", tools: ["a1", "a2", "a3", "a4", "a5"] }];
    const trajectories = generateSyntheticTrajectories({ count: 12, seed: 3, templates });
    const dataset = buildMovementDataset(trajectories);
    const model = trainOn(dataset, 2);
    const evalResult = evaluateMovementModel(model, dataset.samples);
    expect(evalResult.accuracy).toBe(1);
    expect(evalResult.unknownRate).toBe(0);
  });

  it("recalls the majority of movements even on overlapping real-world templates", () => {
    // The default templates share prefixes (focus-window > click-menu > …), so a
    // bounded-order model cannot be perfect — but recall stays high.
    const trajectories = generateSyntheticTrajectories({ count: 12, seed: 3 });
    const dataset = buildMovementDataset(trajectories);
    const model = trainOn(dataset, 3);
    const evalResult = evaluateMovementModel(model, dataset.samples);
    expect(evalResult.accuracy).toBeGreaterThan(0.85);
    expect(evalResult.unknownRate).toBe(0);
  });

  it("regenerates a recorded sequence from an empty prefix", () => {
    const templates = [defaultWorkflowTemplates()[0]!];
    const trajectories = generateSyntheticTrajectories({ count: 4, seed: 2, templates });
    const dataset = buildMovementDataset(trajectories);
    const model = trainOn(dataset, 3);
    const generated = model.generate([]);
    expect(generated).toEqual(templates[0]!.tools);
  });

  it("round-trips through a serialized snapshot", () => {
    const trajectories = generateSyntheticTrajectories({ count: 5, seed: 9 });
    const dataset = buildMovementDataset(trajectories);
    const backend = new MarkovMovementBackend();
    const snapshot = backend.train(dataset, { order: 2 });
    const reloaded = backend.load(JSON.parse(JSON.stringify(snapshot)));
    const before = trainOn(dataset, 2).predict(["focus-window"]);
    const after = reloaded.predict(["focus-window"]);
    expect(after.token).toBe(before.token);
    expect(reloaded.serialize().contexts.length).toBe(snapshot.contexts.length);
  });
});

describe("MarkovMovementBackend — generalization", () => {
  it("predicts sensible movements on held-out but related trajectories via back-off", () => {
    const train = generateSyntheticTrajectories({ count: 40, seed: 4 });
    // Held-out set: same templates but noisy variants the model never saw.
    const heldOut = generateSyntheticTrajectories({ count: 20, seed: 999, noise: 0.4 });
    const dataset = buildMovementDataset(train);
    const model = trainOn(dataset, 2);

    const evalResult = evaluateMovementModel(model, buildMovementDataset(heldOut).samples);
    // Well above the uniform baseline (1/vocab), thanks to context back-off.
    const uniformBaseline = 1 / dataset.vocabulary.length;
    expect(evalResult.accuracy).toBeGreaterThan(uniformBaseline * 3);
    // Some predictions required back-off (novel contexts), proving generalization.
    expect(evalResult.backoffRate).toBeGreaterThan(0);
    // But it never falls through to a pure uniform guess on related data.
    expect(evalResult.unknownRate).toBe(0);
  });

  it("terminates generation with END and never emits the end sentinel", () => {
    const trajectories = generateSyntheticTrajectories({ count: 6, seed: 5 });
    const dataset = buildMovementDataset(trajectories);
    const model = trainOn(dataset, 2);
    const generated = model.generate(["focus-window"], 100);
    expect(generated).not.toContain(MOVEMENT_END);
    expect(generated.length).toBeLessThan(100);
  });
});

describe("MovementModelRegistry", () => {
  it("exposes the markov backend by default and throws on unknown ids", () => {
    const registry = new MovementModelRegistry();
    expect(registry.list()).toContain("markov");
    expect(registry.get("markov").id).toBe("markov");
    expect(() => registry.get("nope")).toThrow(/unknown movement-model backend/);
  });

  it("accepts a custom pluggable backend", () => {
    const registry = new MovementModelRegistry();
    registry.register({
      id: "stub",
      train: () => ({ version: 1, backendId: "stub", order: 1, vocabulary: [], contexts: [] }),
      load: () => ({
        backendId: "stub",
        order: 1,
        predict: () => ({ token: "x", probability: 1, contextLength: 0, backedOff: false, unknown: false }),
        generate: () => [],
        serialize: () => ({ version: 1, backendId: "stub", order: 1, vocabulary: [], contexts: [] }),
      }),
    });
    expect(registry.list()).toEqual(["markov", "stub"]);
  });
});

describe("evaluateMovementModel", () => {
  it("returns zeroed metrics for an empty sample set", () => {
    const model = trainOn(buildMovementDataset(generateSyntheticTrajectories({ count: 1, seed: 1 })), 2);
    const result = evaluateMovementModel(model, []);
    expect(result).toEqual({ total: 0, correct: 0, accuracy: 0, backoffRate: 0, unknownRate: 0 });
  });
});
