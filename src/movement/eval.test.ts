import { describe, expect, it } from "vitest";
import {
  defaultMovementTemplates,
  evaluateGeneralization,
  generateSyntheticMovementDataset,
} from "./eval.js";
import { NgramMovementBackend } from "./ngram-backend.js";

describe("synthetic movement generator", () => {
  it("is deterministic for a fixed seed", () => {
    const templates = defaultMovementTemplates();
    const a = generateSyntheticMovementDataset({ templates, instancesPerTemplate: 5, seed: 7 });
    const b = generateSyntheticMovementDataset({ templates, instancesPerTemplate: 5, seed: 7 });
    expect(JSON.stringify(a)).toEqual(JSON.stringify(b));
  });

  it("produces the requested number of sequences with monotonic timestamps", () => {
    const templates = defaultMovementTemplates();
    const dataset = generateSyntheticMovementDataset({ templates, instancesPerTemplate: 4, seed: 3 });
    expect(dataset.sequences).toHaveLength(templates.length * 4);
    for (const sequence of dataset.sequences) {
      expect(sequence.events.length).toBeGreaterThan(0);
      for (let i = 1; i < sequence.events.length; i += 1) {
        expect(sequence.events[i]!.ts).toBeGreaterThan(sequence.events[i - 1]!.ts);
      }
    }
  });

  it("introduces variation across instances (not all identical)", () => {
    const templates = defaultMovementTemplates();
    const dataset = generateSyntheticMovementDataset({ templates, instancesPerTemplate: 12, seed: 11 });
    const searchInstances = dataset.sequences.filter((s) => s.intent === "search and open");
    const distinct = new Set(searchInstances.map((s) => s.events.map((e) => e.target ?? "").join("|")));
    expect(distinct.size).toBeGreaterThan(1);
  });
});

describe("generalization eval harness", () => {
  it("achieves strong next-token accuracy on held-out synthetic trajectories", async () => {
    const templates = defaultMovementTemplates();
    const dataset = generateSyntheticMovementDataset({ templates, instancesPerTemplate: 40, seed: 5 });
    const report = await evaluateGeneralization({
      backend: new NgramMovementBackend(),
      dataset,
      holdoutRatio: 0.3,
      seed: 99,
      order: 2,
      smoothing: 0.01,
    });

    expect(report.trainSequences).toBeGreaterThan(0);
    expect(report.heldOutSequences).toBeGreaterThan(0);
    expect(report.evaluatedTokens).toBeGreaterThan(0);
    // Templates are structured, so a 2-gram model should predict most tokens.
    expect(report.nextTokenAccuracy).toBeGreaterThan(0.6);
    expect(report.transitionCoverage).toBeGreaterThan(0.9);
  });

  it("keeps at least one training sequence even at extreme holdout ratios", async () => {
    const templates = defaultMovementTemplates();
    const dataset = generateSyntheticMovementDataset({ templates, instancesPerTemplate: 2, seed: 1 });
    const report = await evaluateGeneralization({
      backend: new NgramMovementBackend(),
      dataset,
      holdoutRatio: 0.99,
      seed: 2,
    });
    expect(report.trainSequences).toBeGreaterThanOrEqual(1);
    expect(report.heldOutSequences).toBeGreaterThanOrEqual(1);
  });
});
