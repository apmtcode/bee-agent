import { describe, expect, it } from "vitest";
import { buildMovementDataset, MarkovMovementBackend } from "./movement-policy.js";
import {
  createSeededRandom,
  evaluateMovementPolicy,
  generateSyntheticTrajectories,
  lcsSimilarity,
  type MovementWorkflowTemplate,
} from "./movement-eval.js";

const TEMPLATES: MovementWorkflowTemplate[] = [
  { name: "deploy", steps: ["open", "select-env", "review", "confirm", "watch"] },
  { name: "triage", steps: ["open", "filter", "assign", "comment", "close"] },
];

describe("createSeededRandom", () => {
  it("is deterministic for a given seed", () => {
    const a = createSeededRandom(42);
    const b = createSeededRandom(42);
    const seqA = [a(), a(), a()];
    const seqB = [b(), b(), b()];
    expect(seqA).toEqual(seqB);
    expect(seqA.every((value) => value >= 0 && value < 1)).toBe(true);
  });
});

describe("lcsSimilarity", () => {
  it("scores identical sequences at 1 and disjoint at 0", () => {
    expect(lcsSimilarity(["a", "b", "c"], ["a", "b", "c"])).toBe(1);
    expect(lcsSimilarity(["a", "b"], ["x", "y"])).toBe(0);
    expect(lcsSimilarity([], [])).toBe(1);
  });

  it("captures partial subsequence overlap", () => {
    // LCS of [a,b,c,d] and [a,c,d] is [a,c,d] length 3 over max length 4.
    expect(lcsSimilarity(["a", "b", "c", "d"], ["a", "c", "d"])).toBeCloseTo(0.75);
  });
});

describe("generateSyntheticTrajectories", () => {
  it("emits reproducible related trajectories per template", () => {
    const first = generateSyntheticTrajectories({ templates: TEMPLATES, perTemplate: 3, seed: 7 });
    const second = generateSyntheticTrajectories({ templates: TEMPLATES, perTemplate: 3, seed: 7 });

    expect(first).toHaveLength(6);
    expect(first.map((t) => t.id)).toEqual(second.map((t) => t.id));
    // Same seed -> identical action sequences.
    expect(first.map((t) => t.actions.map((a) => a.tool))).toEqual(
      second.map((t) => t.actions.map((a) => a.tool)),
    );
    // Actions are timestamp-ordered and non-empty.
    for (const traj of first) {
      expect(traj.actions.length).toBeGreaterThan(0);
      const timestamps = traj.actions.map((a) => a.ts);
      expect([...timestamps].sort((x, y) => x - y)).toEqual(timestamps);
    }
  });

  it("varies sequences across seeds", () => {
    const a = generateSyntheticTrajectories({ templates: TEMPLATES, perTemplate: 4, seed: 1 });
    const b = generateSyntheticTrajectories({ templates: TEMPLATES, perTemplate: 4, seed: 99 });
    const toolsA = a.map((t) => t.actions.map((x) => x.tool).join(","));
    const toolsB = b.map((t) => t.actions.map((x) => x.tool).join(","));
    expect(toolsA).not.toEqual(toolsB);
  });
});

describe("evaluateMovementPolicy", () => {
  it("scores a policy near-perfectly on the data it memorized", async () => {
    const trajectories = generateSyntheticTrajectories({
      templates: TEMPLATES,
      perTemplate: 6,
      seed: 3,
      dropRate: 0,
      repeatRate: 0,
    });
    const dataset = buildMovementDataset(trajectories);
    const policy = await new MarkovMovementBackend(3).train(dataset);

    // Seed length 2 disambiguates the two templates (both begin "open"), so
    // rollouts reproduce the recorded movement rather than a sibling workflow.
    const evaluation = evaluateMovementPolicy(policy, trajectories, { seedLength: 2 });
    expect(evaluation.trajectoryCount).toBe(trajectories.length);
    expect(evaluation.nextTokenAccuracy).toBeGreaterThan(0.9);
    expect(evaluation.meanSequenceSimilarity).toBeGreaterThan(0.9);
  });

  it("generalizes to held-out related trajectories better than chance", async () => {
    const all = generateSyntheticTrajectories({
      templates: TEMPLATES,
      perTemplate: 10,
      seed: 5,
      dropRate: 0.2,
      repeatRate: 0.2,
    });
    // Split: train on first 7 instances of each template, hold out the rest.
    const train = all.filter((t) => Number(t.id.split("-").pop()) < 7);
    const heldOut = all.filter((t) => Number(t.id.split("-").pop()) >= 7);

    const dataset = buildMovementDataset(train);
    const policy = await new MarkovMovementBackend(3).train(dataset);

    const evaluation = evaluateMovementPolicy(policy, heldOut, { topK: 3 });
    expect(evaluation.trajectoryCount).toBe(heldOut.length);
    // Held-out instances share template structure, so backoff generalizes well.
    expect(evaluation.topKAccuracy).toBeGreaterThan(0.6);
    expect(evaluation.meanSequenceSimilarity).toBeGreaterThan(0.5);
  });

  it("returns zeroed metrics for empty held-out input", async () => {
    const dataset = buildMovementDataset(
      generateSyntheticTrajectories({ templates: TEMPLATES, perTemplate: 2 }),
    );
    const policy = await new MarkovMovementBackend().train(dataset);
    const evaluation = evaluateMovementPolicy(policy, []);
    expect(evaluation).toMatchObject({
      trajectoryCount: 0,
      predictionCount: 0,
      nextTokenAccuracy: 0,
      meanSequenceSimilarity: 0,
    });
  });
});
