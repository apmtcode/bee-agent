import { describe, expect, it } from "vitest";
import {
  DEFAULT_SYNTHETIC_WORKFLOWS,
  generateSyntheticTrajectories,
  variantWorkflow,
} from "../capture/synthetic.js";
import { buildReplayManifest } from "../capture/replay.js";
import {
  MOVEMENT_END,
  MOVEMENT_START,
  MarkovMovementBackend,
  createDefaultMovementModelRegistry,
  encodeActionToken,
  evaluateMovementPolicy,
  extractMovementSamplesFromReplays,
  extractMovementSamplesFromTrajectories,
  type MovementSample,
} from "./movement-model.js";

const backend = new MarkovMovementBackend();

const composeWorkflow = DEFAULT_SYNTHETIC_WORKFLOWS[0]!;

function samplesFromWorkflowRuns(repeats: number): MovementSample[] {
  const trajectories = generateSyntheticTrajectories({
    sessionId: "s1",
    workflow: composeWorkflow,
    repeats,
  });
  return extractMovementSamplesFromTrajectories(trajectories);
}

describe("MarkovMovementBackend", () => {
  it("repeats a recorded movement sequence exactly", async () => {
    const samples = samplesFromWorkflowRuns(1);
    const policy = await backend.train(samples, { order: 4 });

    const recorded = samples[0]!.tokens;
    // Free-run from the first token → must reproduce the full recorded run.
    const generated = policy.generate([recorded[0]!], recorded.length + 4);
    expect(generated).toEqual(recorded);
  });

  it("predicts the next movement with calibrated confidence", async () => {
    const samples = samplesFromWorkflowRuns(3);
    const policy = await backend.train(samples, { order: 3 });

    const prediction = policy.predictNext([MOVEMENT_START]);
    expect(prediction?.token).toBe(encodeActionToken("device", "tapped compose"));
    // Every run starts identically, so START → first-step is certain.
    expect(prediction?.confidence).toBeCloseTo(1, 5);
  });

  it("emits END after the last recorded action", async () => {
    const samples = samplesFromWorkflowRuns(2);
    const policy = await backend.train(samples, { order: 3 });
    const last = samples[0]!.tokens.at(-1)!;
    const prediction = policy.predictNext([last]);
    expect(prediction?.token).toBe(MOVEMENT_END);
  });

  it("is deterministic — identical input yields identical serialized policy", async () => {
    const samples = samplesFromWorkflowRuns(2);
    const a = await backend.train(samples, { order: 3 });
    const b = await backend.train(samples, { order: 3 });
    expect(a.toJSON()).toEqual(b.toJSON());
  });

  it("returns undefined from an empty model", async () => {
    const policy = await backend.train([], { order: 2 });
    expect(policy.predictNext([MOVEMENT_START])).toBeUndefined();
    expect(policy.generate([], 5)).toEqual([]);
    expect(policy.metrics.vocabularySize).toBe(0);
  });

  it("reports metrics over the training set", async () => {
    const samples = samplesFromWorkflowRuns(2);
    const policy = await backend.train(samples, { order: 3 });
    expect(policy.metrics.sampleCount).toBe(2);
    expect(policy.metrics.backendId).toBe("markov");
    expect(policy.metrics.tokenCount).toBe(composeWorkflow.steps.length * 2);
    expect(policy.metrics.vocabularySize).toBeGreaterThan(0);
  });
});

describe("generalization", () => {
  it("generalizes shared sub-movements to a related workflow via backoff", async () => {
    // Train on the base workflow, evaluate on a variant that drops one middle step.
    const trainSamples = samplesFromWorkflowRuns(3);
    const policy = await backend.train(trainSamples, { order: 2 });

    const variant = variantWorkflow(composeWorkflow, { kind: "drop", at: 2 });
    const heldOut = extractMovementSamplesFromTrajectories(
      generateSyntheticTrajectories({ sessionId: "s2", workflow: variant, repeats: 1, idPrefix: "held" }),
    );

    const evaluation = evaluateMovementPolicy(policy, heldOut, 1);
    expect(evaluation.sampleCount).toBe(1);
    // The unchanged prefix/suffix transitions are known → strong (not perfect) accuracy.
    expect(evaluation.nextTokenAccuracy).toBeGreaterThan(0.5);
  });

  it("achieves perfect teacher-forced accuracy on the training distribution", async () => {
    const samples = samplesFromWorkflowRuns(3);
    const policy = await backend.train(samples, { order: 3 });
    const evaluation = evaluateMovementPolicy(policy, samples, 1);
    expect(evaluation.nextTokenAccuracy).toBeCloseTo(1, 5);
    expect(evaluation.exactReplayRate).toBeCloseTo(1, 5);
  });
});

describe("dataset extraction", () => {
  it("builds samples from replay manifests (action events only, time-ordered)", async () => {
    const trajectories = generateSyntheticTrajectories({
      sessionId: "s3",
      workflow: composeWorkflow,
      repeats: 1,
    });
    const replay = buildReplayManifest({ sessionId: "s3", transcript: [], trajectories });
    const samples = extractMovementSamplesFromReplays([replay]);
    expect(samples).toHaveLength(1);
    expect(samples[0]!.tokens).toEqual(
      composeWorkflow.steps.map((step) => encodeActionToken(step.tool, step.summary)),
    );
  });

  it("normalizes action tokens (case/whitespace)", () => {
    expect(encodeActionToken("  Device ", "Tapped   Send ")).toBe("device:tapped send");
  });
});

describe("MovementModelRegistry", () => {
  it("exposes the markov backend by default and is pluggable", async () => {
    const registry = createDefaultMovementModelRegistry();
    expect(registry.list()).toContain("markov");
    expect(registry.has("markov")).toBe(true);

    const custom = { id: "noop", train: async () => (await backend.train([], {})) };
    registry.register(custom);
    expect(registry.get("noop")).toBe(custom);
    expect(() => registry.get("missing")).toThrow(/unknown movement-model backend/);
  });
});
