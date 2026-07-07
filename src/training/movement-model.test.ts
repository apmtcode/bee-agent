import { describe, expect, it } from "vitest";
import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  ACTION_TOKEN_PREFIX,
  DeterministicNGramBackend,
  buildMovementDataset,
  buildMovementSequence,
  evaluateMovementPolicy,
  generateSyntheticMovementTrajectories,
  type MovementModelArtifact,
} from "./movement-model.js";

function trajectory(id: string, beats: Array<[string, string]>): TrajectorySpan {
  let ts = 0;
  const observations = [];
  const actions = [];
  for (const [source, tool] of beats) {
    observations.push({ kind: "observation" as const, source, summary: `see ${source}`, ts: ts++ });
    actions.push({ kind: "action" as const, tool, summary: `do ${tool}`, ts: ts++ });
  }
  return {
    id,
    sessionId: "s1",
    createdAt: new Date(0).toISOString(),
    captureTier: "full",
    observations,
    actions,
  };
}

describe("buildMovementSequence", () => {
  it("interleaves observations and actions in timestamp order, obs before act on ties", () => {
    const span = trajectory("t1", [
      ["screen", "mouse.click"],
      ["editor", "key.type"],
    ]);
    const sequence = buildMovementSequence(span);
    expect(sequence.steps.map((s) => s.token)).toEqual([
      "obs:screen",
      "act:mouse.click",
      "obs:editor",
      "act:key.type",
    ]);
  });

  it("uses the reviewed/redacted view when present", () => {
    const span = trajectory("t2", [["screen", "secret.tool"]]);
    span.review = {
      status: "approved",
      reviewedAt: new Date(0).toISOString(),
      reviewedBy: "reviewer",
      redactedObservations: [{ ts: 0, source: "screen", summary: "see screen" }],
      redactedActions: [{ ts: 1, tool: "mouse.click", summary: "redacted" }],
    };
    const sequence = buildMovementSequence(span);
    expect(sequence.steps.map((s) => s.token)).toEqual(["obs:screen", "act:mouse.click"]);
  });
});

describe("buildMovementDataset", () => {
  it("emits one example per action with the preceding context and a sorted vocab", () => {
    const dataset = buildMovementDataset([trajectory("t1", [["screen", "mouse.click"], ["editor", "key.type"]])]);
    expect(dataset.examples).toHaveLength(2);
    expect(dataset.examples[0]).toMatchObject({ context: ["obs:screen"], action: "act:mouse.click" });
    expect(dataset.examples[1]).toMatchObject({
      context: ["obs:screen", "act:mouse.click", "obs:editor"],
      action: "act:key.type",
    });
    expect(dataset.vocabulary.actions).toEqual(["act:key.type", "act:mouse.click"]);
    expect(dataset.vocabulary.observations).toEqual(["obs:editor", "obs:screen"]);
  });
});

describe("DeterministicNGramBackend — memorization (objective #2c: repeat recorded movements)", () => {
  it("replays a recorded trajectory with 100% accuracy from full-order memory", () => {
    const span = trajectory("t1", [
      ["screen", "window.focus"],
      ["menu", "mouse.click"],
      ["editor", "key.type"],
      ["editor", "key.press:enter"],
    ]);
    const dataset = buildMovementDataset([span]);
    const policy = new DeterministicNGramBackend().train(dataset, { order: 3 });

    const result = evaluateMovementPolicy(policy, [span]);
    expect(result.accuracy).toBe(1);
    expect(result.exactMemoryHits).toBe(result.totalPredictions);
    expect(result.generalizedHits).toBe(0);
  });

  it("is deterministic across retrains (same dataset -> identical artifact)", () => {
    const dataset = buildMovementDataset(generateSyntheticMovementTrajectories({ count: 20, seed: 7 }));
    const a = new DeterministicNGramBackend().train(dataset).serialize();
    const b = new DeterministicNGramBackend().train(dataset).serialize();
    expect(a).toEqual(b);
  });
});

describe("DeterministicNGramBackend — generalization (objective #2d: new but related movements)", () => {
  it("predicts plausible next actions on held-out trajectories via backoff", () => {
    const backend = new DeterministicNGramBackend();
    const train = generateSyntheticMovementTrajectories({ count: 60, seed: 1, variability: 0.15 });
    const test = generateSyntheticMovementTrajectories({ count: 20, seed: 999, variability: 0.15 });
    const policy = backend.train(buildMovementDataset(train), { order: 3 });

    const result = evaluateMovementPolicy(policy, test);
    // Held-out (different seed) trajectories share local structure with training,
    // so a backoff n-gram should generalize well above the ~1/uniform baseline.
    expect(result.totalPredictions).toBeGreaterThan(0);
    expect(result.accuracy).toBeGreaterThan(0.6);
    // Meaningful generalization: some correct predictions come from backoff,
    // not just exact full-order memory.
    expect(result.generalizedHits).toBeGreaterThan(0);
    for (const token of policy.serialize().tables.flatMap((t) => Object.values(t).flatMap((c) => Object.keys(c)))) {
      expect(token.startsWith(ACTION_TOKEN_PREFIX)).toBe(true);
    }
  });

  it("falls back to the marginal action for a wholly unseen context", () => {
    const dataset = buildMovementDataset([
      trajectory("t1", [["screen", "mouse.click"], ["screen", "mouse.click"], ["editor", "key.type"]]),
    ]);
    const policy = new DeterministicNGramBackend().train(dataset, { order: 2 });
    const prediction = policy.predict(["obs:never-seen-context"]);
    // "act:mouse.click" is the most frequent action -> the marginal fallback.
    expect(prediction.action).toBe("act:mouse.click");
    expect(prediction.backoffOrder).toBe(0);
    expect(prediction.fromMemory).toBe(false);
  });

  it("returns an empty prediction when trained on an empty dataset", () => {
    const policy = new DeterministicNGramBackend().train(buildMovementDataset([]), { order: 2 });
    const prediction = policy.predict(["obs:anything"]);
    expect(prediction.action).toBe("");
    expect(prediction.backoffOrder).toBe(-1);
  });
});

describe("MovementModelArtifact round-trip (pluggable/serializable backend)", () => {
  it("restore(serialize()) reproduces identical predictions", () => {
    const backend = new DeterministicNGramBackend();
    const train = generateSyntheticMovementTrajectories({ count: 30, seed: 42, variability: 0.1 });
    const trained = backend.train(buildMovementDataset(train), { order: 3 });
    const artifact: MovementModelArtifact = trained.serialize();

    const restored = backend.restore(artifact);
    const contexts = [["obs:screen"], ["obs:browser", "act:window.focus"], ["obs:files", "act:mouse.move"], []];
    for (const context of contexts) {
      expect(restored.predict(context)).toEqual(trained.predict(context));
    }
    expect(restored.serialize()).toEqual(artifact);
  });
});

describe("generateSyntheticMovementTrajectories", () => {
  it("is deterministic for a given seed and reflects variability", () => {
    const a = generateSyntheticMovementTrajectories({ count: 5, seed: 3 });
    const b = generateSyntheticMovementTrajectories({ count: 5, seed: 3 });
    expect(a).toEqual(b);

    const different = generateSyntheticMovementTrajectories({ count: 5, seed: 4 });
    expect(different).not.toEqual(a);

    const perturbed = generateSyntheticMovementTrajectories({ count: 40, seed: 3, variability: 0.5 });
    // With variability some trajectories drop/duplicate beats -> variable action counts.
    const counts = new Set(perturbed.map((t) => t.actions.length));
    expect(counts.size).toBeGreaterThan(1);
  });
});
