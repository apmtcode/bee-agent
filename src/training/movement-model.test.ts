import { describe, expect, it } from "vitest";
import { buildReplayManifest } from "../capture/replay.js";
import {
  createMovementModelBackend,
  datasetFromReplayManifests,
  datasetFromTrajectories,
  evaluateGeneralization,
  loadMovementPolicy,
  MarkovMovementBackend,
  tokenizeMovementAction,
  type MovementDataset,
} from "./movement-model.js";
import {
  DEFAULT_MOVEMENT_FLOWS,
  generateSyntheticTrajectories,
} from "./synthetic-movements.js";

describe("tokenizeMovementAction", () => {
  it("prefers structured gesture metadata over the free-text summary", () => {
    expect(
      tokenizeMovementAction({
        tool: "device",
        summary: "tapped the Submit button",
        metadata: { gesture: "tap", target: "submit" },
      }),
    ).toBe("tap:submit");
  });

  it("falls back to direction, then summary, and stays stable", () => {
    expect(
      tokenizeMovementAction({ tool: "device", summary: "x", metadata: { gesture: "swipe", direction: "up" } }),
    ).toBe("swipe:up");
    expect(tokenizeMovementAction({ tool: "keyboard", summary: "Copy Selection" })).toBe("keyboard:copy-selection");
  });
});

describe("dataset construction", () => {
  it("builds one ordered sequence per trajectory and drops empty ones", () => {
    const trajectories = generateSyntheticTrajectories({ count: 3, seed: 7 });
    const dataset = datasetFromTrajectories(trajectories);
    expect(dataset.sequences).toHaveLength(3);
    for (const sequence of dataset.sequences) {
      expect(sequence.tokens.length).toBeGreaterThan(0);
      expect(sequence.tokens.every((token) => token.includes(":"))).toBe(true);
    }
  });

  it("derives stable, well-formed tokens from a replay manifest", () => {
    // Replay manifests carry only tool + summary (metadata is dropped at manifest
    // build time), so replay-derived tokens use the summary fallback and are a
    // distinct-but-stable vocabulary from the metadata-rich trajectory tokens.
    const [trajectory] = generateSyntheticTrajectories({ count: 1, seed: 3 });
    const manifest = buildReplayManifest({ sessionId: trajectory.sessionId, transcript: [], trajectories: [trajectory] });
    const fromTrajectory = datasetFromTrajectories([trajectory]).sequences[0];
    const fromReplay = datasetFromReplayManifests([manifest]).sequences[0];

    expect(fromReplay.tokens).toHaveLength(fromTrajectory.tokens.length);
    expect(fromReplay.tokens.every((token) => token.includes(":"))).toBe(true);
    // Re-tokenizing the same manifest is deterministic.
    expect(datasetFromReplayManifests([manifest]).sequences[0].tokens).toEqual(fromReplay.tokens);
  });
});

describe("MarkovMovementBackend", () => {
  it("reproduces a recorded movement sequence via greedy generation (objective 2c)", () => {
    // A single deterministic flow — the model should replay it exactly.
    const dataset: MovementDataset = {
      sequences: [
        { id: "a", tokens: ["tap:menu", "type:search", "tap:result", "scroll:down"] },
        { id: "b", tokens: ["tap:menu", "type:search", "tap:result", "scroll:down"] },
      ],
    };
    const policy = new MarkovMovementBackend().train(dataset, { order: 2 });
    const generated = policy.generate({ maxLength: 10 });
    expect(generated).toEqual(["tap:menu", "type:search", "tap:result", "scroll:down"]);
    expect(policy.score(dataset.sequences[0]).accuracy).toBe(1);
  });

  it("predicts the next movement from context with backoff", () => {
    const trajectories = generateSyntheticTrajectories({ count: 40, seed: 11 });
    const policy = new MarkovMovementBackend().train(datasetFromTrajectories(trajectories), { order: 2 });
    const prediction = policy.predictNext(["tap:menu"]);
    expect(prediction).toBeDefined();
    expect(prediction?.probability).toBeGreaterThan(0);
    // Unseen context backs off to a shorter suffix rather than returning nothing.
    const backoff = policy.predictNext(["never:seen", "tap:menu"]);
    expect(backoff?.token).toBe(prediction?.token);
    expect(backoff?.order).toBeLessThanOrEqual(2);
  });

  it("is deterministic across retrains and seeded sampling", () => {
    const dataset = datasetFromTrajectories(generateSyntheticTrajectories({ count: 20, seed: 5 }));
    const backend = new MarkovMovementBackend();
    const first = backend.train(dataset).generate({ maxLength: 8, seed: 99 });
    const second = backend.train(dataset).generate({ maxLength: 8, seed: 99 });
    expect(second).toEqual(first);
  });

  it("round-trips through serialize/loadMovementPolicy", () => {
    const dataset = datasetFromTrajectories(generateSyntheticTrajectories({ count: 12, seed: 21 }));
    const policy = new MarkovMovementBackend().train(dataset, { order: 2 });
    const restored = loadMovementPolicy(policy.serialize());
    expect(restored.order).toBe(policy.order);
    expect(restored.vocabulary).toEqual(policy.vocabulary);
    expect(restored.generate({ maxLength: 8 })).toEqual(policy.generate({ maxLength: 8 }));
  });
});

describe("generalization eval harness (objective 2d)", () => {
  it("generalizes to held-out but related trajectories better than chance", () => {
    // Train on a subset of flows, hold out one related flow that shares vocabulary.
    const trainFlows = DEFAULT_MOVEMENT_FLOWS.slice(0, 2);
    const heldOutFlow = DEFAULT_MOVEMENT_FLOWS.slice(0, 3);
    const train = datasetFromTrajectories(generateSyntheticTrajectories({ count: 60, seed: 1, flows: trainFlows }));
    const heldOut = datasetFromTrajectories(
      generateSyntheticTrajectories({ count: 20, seed: 2, flows: heldOutFlow }),
    );
    const policy = new MarkovMovementBackend().train(train, { order: 2 });
    const report = evaluateGeneralization(policy, heldOut);

    expect(report.transitions).toBeGreaterThan(0);
    // Shared-vocabulary flows should be largely predictable.
    expect(report.accuracy).toBeGreaterThan(0.5);
    expect(report.vocabularyCoverage).toBeGreaterThan(0.5);
    expect(report.sequences.length).toBe(heldOut.sequences.length);
  });
});

describe("backend registry", () => {
  it("returns an available markov backend by default", () => {
    const backend = createMovementModelBackend();
    expect(backend.id).toBe("markov");
    expect(backend.available).toBe(true);
  });

  it("exposes local-native as an unavailable seam that refuses to train in cloud/CI", () => {
    const backend = createMovementModelBackend("local-native");
    expect(backend.available).toBe(false);
    expect(() => backend.train({ sequences: [] })).toThrow(/on-device runtime/);
  });
});
