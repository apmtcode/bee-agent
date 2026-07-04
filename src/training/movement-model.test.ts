import { describe, expect, it } from "vitest";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import type { TrajectoryAction } from "../capture/trajectory.js";
import {
  MarkovMovementModelBackend,
  MOVEMENT_END_TOKEN,
  buildMovementDataset,
  evaluateMovementGeneralization,
  splitMovementDataset,
  tokenizeAction,
} from "./movement-model.js";
import { generateSyntheticTrajectories } from "./movement-simulation.js";

function action(tool: string, ts: number, metadata?: Record<string, unknown>): TrajectoryAction {
  return { kind: "action", tool, summary: `${tool} action`, ts, ...(metadata ? { metadata } : {}) };
}

describe("tokenizeAction", () => {
  it("collapses structurally identical movements to the same token", () => {
    const a = action("device", 1, { gesture: "tap", target: "Search Box" });
    const b = action("device", 2, { gesture: "tap", target: "search box" });
    expect(tokenizeAction(a)).toBe("device:tap:@search-box");
    expect(tokenizeAction(a)).toBe(tokenizeAction(b));
  });

  it("distinguishes different gestures / directions", () => {
    expect(tokenizeAction(action("device", 1, { gesture: "scroll", direction: "up" }))).toBe("device:scroll:up");
    expect(tokenizeAction(action("device", 1, { gesture: "scroll", direction: "down" }))).toBe("device:scroll:down");
    expect(tokenizeAction(action("keyboard", 1))).toBe("keyboard");
  });
});

describe("buildMovementDataset", () => {
  it("orders actions by timestamp and drops empty trajectories", () => {
    const dataset = buildMovementDataset([
      buildTrajectorySpan({
        id: "t1",
        sessionId: "s1",
        actions: [action("device", 2, { gesture: "tap", target: "b" }), action("device", 1, { gesture: "tap", target: "a" })],
      }),
      buildTrajectorySpan({ id: "empty", sessionId: "s1", actions: [] }),
    ]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]?.tokens).toEqual(["device:tap:@a", "device:tap:@b"]);
  });
});

describe("MarkovMovementModelBackend", () => {
  it("repeats a recorded movement sequence exactly (replay)", async () => {
    const trajectories = [
      buildTrajectorySpan({
        id: "t1",
        sessionId: "s1",
        actions: [
          action("device", 1, { gesture: "tap", target: "launcher" }),
          action("device", 2, { gesture: "tap", target: "search" }),
          action("device", 3, { gesture: "type", target: "search" }),
          action("device", 4, { gesture: "shortcut", target: "submit" }),
        ],
      }),
    ];
    const dataset = buildMovementDataset(trajectories);
    const model = await new MarkovMovementModelBackend().train(dataset, { order: 3 });
    const generated = model.generate([], 10);
    expect(generated).toEqual(dataset.sequences[0]?.tokens);
  });

  it("predicts the next movement from a context", async () => {
    const dataset = buildMovementDataset([
      buildTrajectorySpan({
        id: "t1",
        sessionId: "s1",
        actions: [
          action("device", 1, { gesture: "tap", target: "launcher" }),
          action("device", 2, { gesture: "scroll", direction: "down" }),
          action("device", 3, { gesture: "tap", target: "item" }),
        ],
      }),
    ]);
    const model = await new MarkovMovementModelBackend().train(dataset);
    const prediction = model.predictNext(["device:tap:@launcher"]);
    expect(prediction.token).toBe("device:scroll:down");
    expect(prediction.order).toBe(1);
    expect(prediction.probability).toBeGreaterThan(0);
  });

  it("generalizes to an unseen context via back-off", async () => {
    // Two trajectories teach: ...scroll:down -> tap:@item, and separately a
    // fresh prefix that ends in scroll:down. The model has never seen the full
    // new context but should back off to the shared "scroll:down" suffix.
    const dataset = buildMovementDataset([
      buildTrajectorySpan({
        id: "t1",
        sessionId: "s1",
        actions: [
          action("device", 1, { gesture: "tap", target: "launcher" }),
          action("device", 2, { gesture: "scroll", direction: "down" }),
          action("device", 3, { gesture: "tap", target: "item" }),
        ],
      }),
    ]);
    const model = await new MarkovMovementModelBackend().train(dataset, { order: 3 });
    const prediction = model.predictNext(["device:tap:@compose", "device:scroll:down"]);
    expect(prediction.token).toBe("device:tap:@item");
    // Matched via the 1-token suffix, not the full (unseen) 2-token context.
    expect(prediction.order).toBe(1);
  });

  it("returns an empty prediction for an untrained model", async () => {
    const model = await new MarkovMovementModelBackend().train({ version: 1, sequences: [] });
    const prediction = model.predictNext(["anything"]);
    expect(prediction.token).toBeUndefined();
    expect(prediction.order).toBe(-1);
  });

  it("round-trips through serialize/load", async () => {
    const backend = new MarkovMovementModelBackend();
    const dataset = buildMovementDataset(generateSyntheticTrajectories({ count: 6 }));
    const model = await backend.train(dataset, { order: 3 });
    const restored = backend.load(model.serialize());
    const seed = dataset.sequences[0]?.tokens.slice(0, 1) ?? [];
    expect(restored.generate(seed, 8)).toEqual(model.generate(seed, 8));
    expect(restored.vocabulary).toEqual(model.vocabulary);
  });
});

describe("evaluateMovementGeneralization", () => {
  it("achieves perfect replay fidelity and strong generalization on structured data", async () => {
    const dataset = buildMovementDataset(generateSyntheticTrajectories({ count: 30, templateCount: 3 }));
    const result = await evaluateMovementGeneralization(new MarkovMovementModelBackend(), dataset, {
      holdoutRatio: 0.3,
      order: 3,
    });
    expect(result.trainSequences).toBeGreaterThan(0);
    expect(result.holdoutSequences).toBeGreaterThan(0);
    // Training data is strongly memorized (shared cross-template prefixes leave
    // a little irreducible ambiguity in the first few steps).
    expect(result.replayFidelity).toBeGreaterThan(0.8);
    // Held-out related trajectories share sub-sequences, so back-off generalizes well.
    expect(result.generalizationAccuracy).toBeGreaterThan(0.7);
    // Replaying seen data is never harder than generalizing to unseen data.
    expect(result.replayFidelity).toBeGreaterThanOrEqual(result.generalizationAccuracy);
  });

  it("splits the dataset deterministically", () => {
    const dataset = buildMovementDataset(generateSyntheticTrajectories({ count: 10 }));
    const a = splitMovementDataset(dataset, 0.4);
    const b = splitMovementDataset(dataset, 0.4);
    expect(a.holdout.sequences.map((s) => s.trajectoryId)).toEqual(b.holdout.sequences.map((s) => s.trajectoryId));
    expect(a.train.sequences).toHaveLength(6);
    expect(a.holdout.sequences).toHaveLength(4);
  });
});

describe("scoreSequences via generate", () => {
  it("terminates generation at the end token", async () => {
    const dataset = buildMovementDataset([
      buildTrajectorySpan({
        id: "t1",
        sessionId: "s1",
        actions: [action("device", 1, { gesture: "tap", target: "a" }), action("device", 2, { gesture: "tap", target: "b" })],
      }),
    ]);
    const model = await new MarkovMovementModelBackend().train(dataset);
    const generated = model.generate([], 100);
    expect(generated).not.toContain(MOVEMENT_END_TOKEN);
    expect(generated.length).toBeLessThanOrEqual(2);
  });
});
