import { describe, expect, it } from "vitest";
import { buildTrajectorySpan, type TrajectorySpan } from "../capture/trajectory.js";
import { buildReplayManifest } from "../capture/replay.js";
import {
  MarkovMovementBackend,
  MOVEMENT_EOS,
  buildMovementDataset,
  buildMovementDatasetFromReplays,
  evaluateMovementModel,
  loadMovementModel,
  tokenizeTrajectory,
  type MovementDataset,
} from "./movement-model.js";

/**
 * Synthetic movement generator — simulates a recorded "open menu, click item,
 * confirm" style interaction so the model layer is validated without any real
 * OS input (the actual capture runs only when the user runs bee-agent locally).
 */
function syntheticTrajectory(id: string, tools: string[], startTs = 1000): TrajectorySpan {
  return buildTrajectorySpan({
    id,
    sessionId: "session-synthetic",
    captureTier: "full",
    observations: [{ kind: "observation", source: "screen", summary: "frame", ts: startTs }],
    actions: tools.map((tool, index) => ({
      kind: "action" as const,
      tool,
      summary: `${tool} step`,
      ts: startTs + index + 1,
    })),
  });
}

describe("movement tokenization", () => {
  it("orders interleaved observations and actions by timestamp", () => {
    const trajectory = buildTrajectorySpan({
      id: "traj-1",
      sessionId: "s1",
      observations: [{ kind: "observation", source: "screen", summary: "f", ts: 5 }],
      actions: [
        { kind: "action", tool: "mouse.move", summary: "m", ts: 3 },
        { kind: "action", tool: "mouse.click", summary: "c", ts: 7 },
      ],
    });
    expect(tokenizeTrajectory(trajectory).tokens).toEqual(["act:mouse.move", "obs:screen", "act:mouse.click"]);
  });

  it("derives one sequence per trajectory from a replay manifest", () => {
    const trajectory = syntheticTrajectory("traj-r", ["mouse.move", "mouse.click"]);
    const manifest = buildReplayManifest({ sessionId: "s1", transcript: [], trajectories: [trajectory] });
    const dataset = buildMovementDatasetFromReplays([manifest]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]?.tokens).toEqual(["obs:screen", "act:mouse.move", "act:mouse.click"]);
  });
});

describe("MarkovMovementBackend", () => {
  const backend = new MarkovMovementBackend();

  it("reproduces a recorded movement exactly from the start (piece c)", () => {
    const trajectory = syntheticTrajectory("traj-a", ["menu.open", "item.click", "dialog.confirm"]);
    const model = backend.train(buildMovementDataset([trajectory]), { order: 2 });
    const replay = model.generate();
    expect(replay).toEqual(["obs:screen", "act:menu.open", "act:item.click", "act:dialog.confirm"]);
  });

  it("predicts deterministically and exposes a normalized distribution", () => {
    const model = backend.train(
      buildMovementDataset([
        syntheticTrajectory("t1", ["a", "b"]),
        syntheticTrajectory("t2", ["a", "b"]),
        syntheticTrajectory("t3", ["a", "c"]),
      ]),
      { order: 3 },
    );
    // After obs:screen -> act:a, the next token is act:b twice vs act:c once.
    const dist = model.distribution(["obs:screen", "act:a"]);
    expect(dist[0]?.token).toBe("act:b");
    expect(dist[0]?.probability).toBeCloseTo(2 / 3);
    const total = dist.reduce((sum, entry) => sum + entry.probability, 0);
    expect(total).toBeCloseTo(1);
    // Determinism: same input -> same output.
    expect(model.predict(["obs:screen", "act:a"])).toBe("act:b");
  });

  it("generalizes to a new but related movement via back-off (piece d)", () => {
    // Training never contains the exact sequence [x, a, b], but the sub-movement
    // "a -> b" is well attested, so back-off predicts b after an unseen prefix.
    const model = backend.train(
      buildMovementDataset([
        syntheticTrajectory("t1", ["a", "b", "c"]),
        syntheticTrajectory("t2", ["z", "a", "b"]),
      ]),
      { order: 2 },
    );
    // "act:q" was never seen before "act:a", but order-1 back-off knows a -> b.
    expect(model.predict(["act:q", "act:a"])).toBe("act:b");
  });

  it("terminates generation at EOS rather than looping forever", () => {
    const model = backend.train(buildMovementDataset([syntheticTrajectory("t1", ["a", "b"])]), { order: 2 });
    const generated = model.generate([], { maxLength: 100 });
    expect(generated).toEqual(["obs:screen", "act:a", "act:b"]);
    expect(generated).not.toContain(MOVEMENT_EOS);
  });

  it("round-trips through serialization for inference-only reload", () => {
    const trajectory = syntheticTrajectory("t1", ["a", "b", "c"]);
    const model = backend.train(buildMovementDataset([trajectory]), { order: 2 });
    const reloaded = loadMovementModel(model.toJSON());
    expect(reloaded.order).toBe(model.order);
    expect(reloaded.generate()).toEqual(model.generate());
  });
});

describe("evaluateMovementModel", () => {
  it("scores perfect fidelity on held-out copies of trained movements", () => {
    const backend = new MarkovMovementBackend();
    const dataset = buildMovementDataset([
      syntheticTrajectory("t1", ["menu.open", "item.click"]),
      syntheticTrajectory("t2", ["menu.open", "item.click"]),
    ]);
    const model = backend.train(dataset, { order: 2 });
    const result = evaluateMovementModel(model, dataset);
    expect(result.sequenceCount).toBe(2);
    expect(result.accuracy).toBe(1);
    expect(result.exactReplays).toBe(2);
  });

  it("measures partial generalization on held-out related movements", () => {
    const backend = new MarkovMovementBackend();
    const train: MovementDataset = buildMovementDataset([
      syntheticTrajectory("t1", ["a", "b", "c"]),
      syntheticTrajectory("t2", ["a", "b", "d"]),
    ]);
    const model = backend.train(train, { order: 2 });
    const heldOut = buildMovementDataset([syntheticTrajectory("t3", ["a", "b", "c"])]);
    const result = evaluateMovementModel(model, heldOut);
    // obs:screen -> act:a -> act:b are all deterministic; act:c vs act:d is a
    // toss-up broken lexicographically to act:c, plus EOS after c. So most of
    // the movement is recovered.
    expect(result.accuracy).toBeGreaterThan(0.5);
    expect(result.predictionCount).toBe(5); // 1 obs + 3 actions + EOS
  });
});
