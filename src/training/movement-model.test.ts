import { describe, expect, it } from "vitest";
import { buildReplayManifest } from "../capture/replay.js";
import { buildTrajectorySpan, type TrajectoryAction } from "../capture/trajectory.js";
import {
  MarkovMovementBackend,
  MOVEMENT_START_TOKEN,
  MovementBackendRegistry,
  buildMovementDataset,
  buildMovementSequenceFromTrajectory,
  buildMovementSequencesFromReplay,
  createDefaultMovementBackendRegistry,
  deserializeMovementModel,
  evaluateGeneralization,
  evaluateReplayFidelity,
  normalizeMovementToken,
  type MovementSequence,
} from "./movement-model.js";

function action(tool: string, ts: number): TrajectoryAction {
  return { kind: "action", tool, summary: tool, ts };
}

function sequence(trajectoryId: string, tokens: string[]): MovementSequence {
  return { trajectoryId, tokens };
}

describe("movement tokenization", () => {
  it("normalizes tool strings into stable tokens", () => {
    expect(normalizeMovementToken("  Mouse Click ")).toBe("mouse-click");
    expect(normalizeMovementToken("keyboard.type")).toBe("keyboard.type");
    expect(normalizeMovementToken("   ")).toBe("unknown");
  });

  it("derives an ordered movement sequence from a trajectory's actions", () => {
    const trajectory = buildTrajectorySpan({
      id: "traj-1",
      sessionId: "session-1",
      actions: [action("mouse.move", 3), action("mouse.click", 1), action("keyboard.type", 2)],
    });
    const result = buildMovementSequenceFromTrajectory(trajectory);
    expect(result.trajectoryId).toBe("traj-1");
    // sorted by timestamp
    expect(result.tokens).toEqual(["mouse.click", "keyboard.type", "mouse.move"]);
  });

  it("builds sequences from a replay manifest, one per trajectory", () => {
    const trajectory = buildTrajectorySpan({
      id: "traj-1",
      sessionId: "session-1",
      actions: [action("mouse.click", 1), action("keyboard.type", 2)],
    });
    const manifest = buildReplayManifest({ sessionId: "session-1", transcript: [], trajectories: [trajectory] });
    const sequences = buildMovementSequencesFromReplay(manifest);
    expect(sequences).toHaveLength(1);
    expect(sequences[0]?.tokens).toEqual(["mouse.click", "keyboard.type"]);
  });

  it("builds a de-duplicated, sorted vocabulary and drops empty sequences", () => {
    const dataset = buildMovementDataset([
      sequence("a", ["mouse.click", "keyboard.type"]),
      sequence("b", ["mouse.click", "mouse.move"]),
      sequence("c", []),
    ]);
    expect(dataset.vocab).toEqual(["keyboard.type", "mouse.click", "mouse.move"]);
    expect(dataset.sequences).toHaveLength(2);
  });
});

describe("MarkovMovementBackend — learning and repetition (objective 2c)", () => {
  const recorded = [
    sequence("t1", ["focus", "mouse.click", "keyboard.type", "keyboard.enter", "wait"]),
    sequence("t2", ["focus", "mouse.click", "keyboard.type", "keyboard.enter", "wait"]),
  ];

  it("reproduces a deterministic recorded movement exactly", () => {
    const dataset = buildMovementDataset(recorded);
    const model = new MarkovMovementBackend().train(dataset, { order: 2 });
    const original = recorded[0]!.tokens;
    const generated = model.generate([MOVEMENT_START_TOKEN, original[0]!], original.length - 1);
    expect(generated).toEqual(original.slice(1));
  });

  it("predicts the first move from the start boundary", () => {
    const dataset = buildMovementDataset(recorded);
    const model = new MarkovMovementBackend().train(dataset);
    const prediction = model.predictNext([MOVEMENT_START_TOKEN]);
    expect(prediction?.token).toBe("focus");
    expect(prediction?.probability).toBeGreaterThan(0);
  });

  it("scores a recorded sequence as more likely than a scrambled one", () => {
    const dataset = buildMovementDataset(recorded);
    const model = new MarkovMovementBackend().train(dataset);
    const good = model.scoreSequence(recorded[0]!.tokens);
    const bad = model.scoreSequence(["wait", "keyboard.enter", "focus", "mouse.click"]);
    expect(good.logProb).toBeGreaterThan(bad.logProb);
    expect(good.perplexity).toBeLessThan(bad.perplexity);
  });

  it("reports high replay fidelity over the training set", () => {
    const dataset = buildMovementDataset(recorded);
    const model = new MarkovMovementBackend().train(dataset);
    const report = evaluateReplayFidelity(model, recorded);
    expect(report.sequenceCount).toBe(2);
    expect(report.replayFidelity).toBe(1);
    expect(report.exactReplays).toBe(2);
  });
});

describe("MarkovMovementBackend — generalization (objective 2d)", () => {
  it("predicts plausible continuations for new-but-related sequences via backoff", () => {
    // Train on two related workflows that share bigrams.
    const training = buildMovementDataset([
      sequence("a", ["open", "mouse.click", "keyboard.type", "save"]),
      sequence("b", ["open", "mouse.click", "keyboard.type", "close"]),
      sequence("c", ["open", "scroll", "mouse.click", "keyboard.type", "save"]),
    ]);
    const model = new MarkovMovementBackend().train(training, { order: 2 });

    // Held-out sequence recombines seen moves in an unseen full-order context.
    const heldOut = [sequence("h1", ["open", "scroll", "mouse.click", "keyboard.type", "close"])];
    const report = evaluateGeneralization(model, heldOut);

    // Every context yields an in-vocabulary prediction (backoff never leaves a gap).
    expect(report.coverage).toBe(1);
    // The model still learned the strong "mouse.click -> keyboard.type" regularity.
    expect(report.nextTokenAccuracy).toBeGreaterThan(0.5);
  });

  it("backs off to a shorter context when the full-order context is unseen", () => {
    const training = buildMovementDataset([
      sequence("a", ["a", "b", "c"]),
      sequence("b", ["x", "b", "c"]),
    ]);
    const model = new MarkovMovementBackend().train(training, { order: 2 });
    // Context ["z","b"] was never seen at order 2, but "b" -> "c" was seen at order 1.
    const prediction = model.predictNext(["z", "b"]);
    expect(prediction?.token).toBe("c");
    expect(prediction?.order).toBe(1);
  });

  it("returns undefined when the model has no data at all", () => {
    const model = new MarkovMovementBackend().train(buildMovementDataset([]));
    expect(model.predictNext(["anything"])).toBeUndefined();
  });
});

describe("serialization and registry (pluggable seam)", () => {
  it("round-trips a trained model through JSON", () => {
    const dataset = buildMovementDataset([sequence("a", ["one", "two", "three"])]);
    const model = new MarkovMovementBackend().train(dataset, { order: 2 });
    const restored = deserializeMovementModel(model.toJSON());
    expect(restored.order).toBe(model.order);
    expect(restored.backendId).toBe(model.backendId);
    expect(restored.generate([MOVEMENT_START_TOKEN, "one"], 2)).toEqual(["two", "three"]);
  });

  it("resolves the built-in backend from the default registry", () => {
    const registry = createDefaultMovementBackendRegistry();
    expect(registry.get("markov")).toBeInstanceOf(MarkovMovementBackend);
    expect(registry.list().map((backend) => backend.id)).toContain("markov");
  });

  it("lets a custom backend be registered and swapped in", () => {
    const registry = new MovementBackendRegistry();
    const custom = new MarkovMovementBackend();
    registry.register(custom);
    expect(registry.get("markov")).toBe(custom);
    expect(registry.get("missing")).toBeUndefined();
  });
});
