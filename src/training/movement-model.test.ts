import { describe, expect, it } from "vitest";
import {
  MarkovMovementBackend,
  buildMovementDataset,
  createDefaultMovementBackendRegistry,
  movementAbstractKey,
  movementTokenKey,
  tokenizeAction,
  tokenizeReplayEvent,
  type MovementDataset,
  type MovementToken,
} from "./movement-model.js";
import { buildTrajectorySpan, type TrajectoryAction } from "../capture/trajectory.js";

function action(tool: string, summary: string, ts: number, metadata?: Record<string, unknown>): TrajectoryAction {
  return { kind: "action", tool, summary, ts, ...(metadata ? { metadata } : {}) };
}

function datasetFrom(sequences: MovementToken[][]): MovementDataset {
  return {
    version: 1,
    sequences: sequences.map((tokens, index) => ({ id: `seq-${index}`, tokens })),
  };
}

describe("movement tokenization", () => {
  it("derives tokens from gesture metadata", () => {
    const token = tokenizeAction(action("device", "tapped submit", 1, { gesture: "tap", target: "submit-button" }));
    expect(token).toEqual({ tool: "device", action: "tap", target: "submit-button" });
  });

  it("falls back to the summary lead word when no gesture metadata is present", () => {
    const token = tokenizeAction(action("shell", "Run the build", 1));
    expect(token).toEqual({ tool: "shell", action: "run" });
  });

  it("uses direction as target when target is absent", () => {
    const token = tokenizeAction(action("device", "scrolled down", 1, { gesture: "scroll", direction: "down" }));
    expect(token).toEqual({ tool: "device", action: "scroll", target: "down" });
  });

  it("tokenizes replay action events and ignores non-actions", () => {
    expect(
      tokenizeReplayEvent({ kind: "action", ts: 1, trajectoryId: "t", tool: "device", summary: "tapped ok" }),
    ).toEqual({ tool: "device", action: "tapped" });
    expect(
      tokenizeReplayEvent({ kind: "observation", ts: 1, trajectoryId: "t", source: "os", summary: "x" }),
    ).toBeUndefined();
  });

  it("keys are stable and abstraction drops the target", () => {
    const token: MovementToken = { tool: "device", action: "tap", target: "submit" };
    expect(movementTokenKey(token)).toContain("submit");
    expect(movementAbstractKey(token)).not.toContain("submit");
  });
});

describe("buildMovementDataset", () => {
  it("orders actions by timestamp", () => {
    const trajectory = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [action("device", "b", 20, { gesture: "type" }), action("device", "a", 10, { gesture: "tap" })],
    });
    const dataset = buildMovementDataset([trajectory]);
    expect(dataset.sequences[0].tokens.map((t) => t.action)).toEqual(["tap", "type"]);
  });
});

describe("MarkovMovementBackend", () => {
  it("reproduces a memorized movement via free rollout", () => {
    const sequence: MovementToken[] = [
      { tool: "device", action: "tap", target: "field" },
      { tool: "device", action: "type", target: "text" },
      { tool: "device", action: "tap", target: "submit" },
    ];
    const model = new MarkovMovementBackend().train(datasetFrom([sequence]));
    const generated = model.generate();
    expect(generated).toEqual(sequence);
  });

  it("terminates rollout when an end transition is learned", () => {
    const model = new MarkovMovementBackend().train(
      datasetFrom([[{ tool: "device", action: "tap", target: "only" }]]),
    );
    const generated = model.generate({ maxSteps: 50 });
    expect(generated).toHaveLength(1);
  });

  it("predicts the most frequent next movement deterministically", () => {
    const common: MovementToken = { tool: "device", action: "type", target: "common" };
    const rare: MovementToken = { tool: "device", action: "type", target: "rare" };
    const seed: MovementToken = { tool: "device", action: "tap", target: "field" };
    const model = new MarkovMovementBackend().train(
      datasetFrom([[seed, common], [seed, common], [seed, rare]]),
    );
    const prediction = model.predict([seed]);
    expect(prediction.token).toEqual(common);
    expect(prediction.confidence).toBeCloseTo(2 / 3, 5);
    // ranked candidates expose the tail
    expect(prediction.candidates.map((c) => c.token?.target)).toContain("rare");
  });

  it("generalizes to novel targets via abstraction backoff", () => {
    // Train on one target set; the structural pattern is tap -> type -> tap.
    const train: MovementToken[] = [
      { tool: "device", action: "tap", target: "old-field" },
      { tool: "device", action: "type", target: "old-value" },
      { tool: "device", action: "tap", target: "old-submit" },
    ];
    const model = new MarkovMovementBackend().train(datasetFrom([train]));

    // Present a context whose targets were never seen in training.
    const novelContext: MovementToken[] = [{ tool: "device", action: "tap", target: "new-field" }];
    const prediction = model.predict(novelContext);
    expect(prediction.token).toBeDefined();
    expect(prediction.abstracted).toBe(true);
    // structurally correct next movement even though the concrete context is unseen
    expect(prediction.token?.action).toBe("type");
  });

  it("round-trips through a snapshot", () => {
    const sequence: MovementToken[] = [
      { tool: "device", action: "tap", target: "field" },
      { tool: "device", action: "type", target: "text" },
    ];
    const model = new MarkovMovementBackend().train(datasetFrom([sequence]));
    const snapshot = model.snapshot();
    const roundTripped = JSON.parse(JSON.stringify(snapshot));
    const restored = MarkovMovementBackend.fromSnapshot(roundTripped);
    expect(restored.generate()).toEqual(model.generate());
    expect(restored.predict([sequence[0]]).token).toEqual(model.predict([sequence[0]]).token);
  });

  it("rejects snapshots from a foreign backend", () => {
    const model = new MarkovMovementBackend().train(datasetFrom([[{ tool: "device", action: "tap" }]]));
    const snapshot = { ...model.snapshot(), backend: "other" };
    expect(() => MarkovMovementBackend.fromSnapshot(snapshot)).toThrow(/other/);
  });
});

describe("MovementBackendRegistry", () => {
  it("provides the markov backend by default and rejects unknown names", () => {
    const registry = createDefaultMovementBackendRegistry();
    expect(registry.has("markov")).toBe(true);
    expect(registry.list()).toContain("markov");
    expect(registry.get("markov")).toBeInstanceOf(MarkovMovementBackend);
    expect(() => registry.get("mlx")).toThrow(/Unknown movement backend/);
  });

  it("accepts a pluggable custom backend", () => {
    const registry = createDefaultMovementBackendRegistry();
    const stub = {
      name: "stub",
      train: () => new MarkovMovementBackend().train(datasetFrom([[{ tool: "x", action: "y" }]])),
    };
    registry.register(stub);
    expect(registry.get("stub")).toBe(stub);
    expect(registry.list()).toEqual(expect.arrayContaining(["markov", "stub"]));
  });
});
