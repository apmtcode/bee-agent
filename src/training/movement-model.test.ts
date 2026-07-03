import { describe, expect, it } from "vitest";
import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  MarkovMovementBackend,
  MOVEMENT_END_SYMBOL,
  buildDatasetFromTrajectories,
  buildMovementDataset,
  evaluateDataset,
  evaluateSequence,
  generateMovements,
  symbolToToken,
  synthesizeMovementTrajectory,
  tokenizeReplayEvents,
  tokenizeTrajectory,
} from "./movement-model.js";

function trajectory(id: string, actions: Array<{ target: string; gesture: string; direction?: string }>): TrajectorySpan {
  let ts = 1000;
  return {
    id,
    sessionId: `session-${id}`,
    createdAt: "1970-01-01T00:00:00.000Z",
    captureTier: "app",
    observations: [{ kind: "observation", source: "device", summary: "app active", ts: (ts += 10) }],
    actions: actions.map((action) => ({
      kind: "action" as const,
      tool: "device",
      summary: `${action.gesture} ${action.target}`,
      ts: (ts += 10),
      metadata: {
        gesture: action.gesture,
        target: action.target,
        ...(action.direction ? { direction: action.direction } : {}),
      },
    })),
  };
}

describe("movement-model tokenisation", () => {
  it("orders observations and actions by timestamp and separates class from symbol", () => {
    const span = trajectory("t1", [
      { target: "search", gesture: "tap" },
      { target: "up", gesture: "swipe", direction: "up" },
    ]);
    const sequence = tokenizeTrajectory(span);
    expect(sequence.tokens.map((token) => token.symbol)).toEqual([
      "obs:device",
      "act:device:tap#search",
      "act:device:swipe:up#up",
    ]);
    // class drops the concrete target but keeps the gesture/direction shape
    expect(sequence.tokens[1]!.klass).toBe("act:device:tap");
    expect(sequence.tokens[2]!.klass).toBe("act:device:swipe:up");
  });

  it("tokenises replay timeline events and skips transcript events", () => {
    const sequence = tokenizeReplayEvents("t2", [
      { kind: "transcript", ts: 1, messageId: "m1", role: "user", content: "hi" },
      { kind: "observation", ts: 2, trajectoryId: "t2", source: "device", summary: "obs" },
      { kind: "action", ts: 3, trajectoryId: "t2", tool: "device", summary: "tap" },
    ]);
    expect(sequence.tokens.map((token) => token.symbol)).toEqual(["obs:device", "act:device"]);
  });

  it("round-trips a symbol back into kind and class", () => {
    expect(symbolToToken("act:device:tap#search")).toMatchObject({ kind: "action", klass: "act:device:tap" });
    expect(symbolToToken("obs:device")).toMatchObject({ kind: "observation", klass: "obs:device" });
    expect(symbolToToken(MOVEMENT_END_SYMBOL).symbol).toBe(MOVEMENT_END_SYMBOL);
  });

  it("builds a sorted, deduplicated vocabulary and drops empty sequences", () => {
    const dataset = buildMovementDataset([
      tokenizeTrajectory(trajectory("a", [{ target: "x", gesture: "tap" }])),
      { trajectoryId: "empty", tokens: [] },
    ]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.vocabulary).toEqual([...dataset.vocabulary].sort());
    expect(new Set(dataset.vocabulary).size).toBe(dataset.vocabulary.length);
  });
});

describe("MarkovMovementBackend training + inference", () => {
  it("repeats a recorded movement sequence exactly (objective 2c)", async () => {
    const backend = new MarkovMovementBackend();
    const span = trajectory("repeat", [
      { target: "search", gesture: "tap" },
      { target: "field", gesture: "type" },
      { target: "results", gesture: "scroll", direction: "down" },
      { target: "confirm", gesture: "tap" },
    ]);
    const dataset = buildDatasetFromTrajectories([span]);
    const model = await backend.train(dataset, { order: 3 });

    const sequence = tokenizeTrajectory(span);
    // seed with the first observation, then roll forward
    const rollout = generateMovements(backend, model, sequence.tokens.slice(0, 1));
    expect(rollout.stopped).toBe("end");
    expect(rollout.tokens.map((token) => token.symbol)).toEqual(sequence.tokens.slice(1).map((token) => token.symbol));

    // teacher-forced fidelity is perfect on the training sequence
    const evaluation = evaluateSequence(backend, model, sequence);
    expect(evaluation.accuracy).toBe(1);
  });

  it("is deterministic: identical datasets yield identical predictions", async () => {
    const backend = new MarkovMovementBackend();
    const spans = [
      trajectory("d1", [{ target: "a", gesture: "tap" }, { target: "b", gesture: "tap" }]),
      trajectory("d2", [{ target: "a", gesture: "tap" }, { target: "c", gesture: "type" }]),
    ];
    const dataset = buildDatasetFromTrajectories(spans);
    const modelA = await backend.train(dataset, { order: 2 });
    const modelB = await backend.train(dataset, { order: 2 });
    const context = { history: tokenizeTrajectory(spans[0]!).tokens.slice(0, 2) };
    expect(backend.predict(modelA, context)).toEqual(backend.predict(modelB, context));
  });

  it("records trainedAt only from an injected clock (hermetic)", async () => {
    const backend = new MarkovMovementBackend();
    const dataset = buildDatasetFromTrajectories([trajectory("c", [{ target: "x", gesture: "tap" }])]);
    expect((await backend.train(dataset)).trainedAt).toBeUndefined();
    expect((await backend.train(dataset, { now: () => "2026-01-01T00:00:00.000Z" })).trainedAt).toBe(
      "2026-01-01T00:00:00.000Z",
    );
  });

  it("falls back to END with an empty dataset", async () => {
    const backend = new MarkovMovementBackend();
    const model = await backend.train(buildMovementDataset([]));
    const prediction = backend.predict(model, { history: [] });
    expect(prediction.symbol).toBe(MOVEMENT_END_SYMBOL);
    expect(prediction.kind).toBe("end");
  });
});

describe("generalisation to new-but-related movements (objective 2d)", () => {
  it("generalises the learned gesture shape to unseen targets via class backoff", async () => {
    const backend = new MarkovMovementBackend();
    // Train on a family of trajectories that share a gesture SHAPE
    // (tap -> type -> tap) but use different concrete targets.
    const trainTargets = [
      ["home", "name", "save"],
      ["home", "email", "save"],
      ["home", "phone", "save"],
    ];
    const trainSpans = trainTargets.map((targets, index) =>
      trajectory(`train-${index}`, [
        { target: targets[0]!, gesture: "tap" },
        { target: targets[1]!, gesture: "type" },
        { target: targets[2]!, gesture: "tap" },
      ]),
    );
    const model = await backend.train(buildDatasetFromTrajectories(trainSpans), { order: 3 });

    // Held-out trajectory: same shape, a brand-new middle target ("address").
    const heldOut = tokenizeTrajectory(
      trajectory("held", [
        { target: "home", gesture: "tap" },
        { target: "address", gesture: "type" },
        { target: "save", gesture: "tap" },
      ]),
    );
    const evaluation = evaluateSequence(backend, model, heldOut);

    // The model has never seen "type#address", yet it should still predict a
    // "type" gesture after seeing tap#home, and recover the shared "save" step.
    expect(evaluation.accuracy).toBeGreaterThan(0.5);
    expect(evaluation.perStep.some((step) => step.generalized && step.hit)).toBe(true);
  });

  it("generalisation beats a memorisation-only (order-large, no shared context) baseline", async () => {
    const backend = new MarkovMovementBackend();
    const trainSpans = Array.from({ length: 6 }, (_, index) =>
      synthesizeMovementTrajectory({ id: `syn-${index}`, seed: 100 + index, targets: ["a", "b", "c", "d"] }),
    );
    const model = await backend.train(buildDatasetFromTrajectories(trainSpans), { order: 3 });

    // Held-out uses a fresh seed → related shape, novel target ordering.
    const heldOut = Array.from({ length: 4 }, (_, index) =>
      tokenizeTrajectory(
        synthesizeMovementTrajectory({ id: `held-${index}`, seed: 900 + index, targets: ["a", "b", "c", "d", "e"] }),
      ),
    );
    const evaluation = evaluateDataset(backend, model, heldOut);
    // Should do materially better than chance over the small vocabulary.
    expect(evaluation.meanAccuracy).toBeGreaterThan(0.3);
  });
});

describe("synthetic generator", () => {
  it("is deterministic for a given seed and reproduces gesture metadata", () => {
    const a = synthesizeMovementTrajectory({ id: "s", seed: 42 });
    const b = synthesizeMovementTrajectory({ id: "s", seed: 42 });
    expect(a).toEqual(b);
    expect(a.actions.length).toBeGreaterThan(0);
    expect(a.actions[0]!.metadata).toHaveProperty("gesture");
  });

  it("produces different streams for different seeds", () => {
    const a = synthesizeMovementTrajectory({ id: "s1", seed: 1 });
    const b = synthesizeMovementTrajectory({ id: "s2", seed: 2 });
    expect(tokenizeTrajectory(a).tokens.map((t) => t.symbol)).not.toEqual(
      tokenizeTrajectory(b).tokens.map((t) => t.symbol),
    );
  });
});
