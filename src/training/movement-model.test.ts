import { describe, expect, it } from "vitest";
import type { ReplayManifest } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  DEFAULT_MOVEMENT_TRAINING_CONFIG,
  MarkovMovementBackend,
  MOVEMENT_END,
  MOVEMENT_START,
  buildMovementDatasetFromReplays,
  buildMovementDatasetFromTrajectories,
  createSyntheticMovementDataset,
  detokenizeMovements,
  evaluateMovementModel,
  generateMovementSequence,
  predictNextMovement,
  tokenizeMovement,
  type MovementSequence,
} from "./movement-model.js";

function trajectory(id: string, tools: string[]): TrajectorySpan {
  return {
    id,
    sessionId: `session-${id}`,
    createdAt: "2026-01-01T00:00:00.000Z",
    captureTier: "full",
    observations: [],
    actions: tools.map((tool, index) => ({
      kind: "action" as const,
      tool,
      summary: `did ${tool}`,
      ts: index,
    })),
  };
}

describe("movement dataset construction", () => {
  it("tokenizes actions into normalized movement primitives", () => {
    expect(tokenizeMovement({ tool: "Mouse Move" })).toBe("act:mouse_move");
    expect(tokenizeMovement({ tool: "key.type" })).toBe("act:key.type");
  });

  it("orders actions by timestamp and records exemplars", () => {
    const span: TrajectorySpan = {
      ...trajectory("t1", []),
      actions: [
        { kind: "action", tool: "b", summary: "second", ts: 5 },
        { kind: "action", tool: "a", summary: "first", ts: 1 },
      ],
    };
    const dataset = buildMovementDatasetFromTrajectories([span]);
    expect(dataset.sequences[0]?.tokens).toEqual(["act:a", "act:b"]);
    expect(dataset.exemplars["act:a"]).toMatchObject({ tool: "a", summary: "first" });
    expect(dataset.vocab).toEqual(["act:a", "act:b"]);
  });

  it("builds a dataset from replay manifests using action events only", () => {
    const replay: ReplayManifest = {
      version: 1,
      sessionId: "s1",
      trajectoryIds: ["t1"],
      eventCount: 3,
      events: [
        { kind: "transcript", ts: 0, messageId: "m1", role: "user", content: "hi" },
        { kind: "action", ts: 1, trajectoryId: "t1", tool: "mouse.move", summary: "move" },
        { kind: "action", ts: 2, trajectoryId: "t1", tool: "mouse.click", summary: "click" },
      ],
    };
    const dataset = buildMovementDatasetFromReplays([replay]);
    expect(dataset.sequences[0]?.tokens).toEqual(["act:mouse.move", "act:mouse.click"]);
  });
});

describe("MarkovMovementBackend", () => {
  const backend = new MarkovMovementBackend();

  it("reproduces a deterministic recorded movement sequence", () => {
    const dataset = buildMovementDatasetFromTrajectories([
      trajectory("t1", ["open", "focus", "type", "save"]),
    ]);
    const model = backend.train(dataset);
    const generated = generateMovementSequence(model);
    expect(generated).toEqual(["act:open", "act:focus", "act:type", "act:save"]);
  });

  it("predicts the most likely next movement with backoff", () => {
    const dataset = buildMovementDatasetFromTrajectories([
      trajectory("t1", ["open", "focus", "type"]),
      trajectory("t2", ["open", "focus", "type"]),
      trajectory("t3", ["open", "focus", "save"]),
    ]);
    const model = backend.train(dataset, { order: 2, smoothing: 0.01 });
    const prediction = predictNextMovement(model, ["act:open", "act:focus"]);
    // "type" appeared twice after (open, focus), "save" once -> type wins.
    expect(prediction.token).toBe("act:type");
    expect(prediction.fallback).toBe(false);
    // An unseen context backs off to shorter contexts rather than failing.
    const backoff = predictNextMovement(model, ["act:never-seen", "act:focus"]);
    expect(backoff.token).toBe("act:type");
    expect(backoff.candidates[0]?.contextOrder).toBeLessThan(2);
  });

  it("falls back to a uniform distribution on an empty model", () => {
    const model = backend.train({ version: 1, vocab: [], sequences: [], exemplars: {} });
    const prediction = predictNextMovement(model, []);
    expect(prediction.fallback).toBe(true);
    expect(prediction.token).toBe(MOVEMENT_END);
  });

  it("never emits START/END tokens as generated movements", () => {
    const dataset = createSyntheticMovementDataset({ sequenceCount: 20, seed: 7 });
    const model = backend.train(dataset);
    const generated = generateMovementSequence(model, { maxLength: 40 });
    expect(generated).not.toContain(MOVEMENT_START);
    expect(generated).not.toContain(MOVEMENT_END);
  });

  it("respects the maxLength cap when generating", () => {
    const dataset = buildMovementDatasetFromTrajectories([
      trajectory("loop", ["move", "move", "move", "move", "move"]),
    ]);
    const model = backend.train(dataset, { order: 1, smoothing: 0 });
    const generated = generateMovementSequence(model, { maxLength: 3 });
    expect(generated.length).toBe(3);
  });

  it("maps generated tokens back to replayable actions via exemplars", () => {
    const dataset = buildMovementDatasetFromTrajectories([trajectory("t1", ["open", "save"])]);
    const model = backend.train(dataset);
    const actions = detokenizeMovements(model, ["act:open", "act:save"]);
    expect(actions).toEqual([
      { token: "act:open", tool: "open", summary: "did open" },
      { token: "act:save", tool: "save", summary: "did save" },
    ]);
  });

  it("is deterministic: identical datasets train identical models", () => {
    const dataset = createSyntheticMovementDataset({ sequenceCount: 10, seed: 42 });
    const a = backend.train(dataset, DEFAULT_MOVEMENT_TRAINING_CONFIG);
    const b = backend.train(dataset, DEFAULT_MOVEMENT_TRAINING_CONFIG);
    expect(JSON.stringify(a)).toBe(JSON.stringify(b));
  });
});

describe("synthetic generator + generalization eval", () => {
  it("produces reproducible datasets for a fixed seed", () => {
    const a = createSyntheticMovementDataset({ sequenceCount: 5, seed: 99 });
    const b = createSyntheticMovementDataset({ sequenceCount: 5, seed: 99 });
    expect(a.sequences).toEqual(b.sequences);
    const c = createSyntheticMovementDataset({ sequenceCount: 5, seed: 100 });
    expect(c.sequences).not.toEqual(a.sequences);
  });

  it("generalizes: a model trained on a grammar beats an untrained baseline on held-out data", () => {
    const backend = new MarkovMovementBackend();
    const train = createSyntheticMovementDataset({ sequenceCount: 120, seed: 1, idPrefix: "train" });
    const test = createSyntheticMovementDataset({ sequenceCount: 40, seed: 2, idPrefix: "test" });

    const trained = backend.train(train, { order: 2, smoothing: 0.05 });
    const report = evaluateMovementModel(trained, test.sequences);

    // Held-out sequences share the grammar but are not identical strings, so
    // meaningful accuracy demonstrates generalization, not memorization.
    expect(report.predictions).toBeGreaterThan(0);
    expect(report.accuracy).toBeGreaterThan(0.5);
    expect(Number.isFinite(report.perplexity)).toBe(true);

    // Compare against an untrained model (trained on empty data) -> pure fallback.
    const untrained = backend.train({ version: 1, vocab: train.vocab, sequences: [], exemplars: {} });
    const baseline = evaluateMovementModel(untrained, test.sequences);
    expect(report.accuracy).toBeGreaterThan(baseline.accuracy);
    expect(report.perplexity).toBeLessThan(baseline.perplexity);
  });

  it("scores perfect accuracy when evaluating on memorized sequences", () => {
    const backend = new MarkovMovementBackend();
    const sequences: MovementSequence[] = [{ id: "m", tokens: ["act:a", "act:b", "act:c"] }];
    const model = backend.train({
      version: 1,
      vocab: ["act:a", "act:b", "act:c"],
      sequences,
      exemplars: {},
    }, { order: 3, smoothing: 0.001 });
    const report = evaluateMovementModel(model, sequences);
    expect(report.correct).toBe(report.predictions);
    expect(report.accuracy).toBe(1);
  });
});
