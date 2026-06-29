import { describe, expect, it } from "vitest";
import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  MarkovMovementBackend,
  buildMovementDataset,
  buildMovementDatasetFromReplays,
  evaluateNextActionAccuracy,
  generateSyntheticMovementDataset,
  restoreMovementModel,
  tokenizeMovement,
  type MovementDataset,
  type MovementEvent,
} from "./movement-model.js";

function makeTrajectory(id: string, actions: Array<{ tool: string; ts: number; metadata?: Record<string, unknown> }>): TrajectorySpan {
  return {
    id,
    sessionId: "session-1",
    createdAt: "2026-06-29T00:00:00.000Z",
    captureTier: "operator",
    observations: [],
    actions: actions.map((action) => ({ kind: "action", summary: `${action.tool} action`, ...action })),
  };
}

describe("movement tokenization", () => {
  it("produces stable round-trippable tokens", () => {
    const event: MovementEvent = { tool: "device", gesture: "tap", target: "search" };
    expect(tokenizeMovement(event)).toBe("device|tap|search|");
    expect(tokenizeMovement(event)).toBe(tokenizeMovement({ ...event }));
  });
});

describe("MarkovMovementBackend", () => {
  it("repeats a recorded movement exactly (memorization)", async () => {
    const dataset: MovementDataset = {
      version: 1,
      sequences: [
        {
          trajectoryId: "t1",
          events: [
            { tool: "device", gesture: "tap", target: "search" },
            { tool: "device", gesture: "type", target: "query" },
            { tool: "device", gesture: "tap", target: "result" },
          ],
        },
      ],
    };
    const model = await new MarkovMovementBackend().train(dataset);
    const rollout = model.generate([dataset.sequences[0].events[0]], 2);
    expect(rollout).toEqual([
      { tool: "device", gesture: "type", target: "query" },
      { tool: "device", gesture: "tap", target: "result" },
    ]);
  });

  it("is deterministic: argmax with lexical tie-break", async () => {
    // From context [A], B appears twice and C once -> B wins; ties go lexical.
    const a: MovementEvent = { tool: "a" };
    const b: MovementEvent = { tool: "b" };
    const c: MovementEvent = { tool: "c" };
    const dataset: MovementDataset = {
      version: 1,
      sequences: [
        { trajectoryId: "s1", events: [a, b] },
        { trajectoryId: "s2", events: [a, b] },
        { trajectoryId: "s3", events: [a, c] },
      ],
    };
    const model = await new MarkovMovementBackend().train(dataset);
    const prediction = model.predictNext([a]);
    expect(prediction?.event).toEqual(b);
    expect(prediction?.probability).toBeCloseTo(2 / 3, 10);
    expect(prediction?.order).toBe(1);
  });

  it("generalizes to an unseen prefix via suffix backoff", async () => {
    // Train: X -> Y -> Z appears, plus a different lead-in W -> Y.
    const w: MovementEvent = { tool: "w" };
    const x: MovementEvent = { tool: "x" };
    const y: MovementEvent = { tool: "y" };
    const z: MovementEvent = { tool: "z" };
    const dataset: MovementDataset = {
      version: 1,
      sequences: [
        { trajectoryId: "s1", events: [x, y, z] },
        { trajectoryId: "s2", events: [w, y, z] },
      ],
    };
    const model = await new MarkovMovementBackend().train(dataset, { maxOrder: 3 });
    // Novel prefix [w, x, y] was never seen, but suffix [y] -> z was: backoff predicts z.
    const prediction = model.predictNext([w, x, y]);
    expect(prediction?.event).toEqual(z);
    expect(prediction?.order).toBeLessThan(3);
  });

  it("returns undefined when nothing was learned for any context", async () => {
    const model = await new MarkovMovementBackend().train({ version: 1, sequences: [] });
    expect(model.predictNext([{ tool: "anything" }])).toBeUndefined();
    expect(model.generate([{ tool: "anything" }], 3)).toEqual([]);
  });

  it("round-trips through serialize/restore with identical predictions", async () => {
    const dataset = generateSyntheticMovementDataset({ seed: 7, sequenceCount: 5 });
    const model = await new MarkovMovementBackend().train(dataset);
    const restored = restoreMovementModel(model.serialize());
    const context = dataset.sequences[0].events.slice(0, 1);
    expect(restored.predictNext(context)).toEqual(model.predictNext(context));
    expect(restored.backend).toBe(model.backend);
  });
});

describe("dataset builders", () => {
  it("builds a dataset from trajectory actions, sorted by timestamp", () => {
    const trajectory = makeTrajectory("t1", [
      { tool: "device", ts: 30, metadata: { gesture: "tap", target: "result" } },
      { tool: "device", ts: 10, metadata: { gesture: "tap", target: "search" } },
      { tool: "device", ts: 20, metadata: { gesture: "type", target: "query" } },
    ]);
    const dataset = buildMovementDataset([trajectory]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0].events.map((event) => event.target)).toEqual(["search", "query", "result"]);
  });

  it("prefers reviewed/redacted actions when present", () => {
    const trajectory: TrajectorySpan = {
      ...makeTrajectory("t1", [{ tool: "device", ts: 10, metadata: { target: "secret" } }]),
      review: {
        status: "approved",
        reviewedAt: "2026-06-29T00:00:00.000Z",
        reviewedBy: "reviewer",
        redactedActions: [{ ts: 10, tool: "device", summary: "redacted" }],
      },
    };
    const dataset = buildMovementDataset([trajectory]);
    expect(dataset.sequences[0].events).toEqual([{ tool: "device" }]);
  });

  it("builds a dataset from replay manifests", () => {
    const dataset = buildMovementDatasetFromReplays([
      {
        version: 1,
        sessionId: "s1",
        trajectoryIds: ["t1"],
        eventCount: 3,
        events: [
          { kind: "observation", ts: 1, trajectoryId: "t1", source: "device", summary: "obs" },
          { kind: "action", ts: 2, trajectoryId: "t1", tool: "device", summary: "tap" },
          { kind: "action", ts: 3, trajectoryId: "t1", tool: "browser", summary: "scroll" },
        ],
      },
    ]);
    expect(dataset.sequences).toEqual([
      { trajectoryId: "t1", events: [{ tool: "device" }, { tool: "browser" }] },
    ]);
  });
});

describe("synthetic generator + eval harness", () => {
  it("is reproducible for a fixed seed", () => {
    const a = generateSyntheticMovementDataset({ seed: 42, sequenceCount: 4 });
    const b = generateSyntheticMovementDataset({ seed: 42, sequenceCount: 4 });
    expect(a).toEqual(b);
    expect(a.sequences).toHaveLength(4);
  });

  it("trained model generalizes to held-out synthetic sequences above chance", async () => {
    const train = generateSyntheticMovementDataset({ seed: 1, sequenceCount: 40 });
    const heldOut = generateSyntheticMovementDataset({ seed: 999, sequenceCount: 10 });
    const model = await new MarkovMovementBackend().train(train, { maxOrder: 3 });
    const evalResult = evaluateNextActionAccuracy(model, heldOut.sequences);
    expect(evalResult.predictions).toBeGreaterThan(0);
    // The synthetic grammar has strong motif structure; backoff should beat chance.
    expect(evalResult.accuracy).toBeGreaterThan(0.5);
  });

  it("memorizes its own training set with high fidelity", async () => {
    const train = generateSyntheticMovementDataset({ seed: 5, sequenceCount: 20 });
    const model = await new MarkovMovementBackend().train(train, { maxOrder: 4 });
    const evalResult = evaluateNextActionAccuracy(model, train.sequences);
    expect(evalResult.accuracy).toBeGreaterThan(0.8);
  });
});
