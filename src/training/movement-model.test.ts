import { describe, expect, it } from "vitest";
import {
  MarkovMovementBackend,
  buildMovementDatasetFromReplay,
  buildMovementDatasetFromTrajectories,
  evaluateMovementModel,
  generateSyntheticMovementSessions,
  tokenizeAction,
  type MovementDataset,
} from "./movement-model.js";
import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";
import type { ReplayManifest } from "../capture/replay.js";

function action(tool: string, ts: number, metadata?: Record<string, unknown>, summary = tool): TrajectoryAction {
  return { kind: "action", tool, summary, ts, ...(metadata ? { metadata } : {}) };
}

function span(id: string, actions: TrajectoryAction[]): TrajectorySpan {
  return {
    id,
    sessionId: `session-${id}`,
    createdAt: "2026-07-03T00:00:00.000Z",
    captureTier: "full",
    observations: [],
    actions,
  };
}

describe("tokenizeAction", () => {
  it("prefers structured gesture metadata for a stable token", () => {
    expect(tokenizeAction(action("device", 1, { gesture: "tap", target: "Compose Button" }))).toBe(
      "device:tap:compose-button",
    );
    expect(tokenizeAction(action("device", 1, { gesture: "swipe", direction: "up" }))).toBe("device:swipe:up");
  });

  it("falls back to a summary slug when no gesture metadata is present", () => {
    expect(tokenizeAction(action("editor", 1, undefined, "Insert new line here"))).toBe("editor:insert-new-line");
  });
});

describe("dataset builders", () => {
  it("builds ordered token sequences from trajectory spans", () => {
    const dataset = buildMovementDatasetFromTrajectories([
      span("t1", [
        action("device", 2, { gesture: "type", target: "search" }),
        action("device", 1, { gesture: "tap", target: "search" }),
      ]),
    ]);
    expect(dataset.sequences).toHaveLength(1);
    // Sorted by ts, so tap (ts=1) precedes type (ts=2).
    expect(dataset.sequences[0]?.tokens).toEqual(["device:tap:search", "device:type:search"]);
  });

  it("drops empty sequences", () => {
    expect(buildMovementDatasetFromTrajectories([span("empty", [])]).sequences).toHaveLength(0);
  });

  it("builds sequences from a replay manifest, one per trajectory", () => {
    const manifest: ReplayManifest = {
      version: 1,
      sessionId: "s1",
      trajectoryIds: ["t1"],
      eventCount: 3,
      events: [
        { kind: "observation", ts: 0, trajectoryId: "t1", source: "device", summary: "app active" },
        { kind: "action", ts: 1, trajectoryId: "t1", tool: "device", summary: "tapped x" },
        { kind: "action", ts: 2, trajectoryId: "t1", tool: "device", summary: "swiped up" },
      ],
    };
    const dataset = buildMovementDatasetFromReplay(manifest);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]?.tokens).toEqual(["device:tapped-x", "device:swiped-up"]);
  });
});

describe("MarkovMovementBackend — repeat recorded movements", () => {
  it("regenerates a recorded single-path movement exactly", () => {
    const dataset: MovementDataset = {
      version: 1,
      sequences: [{ sessionId: "s", tokens: ["open", "search", "type", "submit"] }],
    };
    const model = new MarkovMovementBackend().train(dataset);
    expect(model.generate()).toEqual(["open", "search", "type", "submit"]);
  });

  it("predicts the next movement from a prefix", () => {
    const dataset: MovementDataset = {
      version: 1,
      sequences: [
        { sessionId: "a", tokens: ["open", "search", "type", "submit"] },
        { sessionId: "b", tokens: ["open", "search", "type", "submit"] },
      ],
    };
    const model = new MarkovMovementBackend().train(dataset);
    const prediction = model.predictNext(["open", "search"]);
    expect(prediction?.token).toBe("type");
    expect(prediction?.probability).toBe(1);
  });

  it("survives a serialize/restore round-trip", () => {
    const dataset: MovementDataset = {
      version: 1,
      sequences: [{ sessionId: "s", tokens: ["a", "b", "c"] }],
    };
    const backend = new MarkovMovementBackend();
    const model = backend.train(dataset);
    const restored = backend.restore(JSON.parse(JSON.stringify(model.toJSON())));
    expect(restored.generate()).toEqual(model.generate());
  });

  it("is deterministic on ties (stable lexical tie-break)", () => {
    const dataset: MovementDataset = {
      version: 1,
      sequences: [
        { sessionId: "a", tokens: ["start", "zebra"] },
        { sessionId: "b", tokens: ["start", "alpha"] },
      ],
    };
    const model = new MarkovMovementBackend().train(dataset);
    // Both continuations have count 1; the smaller token wins deterministically.
    expect(model.predictNext(["start"])?.token).toBe("alpha");
  });
});

describe("generalization", () => {
  it("predicts held-out but related movements above chance via backoff", () => {
    const skills = [
      { name: "compose", steps: ["open-app", "tap-compose", "type-body", "tap-send"] },
      { name: "reply", steps: ["open-app", "tap-thread", "tap-reply", "type-body", "tap-send"] },
      { name: "search", steps: ["open-app", "tap-search", "type-query", "tap-result"] },
    ];
    const train = generateSyntheticMovementSessions({ seed: 7, skills, sessionCount: 60, recombineRate: 0.25 });
    const heldOut = generateSyntheticMovementSessions({ seed: 999, skills, sessionCount: 20, recombineRate: 0.25 });

    const model = new MarkovMovementBackend(3).train({ version: 1, sequences: train });
    const result = evaluateMovementModel(model, heldOut);

    // Random next-token chance across the token vocabulary is well under 20%.
    expect(result.accuracy).toBeGreaterThan(0.7);
    expect(result.predictions).toBeGreaterThan(0);
  });

  it("synthetic generator is deterministic for a given seed", () => {
    const skills = [{ name: "a", steps: ["x", "y"] }];
    const first = generateSyntheticMovementSessions({ seed: 42, skills, sessionCount: 5 });
    const second = generateSyntheticMovementSessions({ seed: 42, skills, sessionCount: 5 });
    expect(first).toEqual(second);
  });
});
