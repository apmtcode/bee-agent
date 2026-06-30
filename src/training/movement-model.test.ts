import { describe, expect, it } from "vitest";
import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  MarkovMovementBackend,
  MOVEMENT_END_TOKEN,
  buildMovementDataset,
  buildMovementDatasetFromTrajectories,
  evaluateMovementModel,
  getMovementModelBackend,
  listMovementModelBackends,
  movementSequenceFromReplayEvents,
  movementSequenceFromTrajectory,
  normalizeMovementToken,
  registerMovementModelBackend,
  synthesizeMovementSequences,
  type MovementModel,
  type MovementModelBackend,
} from "./movement-model.js";

function makeTrajectory(id: string, actions: Array<{ tool: string; summary: string; ts: number }>): TrajectorySpan {
  return {
    id,
    sessionId: `sess-${id}`,
    createdAt: "2026-01-01T00:00:00.000Z",
    captureTier: "operator",
    observations: [],
    actions: actions.map((action) => ({ kind: "action", ...action })),
  };
}

describe("movement tokenization", () => {
  it("normalizes tool + summary into a stable token", () => {
    expect(normalizeMovementToken("  Device ", "Tapped   Send")).toBe("device:tapped send");
    expect(normalizeMovementToken("", "")).toBe("unknown:step");
  });

  it("orders trajectory actions by timestamp", () => {
    const trajectory = makeTrajectory("t1", [
      { tool: "device", summary: "send", ts: 30 },
      { tool: "device", summary: "compose", ts: 10 },
      { tool: "device", summary: "type", ts: 20 },
    ]);
    expect(movementSequenceFromTrajectory(trajectory).tokens).toEqual([
      "device:compose",
      "device:type",
      "device:send",
    ]);
  });

  it("extracts only action events from a replay timeline", () => {
    const sequence = movementSequenceFromReplayEvents("sess-1", [
      { kind: "observation", ts: 1, trajectoryId: "t1", source: "browser", summary: "opened" },
      { kind: "action", ts: 2, trajectoryId: "t1", tool: "browser", summary: "clicked deploy" },
      { kind: "transcript", ts: 3, messageId: "m1", role: "assistant", content: "done" },
    ]);
    expect(sequence.tokens).toEqual(["browser:clicked deploy"]);
  });

  it("builds a dataset with a sorted vocabulary and drops empty sequences", () => {
    const dataset = buildMovementDataset([
      { sourceId: "a", tokens: ["b:two", "a:one"] },
      { sourceId: "empty", tokens: [] },
    ]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.vocabulary).toEqual(["a:one", "b:two"]);
  });
});

describe("MarkovMovementBackend", () => {
  it("repeats a recorded movement sequence exactly (objective 2c)", () => {
    const trajectories = [
      makeTrajectory("t1", [
        { tool: "device", summary: "tap menu", ts: 1 },
        { tool: "device", summary: "tap compose", ts: 2 },
        { tool: "device", summary: "type body", ts: 3 },
        { tool: "device", summary: "tap send", ts: 4 },
      ]),
    ];
    const dataset = buildMovementDatasetFromTrajectories(trajectories);
    const model = new MarkovMovementBackend().train(dataset);
    const head = dataset.sequences[0]!.tokens[0]!;
    const replayed = [head, ...model.rollout([head])];
    expect(replayed).toEqual(dataset.sequences[0]!.tokens);
  });

  it("predicts END after the last recorded movement", () => {
    const dataset = buildMovementDataset([{ sourceId: "a", tokens: ["x:1", "x:2"] }]);
    const model = new MarkovMovementBackend().train(dataset);
    expect(model.predictNext(["x:1", "x:2"])?.token).toBe(MOVEMENT_END_TOKEN);
  });

  it("generalizes to a related held-out movement via n-gram backoff (objective 2d)", () => {
    // Train on synthetic sequences, evaluate on a disjoint held-out split that
    // shares structure. Backoff should yield non-trivial next-token accuracy.
    const all = synthesizeMovementSequences({ count: 40, seed: 7 });
    const train = all.slice(0, 30);
    const heldOut = all.slice(30);
    const model = new MarkovMovementBackend().train(buildMovementDataset(train));
    const report = evaluateMovementModel(model, heldOut);
    expect(report.sequenceCount).toBe(heldOut.length);
    expect(report.predictions).toBeGreaterThan(0);
    // A model that learned nothing would score ~1/vocab; require real signal.
    expect(report.nextTokenAccuracy).toBeGreaterThan(0.5);
  });

  it("is deterministic: same dataset trains to an identical snapshot", () => {
    const dataset = buildMovementDataset(synthesizeMovementSequences({ count: 12, seed: 99 }));
    const a = new MarkovMovementBackend().train(dataset).serialize();
    const b = new MarkovMovementBackend().train(dataset).serialize();
    expect(a).toEqual(b);
    expect(a.backendId).toBe("markov");
    expect(a.transitionCount).toBeGreaterThan(0);
  });

  it("rollout terminates at maxSteps even with cyclic transitions", () => {
    const dataset = buildMovementDataset([{ sourceId: "loop", tokens: ["a:1", "a:1", "a:1"] }]);
    const model = new MarkovMovementBackend().train(dataset, { maxOrder: 1 });
    expect(model.rollout(["a:1"], { maxSteps: 5 }).length).toBeLessThanOrEqual(5);
  });
});

describe("backend registry", () => {
  it("returns the default markov backend and lists registered backends", () => {
    expect(getMovementModelBackend().id).toBe("markov");
    expect(listMovementModelBackends()).toContain("markov");
  });

  it("supports registering a pluggable on-device backend seam", () => {
    const stub: MovementModelBackend = {
      id: "native-stub",
      train(): MovementModel {
        throw new Error("requires on-device runtime");
      },
    };
    registerMovementModelBackend(stub);
    expect(listMovementModelBackends()).toContain("native-stub");
    expect(() => getMovementModelBackend("native-stub").train(buildMovementDataset([]))).toThrow(
      /on-device runtime/,
    );
  });

  it("throws a helpful error for an unknown backend", () => {
    expect(() => getMovementModelBackend("does-not-exist")).toThrow(/unknown movement model backend/);
  });
});

describe("synthetic generator", () => {
  it("is reproducible for a given seed", () => {
    const a = synthesizeMovementSequences({ count: 5, seed: 42 });
    const b = synthesizeMovementSequences({ count: 5, seed: 42 });
    expect(a).toEqual(b);
  });

  it("differs across seeds", () => {
    const a = synthesizeMovementSequences({ count: 5, seed: 1 });
    const b = synthesizeMovementSequences({ count: 5, seed: 2 });
    expect(a).not.toEqual(b);
  });
});
