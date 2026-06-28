import { describe, expect, it } from "vitest";
import { buildReplayManifest } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  MarkovMovementBackend,
  createMovementBackend,
  datasetFromReplayEvents,
  datasetFromTrajectories,
  evaluateGeneralization,
  evaluateReplayFidelity,
  listMovementBackends,
  registerMovementBackend,
  synthesizeMovementTrajectories,
  tokenizeAction,
  type MovementDataset,
} from "./movement-model.js";

function span(id: string, actions: { tool: string; summary: string; ts: number }[]): TrajectorySpan {
  return {
    id,
    sessionId: `session-${id}`,
    createdAt: new Date(0).toISOString(),
    captureTier: "full",
    observations: [],
    actions: actions.map((a) => ({ kind: "action", tool: a.tool, summary: a.summary, ts: a.ts })),
  };
}

describe("tokenizeAction", () => {
  it("normalizes tool/summary into a stable token", () => {
    expect(tokenizeAction({ tool: "Mouse", summary: "Move To Target" })).toBe("mouse:move-to-target");
    expect(tokenizeAction({ tool: "mouse", summary: "move  to   target" })).toBe("mouse:move-to-target");
  });
});

describe("datasetFromTrajectories", () => {
  it("orders actions by timestamp and drops empty trajectories", () => {
    const dataset = datasetFromTrajectories([
      span("a", [
        { tool: "mouse", summary: "second", ts: 20 },
        { tool: "mouse", summary: "first", ts: 10 },
      ]),
      span("empty", []),
    ]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]!.tokens).toEqual(["mouse:first", "mouse:second"]);
  });

  it("matches the action tokens produced from a replay manifest", () => {
    const trajectory = span("t", [
      { tool: "keyboard", summary: "type", ts: 10 },
      { tool: "keyboard", summary: "submit", ts: 20 },
    ]);
    const replay = buildReplayManifest({ sessionId: "s", transcript: [], trajectories: [trajectory] });
    const fromReplay = datasetFromReplayEvents([replay]);
    const fromTrajectory = datasetFromTrajectories([trajectory]);
    expect(fromReplay.sequences[0]!.tokens).toEqual(fromTrajectory.sequences[0]!.tokens);
  });
});

describe("MarkovMovementBackend — repeat recorded movements", () => {
  it("reproduces a recorded sequence exactly (replay fidelity)", () => {
    const dataset = datasetFromTrajectories([
      span("rec", [
        { tool: "mouse", summary: "move", ts: 10 },
        { tool: "mouse", summary: "press", ts: 20 },
        { tool: "mouse", summary: "release", ts: 30 },
      ]),
    ]);
    const model = new MarkovMovementBackend().train(dataset);
    const result = evaluateReplayFidelity(model, dataset.sequences[0]!);
    expect(result.accuracy).toBe(1);
    expect(result.exactMatch).toBe(true);
    expect(model.generate()).toEqual(["mouse:move", "mouse:press", "mouse:release"]);
  });

  it("is deterministic across two independent trainings", () => {
    const trajectories = synthesizeMovementTrajectories({ count: 6, seed: 42 });
    const dataset = datasetFromTrajectories(trajectories);
    const a = new MarkovMovementBackend().train(dataset).generate(["mouse:move-to-target"]);
    const b = new MarkovMovementBackend().train(dataset).generate(["mouse:move-to-target"]);
    expect(a).toEqual(b);
  });
});

describe("generalization to new-but-related movements", () => {
  it("predicts continuations on held-out sequences via backoff", () => {
    const all = synthesizeMovementTrajectories({ count: 40, seed: 7 });
    const train = all.slice(0, 30);
    const heldOut = all.slice(30);
    // Held-out trajectories are distinct sequences not present in training...
    const trainIds = new Set(train.map((t) => JSON.stringify(datasetFromTrajectories([t]).sequences[0]!.tokens)));
    const novel = heldOut.filter(
      (t) => !trainIds.has(JSON.stringify(datasetFromTrajectories([t]).sequences[0]!.tokens)),
    );
    expect(novel.length).toBeGreaterThan(0);

    const model = createMovementBackend("markov").train(datasetFromTrajectories(train));
    const evalResult = evaluateGeneralization(model, datasetFromTrajectories(novel).sequences);
    // Because motifs are shared, the model generalizes well above chance even
    // though it never saw these exact trajectories.
    expect(evalResult.meanAccuracy).toBeGreaterThan(0.6);
    expect(evalResult.sequences).toBe(novel.length);
  });

  it("falls back to shorter contexts when the full context is unseen", () => {
    const dataset: MovementDataset = {
      version: 1,
      sequences: [{ id: "s", tokens: ["a", "b", "c", "d"] }],
    };
    const model = new MarkovMovementBackend().train(dataset, { order: 3 });
    // Context "x b" was never seen at order 2, but "b" -> "c" was at order 1.
    const prediction = model.predictNext(["x", "b"]);
    expect(prediction?.token).toBe("c");
    expect(prediction?.backoffOrder).toBe(1);
  });
});

describe("snapshot / restore", () => {
  it("restores an identical model from a serialized snapshot", () => {
    const dataset = datasetFromTrajectories(synthesizeMovementTrajectories({ count: 5, seed: 3 }));
    const backend = new MarkovMovementBackend();
    const model = backend.train(dataset);
    const snapshot = model.snapshot();
    // Snapshot must round-trip through JSON (persistence path).
    const restored = backend.restore(JSON.parse(JSON.stringify(snapshot)));
    expect(restored.generate(["mouse:move-to-target"])).toEqual(model.generate(["mouse:move-to-target"]));
    expect(restored.vocabulary()).toEqual(model.vocabulary());
  });
});

describe("backend registry (pluggability)", () => {
  it("lists and creates the default markov backend", () => {
    expect(listMovementBackends()).toContain("markov");
    expect(createMovementBackend().id).toBe("markov");
  });

  it("allows registering a custom backend behind the same interface", () => {
    registerMovementBackend("echo-test", () => new MarkovMovementBackend());
    expect(listMovementBackends()).toContain("echo-test");
    expect(() => createMovementBackend("echo-test")).not.toThrow();
  });

  it("throws on an unknown backend id", () => {
    expect(() => createMovementBackend("does-not-exist")).toThrow(/Unknown movement-model backend/);
  });
});

describe("synthesizeMovementTrajectories", () => {
  it("is deterministic for a fixed seed and varies by seed", () => {
    const a = synthesizeMovementTrajectories({ count: 4, seed: 99 });
    const b = synthesizeMovementTrajectories({ count: 4, seed: 99 });
    const c = synthesizeMovementTrajectories({ count: 4, seed: 100 });
    expect(JSON.stringify(a)).toBe(JSON.stringify(b));
    expect(JSON.stringify(a)).not.toBe(JSON.stringify(c));
    expect(a.every((t) => t.actions.length > 0)).toBe(true);
  });
});
