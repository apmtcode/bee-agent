import { describe, expect, it } from "vitest";
import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  DeterministicMarkovMovementBackend,
  MOVEMENT_STOP_TOKEN,
  MovementModelTrainingService,
  buildMovementDatasetFromReplays,
  buildMovementDatasetFromTrajectories,
  buildMovementSequence,
  evaluateMovementModel,
  movementToken,
  rolloutMovements,
  type MovementSequence,
} from "./movement-model.js";

/**
 * Synthetic movement-stream generator. Models a recurring workflow — e.g.
 * "focus a window, type, save" — as an action grammar so we can validate the
 * capture -> dataset -> train -> replay/generalize pipeline with zero real OS
 * input. Deterministic: same channels in, same events out.
 */
function syntheticReplay(sessionId: string, channels: string[], startTs = 1_000): ReplayManifest {
  const events: ReplayTimelineEvent[] = channels.map((channel, index) => ({
    kind: "action",
    ts: startTs + index * 10,
    trajectoryId: `${sessionId}-traj`,
    tool: channel,
    summary: `${channel} step ${index}`,
  }));
  return {
    version: 1,
    sessionId,
    trajectoryIds: [`${sessionId}-traj`],
    eventCount: events.length,
    events,
  };
}

const WORKFLOW = ["focus", "click", "type", "type", "save", "confirm"];

describe("movement token + sequence building", () => {
  it("derives coarse channel tokens and drops transcript events", () => {
    const events: ReplayTimelineEvent[] = [
      { kind: "transcript", ts: 5, messageId: "m1", role: "user", content: "go" },
      { kind: "observation", ts: 20, trajectoryId: "t", source: "screen", summary: "window shown" },
      { kind: "action", ts: 10, trajectoryId: "t", tool: "click", summary: "click ok" },
    ];
    const sequence = buildMovementSequence("s1", events);
    // Sorted by ts, transcript excluded.
    expect(sequence.tokens).toEqual(["action:click", "observation:screen"]);
    expect(movementToken(sequence.events[0]!)).toBe("action:click");
  });
});

describe("dataset builders", () => {
  it("builds a vocabulary from replay manifests and skips empty sequences", () => {
    const dataset = buildMovementDatasetFromReplays([
      syntheticReplay("a", WORKFLOW),
      syntheticReplay("b", []),
    ]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.vocabulary).toEqual(
      ["action:click", "action:confirm", "action:focus", "action:save", "action:type"].sort(),
    );
  });

  it("builds sequences from trajectory spans, ordering by timestamp", () => {
    const trajectory: TrajectorySpan = {
      id: "traj-1",
      sessionId: "sess-1",
      createdAt: "2026-07-04T00:00:00.000Z",
      captureTier: "full",
      observations: [{ kind: "observation", source: "screen", summary: "shown", ts: 5 }],
      actions: [
        { kind: "action", tool: "click", summary: "c", ts: 10 },
        { kind: "action", tool: "type", summary: "t", ts: 20 },
      ],
    };
    const dataset = buildMovementDatasetFromTrajectories([trajectory]);
    expect(dataset.sequences[0]!.tokens).toEqual([
      "observation:screen",
      "action:click",
      "action:type",
    ]);
  });
});

describe("DeterministicMarkovMovementBackend", () => {
  const backend = new DeterministicMarkovMovementBackend();

  it("replays a recorded movement with perfect fidelity", () => {
    const dataset = buildMovementDatasetFromReplays([syntheticReplay("a", WORKFLOW)]);
    const model = backend.train(dataset, { maxOrder: 3 });
    const steps = rolloutMovements(backend, model, { maxSteps: 20 });
    expect(steps.map((step) => step.token)).toEqual(WORKFLOW.map((c) => `action:${c}`));
  });

  it("terminates rollout at the learned stop sentinel, never emitting it", () => {
    const dataset = buildMovementDatasetFromReplays([syntheticReplay("a", WORKFLOW)]);
    const model = backend.train(dataset, { maxOrder: 2 });
    const steps = rolloutMovements(backend, model, { maxSteps: 100 });
    expect(steps.every((step) => step.token !== MOVEMENT_STOP_TOKEN)).toBe(true);
    expect(steps).toHaveLength(WORKFLOW.length);
  });

  it("continues a partial seed toward the recorded completion", () => {
    const dataset = buildMovementDatasetFromReplays([syntheticReplay("a", WORKFLOW)]);
    const model = backend.train(dataset, { maxOrder: 3 });
    const steps = rolloutMovements(backend, model, {
      seed: ["action:focus", "action:click"],
      maxSteps: 20,
    });
    expect(steps.map((step) => step.token)).toEqual([
      "action:type",
      "action:type",
      "action:save",
      "action:confirm",
    ]);
  });

  it("is deterministic and round-trips through serialization", () => {
    const dataset = buildMovementDatasetFromReplays([
      syntheticReplay("a", WORKFLOW),
      syntheticReplay("b", ["focus", "click", "type", "save"]),
    ]);
    const model = backend.train(dataset, { maxOrder: 3 });
    const clone = backend.deserialize(backend.serialize(model));
    expect(clone).toEqual(model);
    const a = backend.predictNext(model, ["action:focus"]);
    const b = backend.predictNext(clone, ["action:focus"]);
    expect(a).toEqual(b);
    expect(a.candidates.length).toBeGreaterThan(0);
  });

  it("rejects a serialized model from a different backend", () => {
    expect(() => backend.deserialize(JSON.stringify({ backend: "other" }))).toThrow(/backend/);
  });

  it("returns an empty prediction for an untrained model", () => {
    const model = backend.train({ version: 1, sequences: [], vocabulary: [] });
    const prediction = backend.predictNext(model, ["action:focus"]);
    expect(prediction.token).toBeUndefined();
    expect(prediction.candidates).toEqual([]);
  });
});

describe("generalization via backoff", () => {
  const backend = new DeterministicMarkovMovementBackend();

  it("predicts a related-but-unseen continuation by backing off to lower order", () => {
    // Two related workflows share the "type -> save" transition.
    const dataset = buildMovementDatasetFromReplays([
      syntheticReplay("a", ["focus", "type", "save"]),
      syntheticReplay("b", ["click", "type", "save"]),
    ]);
    const model = backend.train(dataset, { maxOrder: 2 });
    // Unseen high-order prefix ["scroll","type"]; order-2 context is absent, so
    // it backs off to order-1 ("type" -> "save"), the shared transition.
    const prediction = backend.predictNext(model, ["action:scroll", "action:type"]);
    expect(prediction.token).toBe("action:save");
    expect(prediction.order).toBeLessThan(2);
  });

  it("scores held-out related sequences above chance", () => {
    const train = buildMovementDatasetFromReplays([
      syntheticReplay("a", WORKFLOW),
      syntheticReplay("b", WORKFLOW),
    ]);
    const model = backend.train(train, { maxOrder: 2 });
    const heldOut: MovementSequence[] = [
      buildMovementSequence("held", syntheticReplay("held", WORKFLOW).events),
    ];
    const report = evaluateMovementModel(backend, model, heldOut);
    expect(report.accuracy).toBeGreaterThan(0.8);
  });
});

describe("MovementModelTrainingService", () => {
  it("trains, reports fidelity and generalization, and rolls out", () => {
    const service = new MovementModelTrainingService();
    const dataset = buildMovementDatasetFromReplays([
      syntheticReplay("a", WORKFLOW),
      // Distinct workflow (no conflicting shared prefix) so replay fidelity is exact.
      syntheticReplay("b", ["open", "search", "read", "close"]),
    ]);
    const heldOut = [buildMovementSequence("held", syntheticReplay("held", WORKFLOW).events)];
    const { model, serialized, report } = service.trainAndEvaluate({
      dataset,
      heldOut,
      config: { maxOrder: 3 },
      trainedAt: "2026-07-04T00:00:00.000Z",
    });

    expect(report.backend).toBe("deterministic-markov");
    expect(report.sequenceCount).toBe(2);
    // High aggregate replay fidelity; the only miss is the unavoidable start
    // tie between two distinct workflows sharing the START context.
    expect(report.trainFidelity.accuracy).toBeGreaterThan(0.9);
    // The first recorded workflow replays with perfect per-sequence fidelity.
    expect(report.trainFidelity.perSequence[0]!.accuracy).toBe(1);
    expect(report.generalization?.accuracy).toBeGreaterThan(0.8);
    expect(report.trainedAt).toBe("2026-07-04T00:00:00.000Z");
    expect(JSON.parse(serialized).backend).toBe("deterministic-markov");

    const rollout = service.rollout(model, { seed: ["action:focus"], maxSteps: 10 });
    expect(rollout[0]!.token).toBe("action:click");
  });

  it("accepts an injected custom backend (pluggability)", () => {
    const calls: string[] = [];
    const service = new MovementModelTrainingService({
      name: "stub",
      train: () => ({ trained: true }),
      predictNext: () => ({ token: "action:noop", probability: 1, order: 0, candidates: [] }),
      serialize: (model) => {
        calls.push("serialize");
        return JSON.stringify(model);
      },
      deserialize: (data) => JSON.parse(data),
    });
    const { report, serialized } = service.trainAndEvaluate({
      dataset: buildMovementDatasetFromReplays([syntheticReplay("a", WORKFLOW)]),
    });
    expect(report.backend).toBe("stub");
    expect(calls).toContain("serialize");
    expect(serialized).toBe(JSON.stringify({ trained: true }));
  });
});
