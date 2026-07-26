import { describe, expect, it } from "vitest";
import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  buildMovementDatasetFromTrajectories,
  createMarkovMovementBackend,
  evaluateNextActionAccuracy,
  generateMovementSequence,
  MarkovMovementBackend,
  replayEventsToMovementSequence,
  trajectorySpanToTokens,
  type LocalMovementModelBackend,
  type MovementDataset,
  type MovementToken,
} from "./movement-model.js";

function trajectory(id: string, actions: string[], observations: string[] = []): TrajectorySpan {
  let ts = 0;
  return {
    id,
    sessionId: `session-${id}`,
    createdAt: "2026-07-26T00:00:00.000Z",
    captureTier: "operator",
    observations: observations.map((source) => ({ kind: "observation", source, summary: source, ts: ts++ })),
    actions: actions.map((tool) => ({ kind: "action", tool, summary: tool, ts: ts++ })),
  };
}

function actionTokens(labels: string[]): MovementToken[] {
  return labels.map((label) => ({ kind: "action", label }));
}

describe("movement dataset builders", () => {
  it("orders observation/action tokens by timestamp", () => {
    const span: TrajectorySpan = {
      id: "t1",
      sessionId: "s1",
      createdAt: "2026-07-26T00:00:00.000Z",
      captureTier: "operator",
      observations: [{ kind: "observation", source: "screen", summary: "", ts: 10 }],
      actions: [
        { kind: "action", tool: "click", summary: "", ts: 5 },
        { kind: "action", tool: "type", summary: "", ts: 15 },
      ],
    };
    expect(trajectorySpanToTokens(span)).toEqual([
      { kind: "action", label: "click" },
      { kind: "observation", label: "screen" },
      { kind: "action", label: "type" },
    ]);
  });

  it("builds a dataset from trajectories and from replay events equivalently", () => {
    const dataset = buildMovementDatasetFromTrajectories([trajectory("t1", ["click", "type"], ["screen"])]);
    expect(dataset.version).toBe(1);
    expect(dataset.sequences).toHaveLength(1);

    const fromReplay = replayEventsToMovementSequence("t1", [
      { kind: "observation", ts: 0, trajectoryId: "t1", source: "screen", summary: "" },
      { kind: "action", ts: 1, trajectoryId: "t1", tool: "click", summary: "" },
      { kind: "action", ts: 2, trajectoryId: "t1", tool: "type", summary: "" },
    ]);
    expect(fromReplay.tokens.filter((token) => token.kind === "action")).toEqual(actionTokens(["click", "type"]));
  });
});

describe("MarkovMovementBackend", () => {
  it("replays a recorded movement exactly from its seed", () => {
    const backend = createMarkovMovementBackend();
    const dataset = buildMovementDatasetFromTrajectories([
      trajectory("t1", ["open", "select", "copy", "paste", "save"]),
    ]);
    const model = backend.train(dataset, { order: 2 });

    const steps = generateMovementSequence(model, actionTokens(["open"]), { maxSteps: 10 });
    expect(steps.map((step) => step.action)).toEqual(["select", "copy", "paste", "save"]);
    // A recorded, unambiguous sequence should be predicted with full confidence.
    expect(steps.every((step) => step.confidence === 1)).toBe(true);
  });

  it("is deterministic and breaks ties by action name", () => {
    const backend = new MarkovMovementBackend();
    // From context "a": one path goes to "z", one to "b" — equal counts.
    const dataset: MovementDataset = {
      version: 1,
      sequences: [
        { trajectoryId: "s1", tokens: actionTokens(["a", "z"]) },
        { trajectoryId: "s2", tokens: actionTokens(["a", "b"]) },
      ],
    };
    const model = backend.train(dataset, { order: 1 });
    const prediction = model.predictNext(actionTokens(["a"]));
    expect(prediction.action).toBe("b"); // tie broken lexically
    expect(prediction.confidence).toBeCloseTo(0.5);
    expect(prediction.candidates.map((candidate) => candidate.action)).toEqual(["b", "z"]);
  });

  it("generalizes to novel contexts by backing off to lower-order statistics", () => {
    const backend = createMarkovMovementBackend();
    // "login" is always followed by "dashboard" across many flows.
    const dataset = buildMovementDatasetFromTrajectories([
      trajectory("t1", ["home", "login", "dashboard"]),
      trajectory("t2", ["settings", "login", "dashboard"]),
      trajectory("t3", ["profile", "login", "dashboard"]),
    ]);
    const model = backend.train(dataset, { order: 2 });

    // Novel 2-gram context ["search","login"] never seen during training.
    const prediction = model.predictNext(actionTokens(["search", "login"]));
    expect(prediction.action).toBe("dashboard");
    expect(prediction.backedOff).toBe(true);
    expect(prediction.contextOrderUsed).toBe(1); // fell back from order 2 to order 1
  });

  it("falls back to the global prior for a fully unseen context", () => {
    const backend = createMarkovMovementBackend();
    const dataset = buildMovementDatasetFromTrajectories([trajectory("t1", ["a", "b", "b", "b"])]);
    const model = backend.train(dataset);
    const prediction = model.predictNext(actionTokens(["totally-unseen"]));
    expect(prediction.action).toBe("b"); // most frequent action overall
    expect(prediction.contextOrderUsed).toBe(0);
    expect(prediction.backedOff).toBe(true);
  });

  it("returns an empty prediction when nothing was trained", () => {
    const model = createMarkovMovementBackend().train({ version: 1, sequences: [] });
    const prediction = model.predictNext(actionTokens(["x"]));
    expect(prediction.action).toBeUndefined();
    expect(prediction.confidence).toBe(0);
    expect(prediction.contextOrderUsed).toBe(-1);
  });

  it("round-trips through serialize/restore preserving predictions", () => {
    const backend = createMarkovMovementBackend();
    const dataset = buildMovementDatasetFromTrajectories([
      trajectory("t1", ["open", "select", "copy", "paste"]),
      trajectory("t2", ["open", "select", "delete"]),
    ]);
    const model = backend.train(dataset, { order: 2 });
    const serialized = model.serialize();

    // Serialization is a plain JSON-safe object.
    expect(JSON.parse(JSON.stringify(serialized))).toEqual(serialized);

    const restored = backend.restore(serialized);
    expect(restored.order).toBe(model.order);
    expect(restored.actionVocabulary).toEqual(model.actionVocabulary);
    for (const context of [["open"], ["open", "select"], ["nope"]]) {
      expect(restored.predictNext(actionTokens(context))).toEqual(model.predictNext(actionTokens(context)));
    }
  });
});

describe("evaluateNextActionAccuracy", () => {
  it("scores perfect accuracy on the training sequence itself", () => {
    const backend = createMarkovMovementBackend();
    const dataset = buildMovementDatasetFromTrajectories([trajectory("t1", ["a", "b", "c", "d"])]);
    const model = backend.train(dataset, { order: 2 });
    const result = evaluateNextActionAccuracy(model, dataset);
    expect(result.evaluated).toBe(4);
    expect(result.accuracy).toBe(1);
  });

  it("generalizes to a held-out related trajectory above chance", () => {
    const backend = createMarkovMovementBackend();
    const train = buildMovementDatasetFromTrajectories([
      trajectory("t1", ["open", "login", "dashboard", "logout"]),
      trajectory("t2", ["open", "login", "dashboard", "logout"]),
    ]);
    const heldOut = buildMovementDatasetFromTrajectories([trajectory("t3", ["open", "login", "dashboard", "logout"])]);
    const model = backend.train(train, { order: 2 });
    const result = evaluateNextActionAccuracy(model, heldOut);
    expect(result.accuracy).toBeGreaterThan(0.5);
  });
});

describe("LocalMovementModelBackend interface", () => {
  it("is satisfied by the Markov backend (pluggability seam)", () => {
    const backend: LocalMovementModelBackend = createMarkovMovementBackend();
    expect(backend.id).toBe("markov");
    const model = backend.train({ version: 1, sequences: [{ trajectoryId: "t", tokens: actionTokens(["a", "b"]) }] });
    expect(model.backendId).toBe("markov");
    expect(typeof model.predictNext).toBe("function");
  });
});
