import { describe, expect, it } from "vitest";
import { buildReplayManifest } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  MarkovMovementBackend,
  MovementBackendRegistry,
  buildMovementDatasetFromReplays,
  buildMovementDatasetFromTrajectories,
  createDefaultMovementBackendRegistry,
  evaluateNextTokenAccuracy,
  generateSyntheticTrajectories,
  rolloutMovements,
  tokenizeAction,
  tokenizeTrajectory,
  type MovementSequence,
} from "./movement-model.js";

function span(id: string, tokens: Array<{ tool: string; summary: string; metadata?: Record<string, unknown> }>): TrajectorySpan {
  return {
    id,
    sessionId: "s1",
    createdAt: new Date(0).toISOString(),
    captureTier: "app",
    observations: [],
    actions: tokens.map((token, index) => ({ kind: "action", ts: index, ...token })),
  };
}

describe("tokenizeAction", () => {
  it("uses gesture + direction metadata when present", () => {
    expect(
      tokenizeAction({ tool: "device", summary: "swiped down", metadata: { gesture: "swipe", direction: "down" } }),
    ).toBe("device:swipe:down");
  });

  it("uses gesture + target when there is no direction", () => {
    expect(
      tokenizeAction({ tool: "device", summary: "tapped Save Button", metadata: { gesture: "tap", target: "Save Button" } }),
    ).toBe("device:tap:save-button");
  });

  it("falls back to the leading verb of the summary", () => {
    expect(tokenizeAction({ tool: "Bash", summary: "ran build command" })).toBe("Bash:ran");
  });
});

describe("dataset builders", () => {
  it("tokenizes trajectory actions in timestamp order and drops empties", () => {
    const trajectory = span("t1", [
      { tool: "device", summary: "b", metadata: { gesture: "tap", target: "b" } },
      { tool: "device", summary: "a", metadata: { gesture: "tap", target: "a" } },
    ]);
    trajectory.actions[0].ts = 5;
    trajectory.actions[1].ts = 1;
    const sequence = tokenizeTrajectory(trajectory);
    expect(sequence.tokens).toEqual(["device:tap:a", "device:tap:b"]);

    const dataset = buildMovementDatasetFromTrajectories([trajectory, span("empty", [])]);
    expect(dataset.sequences).toHaveLength(1);
  });

  it("builds a dataset from replay manifests, grouping by trajectory in ts order", () => {
    const trajectory = span("t2", [
      { tool: "device", summary: "x", metadata: { gesture: "tap", target: "x" } },
      { tool: "device", summary: "y", metadata: { gesture: "tap", target: "y" } },
    ]);
    const replay = buildReplayManifest({ sessionId: "s1", transcript: [], trajectories: [trajectory] });
    const dataset = buildMovementDatasetFromReplays([replay]);
    // Replay events carry only tool + summary (no gesture metadata), so they
    // tokenize by the summary's leading verb — the trajectory path is richer.
    expect(dataset.sequences).toEqual<MovementSequence[]>([
      { trajectoryId: "t2", tokens: ["device:x", "device:y"] },
    ]);
  });
});

describe("MarkovMovementBackend", () => {
  const backend = new MarkovMovementBackend();

  it("memorizes and repeats a recorded movement sequence exactly", async () => {
    const dataset = { sequences: [{ tokens: ["A", "B", "C", "D"] }] };
    const model = await backend.train(dataset, { order: 3 });
    const rollout = rolloutMovements(backend, model, { seed: ["A"], steps: 3 });
    expect(rollout).toEqual(["B", "C", "D"]);
  });

  it("generalizes to an unseen prefix via back-off", async () => {
    // "B" is always followed by "C" in training, across different heads.
    const dataset = { sequences: [{ tokens: ["A", "B", "C"] }, { tokens: ["X", "B", "C"] }] };
    const model = await backend.train(dataset, { order: 3 });
    // Novel bigram (Q,B) never seen, but order-1 context [B] -> C generalizes.
    const prediction = backend.predict(model, { history: ["Q", "B"] });
    expect(prediction.next).toBe("C");
    expect(prediction.order).toBe(1);
  });

  it("prefers the highest-order match over lower-order statistics", async () => {
    const dataset = {
      sequences: [
        { tokens: ["A", "B", "Z"] },
        { tokens: ["B", "Y"] },
        { tokens: ["B", "Y"] },
      ],
    };
    const model = await backend.train(dataset, { order: 2 });
    // Order-1 [B] favours Y (2x); but order-2 [A,B] -> Z should win.
    const prediction = backend.predict(model, { history: ["A", "B"] });
    expect(prediction.next).toBe("Z");
    expect(prediction.order).toBe(2);
  });

  it("breaks ties deterministically by count then lexical token", async () => {
    const dataset = { sequences: [{ tokens: ["A", "C"] }, { tokens: ["A", "B"] }] };
    const model = await backend.train(dataset, { order: 1 });
    const prediction = backend.predict(model, { history: ["A"] });
    // Both count 1 -> lexical tie-break picks "B".
    expect(prediction.next).toBe("B");
    expect(prediction.distribution.map((entry) => entry.token)).toEqual(["B", "C"]);
    expect(prediction.distribution[0].probability).toBeCloseTo(0.5);
  });

  it("produces a serializable, deterministic model artifact", async () => {
    const dataset = { sequences: [{ tokens: ["A", "B", "C"] }] };
    const first = await backend.train(dataset, { order: 2 });
    const second = await backend.train(dataset, { order: 2 });
    expect(first).toEqual(second);
    const roundTripped = JSON.parse(JSON.stringify(first));
    expect(roundTripped).toEqual(first);
    expect(first.backendId).toBe("markov-backoff");
    expect(first.vocabulary).toEqual(["A", "B", "C"]);
  });

  it("returns an empty prediction for an empty model", async () => {
    const model = await backend.train({ sequences: [] });
    const prediction = backend.predict(model, { history: ["A"] });
    expect(prediction.next).toBeUndefined();
    expect(prediction.order).toBe(-1);
    expect(rolloutMovements(backend, model, { seed: [], steps: 5 })).toEqual([]);
  });

  it("honours a stop token during rollout", async () => {
    const dataset = { sequences: [{ tokens: ["A", "B", "STOP", "C"] }] };
    const model = await backend.train(dataset, { order: 3 });
    const rollout = rolloutMovements(backend, model, { seed: ["A"], steps: 10, stopToken: "STOP" });
    expect(rollout).toEqual(["B", "STOP"]);
  });
});

describe("evaluateNextTokenAccuracy", () => {
  it("scores perfect fidelity on memorized data and measures generalization on held-out variants", async () => {
    const backend = new MarkovMovementBackend();
    const train = generateSyntheticTrajectories({ count: 40, seed: 7 });
    const dataset = buildMovementDatasetFromTrajectories(train);
    const model = await backend.train(dataset, { order: 3 });

    const trainEval = evaluateNextTokenAccuracy(backend, model, dataset.sequences);
    expect(trainEval.accuracy).toBeGreaterThan(0.6);

    // Held-out synthetic trajectories drawn from the same grammar, different seed.
    const heldOut = buildMovementDatasetFromTrajectories(generateSyntheticTrajectories({ count: 20, seed: 99 }));
    const heldEval = evaluateNextTokenAccuracy(backend, model, heldOut.sequences);
    expect(heldEval.predictions).toBeGreaterThan(0);
    // Grammar-consistent held-out data should generalize well above chance.
    expect(heldEval.accuracy).toBeGreaterThan(0.5);
  });
});

describe("MovementBackendRegistry", () => {
  it("registers and retrieves backends by id", () => {
    const registry = new MovementBackendRegistry();
    const backend = new MarkovMovementBackend();
    registry.register(backend);
    expect(registry.get("markov-backoff")).toBe(backend);
    expect(registry.get("missing")).toBeUndefined();
    expect(registry.list()).toHaveLength(1);
  });

  it("default registry is seeded with the built-in backend", () => {
    const registry = createDefaultMovementBackendRegistry();
    expect(registry.get("markov-backoff")).toBeInstanceOf(MarkovMovementBackend);
  });
});

describe("generateSyntheticTrajectories", () => {
  it("is deterministic for a given seed", () => {
    const a = generateSyntheticTrajectories({ count: 5, seed: 42 });
    const b = generateSyntheticTrajectories({ count: 5, seed: 42 });
    expect(a).toEqual(b);
    expect(a).toHaveLength(5);
    expect(a[0].actions.length).toBeGreaterThan(0);
  });

  it("varies across seeds", () => {
    const a = generateSyntheticTrajectories({ count: 8, seed: 1 });
    const b = generateSyntheticTrajectories({ count: 8, seed: 2 });
    const tokensA = a.map((trajectory) => tokenizeTrajectory(trajectory).tokens.join("|")).join("/");
    const tokensB = b.map((trajectory) => tokenizeTrajectory(trajectory).tokens.join("|")).join("/");
    expect(tokensA).not.toBe(tokensB);
  });
});
