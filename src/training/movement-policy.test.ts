import { describe, expect, it } from "vitest";
import { buildTrajectorySpan, type TrajectorySpan } from "../capture/trajectory.js";
import {
  MarkovMovementBackend,
  buildContextKeys,
  createMovementBackend,
  evaluateMovementPolicy,
  extractMovementSamples,
  listMovementBackends,
  registerMovementBackend,
  rolloutMovementPolicy,
  type MovementPolicyBackend,
  type MovementSample,
} from "./movement-policy.js";

/** Build a synthetic reviewed movement trajectory from an app + gesture labels. */
function syntheticTrajectory(id: string, app: string, actions: string[]): TrajectorySpan {
  return buildTrajectorySpan({
    id,
    sessionId: `session-${id}`,
    captureTier: "full",
    observations: [{ kind: "observation", source: "device", summary: app, ts: 0 }],
    actions: actions.map((summary, index) => ({
      kind: "action",
      tool: "device",
      summary,
      ts: (index + 1) * 10,
    })),
  });
}

describe("extractMovementSamples", () => {
  it("emits ordered context->action samples with a bounded history window", () => {
    const samples = extractMovementSamples([syntheticTrajectory("t1", "editor", ["a", "b", "c"])], { order: 2 });
    expect(samples).toEqual<MovementSample[]>([
      { context: { contextTag: "editor", recentActions: [] }, action: "a" },
      { context: { contextTag: "editor", recentActions: ["a"] }, action: "b" },
      { context: { contextTag: "editor", recentActions: ["a", "b"] }, action: "c" },
    ]);
  });

  it("sorts actions by timestamp before building the sequence", () => {
    const trajectory = buildTrajectorySpan({
      id: "t2",
      sessionId: "s2",
      observations: [{ kind: "observation", source: "device", summary: "browser", ts: 0 }],
      actions: [
        { kind: "action", tool: "device", summary: "second", ts: 20 },
        { kind: "action", tool: "device", summary: "first", ts: 10 },
      ],
    });
    const samples = extractMovementSamples([trajectory]);
    expect(samples.map((sample) => sample.action)).toEqual(["first", "second"]);
  });
});

describe("MarkovMovementBackend", () => {
  const backend = new MarkovMovementBackend();

  it("memorizes and replays a recorded movement sequence exactly", () => {
    const samples = extractMovementSamples([syntheticTrajectory("t1", "editor", ["open", "select", "copy", "paste"])]);
    const policy = backend.train(samples, { order: 2 });

    const prediction = backend.predict(policy, { contextTag: "editor", recentActions: ["open", "select"] });
    expect(prediction.action).toBe("copy");
    expect(prediction.backoff).toBe("exact");
    expect(prediction.confidence).toBe(1);

    // Seeded with the recorded opening action, the greedy rollout replays the rest exactly.
    const rollout = rolloutMovementPolicy(backend, policy, { contextTag: "editor", recentActions: ["open"] }, 3);
    expect(rollout.actions).toEqual(["select", "copy", "paste"]);
  });

  it("generalizes to an unseen sequence via context-prior backoff", () => {
    // Two recordings share an app but never the exact 2-gram we query.
    const samples = extractMovementSamples([
      syntheticTrajectory("t1", "mail", ["compose", "type", "send"]),
      syntheticTrajectory("t2", "mail", ["compose", "attach", "send"]),
    ]);
    const policy = backend.train(samples, { order: 2 });

    const prediction = backend.predict(policy, { contextTag: "mail", recentActions: ["reply", "review"] });
    // "reply,review" was never seen, so it backs off to the mail context prior.
    expect(prediction.backoff).toBe("context");
    expect(prediction.action).toBeDefined();
    // "send" and "compose" are the most frequent mail actions (2 each); tie broken lexically.
    expect(prediction.action).toBe("compose");
  });

  it("falls back to a global prior across context tags", () => {
    const samples = extractMovementSamples([syntheticTrajectory("t1", "editor", ["z", "z", "z", "y"])]);
    const policy = backend.train(samples, { order: 1 });
    const prediction = backend.predict(policy, { contextTag: "never-seen-app", recentActions: [] });
    expect(prediction.backoff).toBe("global");
    expect(prediction.action).toBe("z"); // most frequent action overall
  });

  it("returns an empty prediction for an empty policy", () => {
    const policy = backend.train([]);
    const prediction = backend.predict(policy, { contextTag: "x", recentActions: [] });
    expect(prediction).toEqual({ action: undefined, confidence: 0, backoff: "none", candidates: [] });
  });

  it("breaks ties deterministically by label and ranks all candidates", () => {
    const samples: MovementSample[] = [
      { context: { contextTag: "app", recentActions: [] }, action: "beta" },
      { context: { contextTag: "app", recentActions: [] }, action: "alpha" },
    ];
    const policy = backend.train(samples, { order: 0 });
    const prediction = backend.predict(policy, { contextTag: "app", recentActions: [] });
    expect(prediction.action).toBe("alpha");
    expect(prediction.candidates.map((candidate) => candidate.action)).toEqual(["alpha", "beta"]);
    expect(prediction.confidence).toBeCloseTo(0.5);
  });

  it("does not confuse separator characters inside labels", () => {
    const samples: MovementSample[] = [
      { context: { contextTag: "a|b", recentActions: ["x,y"] }, action: "hit" },
    ];
    const policy = backend.train(samples, { order: 1 });
    // A different context that would collide under naive concatenation must not match.
    const collide = backend.predict(policy, { contextTag: "a", recentActions: ["b|x", "y"] });
    expect(collide.backoff).not.toBe("exact");
    const exact = backend.predict(policy, { contextTag: "a|b", recentActions: ["x,y"] });
    expect(exact.action).toBe("hit");
    expect(exact.backoff).toBe("exact");
  });
});

describe("buildContextKeys", () => {
  it("orders keys longest-match-first ending in context then global", () => {
    const keys = buildContextKeys({ contextTag: "app", recentActions: ["a", "b", "c"] }, 2);
    expect(keys.map((entry) => entry.level)).toEqual(["exact", "recent", "context", "global"]);
    expect(keys[keys.length - 1].key).toBe("*");
  });
});

describe("backend registry", () => {
  it("exposes the built-in markov backend and instantiates it by name", () => {
    expect(listMovementBackends()).toContain("markov");
    const backend = createMovementBackend("markov");
    expect(backend.name).toBe("markov");
  });

  it("throws on unknown backends", () => {
    expect(() => createMovementBackend("does-not-exist")).toThrow(/unknown movement backend/);
  });

  it("supports registering a custom pluggable backend", () => {
    const custom: MovementPolicyBackend = {
      name: "always-tap",
      train: (samples) => ({
        version: 1,
        backend: "always-tap",
        order: 0,
        actions: ["tap"],
        sampleCount: samples.length,
        transitions: {},
      }),
      predict: () => ({ action: "tap", confidence: 1, backoff: "context", candidates: [{ action: "tap", probability: 1 }] }),
    };
    registerMovementBackend("always-tap", () => custom);
    expect(createMovementBackend("always-tap").predict({} as never, { contextTag: "x", recentActions: [] }).action).toBe("tap");
  });
});

describe("evaluateMovementPolicy", () => {
  it("scores held-out accuracy and separates memorized vs generalized hits", () => {
    const backend = new MarkovMovementBackend();
    const train = extractMovementSamples([
      syntheticTrajectory("t1", "mail", ["compose", "type", "send"]),
      syntheticTrajectory("t2", "mail", ["compose", "type", "send"]),
    ]);
    const policy = backend.train(train, { order: 2 });

    // Held-out: one exact-match continuation (memorized) + one unseen-context query (generalized).
    const heldOut: MovementSample[] = [
      { context: { contextTag: "mail", recentActions: ["compose", "type"] }, action: "send" },
      { context: { contextTag: "mail", recentActions: ["draft"] }, action: "compose" },
    ];
    const evaluation = evaluateMovementPolicy(backend, policy, heldOut);
    expect(evaluation.total).toBe(2);
    expect(evaluation.correct).toBe(2);
    expect(evaluation.accuracy).toBe(1);
    expect(evaluation.memorized).toBe(1);
    expect(evaluation.generalized).toBe(1);
  });

  it("returns zero accuracy for an empty held-out set", () => {
    const backend = new MarkovMovementBackend();
    const policy = backend.train([]);
    expect(evaluateMovementPolicy(backend, policy, [])).toEqual({
      total: 0,
      correct: 0,
      accuracy: 0,
      memorized: 0,
      generalized: 0,
    });
  });
});
