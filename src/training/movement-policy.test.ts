import { describe, expect, it } from "vitest";
import { MovementPolicyEngine, trajectoryToMovement } from "./movement-policy.js";
import { MockMovementPolicyBackend, jaccard, tokenize } from "./mock-policy-backend.js";
import type { MovementTrajectory } from "./movement-policy.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

const dataset: MovementTrajectory[] = [
  {
    id: "b-search",
    goal: "search notes for invoice",
    appId: "notes",
    steps: [
      { gesture: "tap", appId: "notes", target: "search field", ts: 0 },
      { gesture: "type", appId: "notes", target: "search field", valueSummary: "invoice", ts: 1 },
    ],
  },
  {
    id: "a-compose",
    goal: "compose message to alice",
    appId: "chat",
    steps: [
      { gesture: "tap", appId: "chat", target: "new message", ts: 0 },
      { gesture: "type", appId: "chat", target: "body", valueSummary: "hello alice", ts: 1 },
    ],
  },
];

describe("MockMovementPolicyBackend", () => {
  it("repeats the nearest recorded movement verbatim", () => {
    const model = new MockMovementPolicyBackend().fit(dataset);
    const prediction = model.predict({ goal: "search notes for invoice" });

    expect(prediction.matchedTrajectoryId).toBe("b-search");
    expect(prediction.confidence).toBe(1);
    expect(prediction.generalized).toBe(false);
    expect(prediction.steps).toEqual(dataset[0].steps);
  });

  it("generalizes to a new target/value by re-parameterizing the match", () => {
    const model = new MockMovementPolicyBackend().fit(dataset);
    const prediction = model.predict({
      goal: "search notes for receipt",
      appId: "notes",
      parameters: { valueSummary: "receipt" },
    });

    expect(prediction.matchedTrajectoryId).toBe("b-search");
    expect(prediction.generalized).toBe(true);
    // gesture sequence preserved, only the typed value substituted
    expect(prediction.steps.map((step) => step.gesture)).toEqual(["tap", "type"]);
    expect(prediction.steps[1].valueSummary).toBe("receipt");
    // the tap step (no valueSummary) is untouched
    expect(prediction.steps[0].valueSummary).toBeUndefined();
  });

  it("filters candidates by appId", () => {
    const model = new MockMovementPolicyBackend().fit(dataset);
    const prediction = model.predict({ goal: "compose something", appId: "chat" });
    expect(prediction.matchedTrajectoryId).toBe("a-compose");
  });

  it("returns an empty prediction below the similarity floor", () => {
    const model = new MockMovementPolicyBackend({ minSimilarity: 0.5 }).fit(dataset);
    const prediction = model.predict({ goal: "totally unrelated zzz" });
    expect(prediction.matchedTrajectoryId).toBeNull();
    expect(prediction.steps).toEqual([]);
    expect(prediction.confidence).toBe(0);
  });

  it("is deterministic under input reordering (id tie-break)", () => {
    const tied: MovementTrajectory[] = [
      { id: "z", goal: "open menu", appId: "app", steps: [{ gesture: "tap", appId: "app", target: "menu", ts: 0 }] },
      { id: "a", goal: "open menu", appId: "app", steps: [{ gesture: "tap", appId: "app", target: "menu", ts: 0 }] },
    ];
    const forward = new MockMovementPolicyBackend().fit(tied).predict({ goal: "open menu" });
    const reversed = new MockMovementPolicyBackend().fit([...tied].reverse()).predict({ goal: "open menu" });
    expect(forward.matchedTrajectoryId).toBe("a");
    expect(reversed.matchedTrajectoryId).toBe("a");
  });
});

describe("similarity primitives", () => {
  it("tokenizes and scores overlap", () => {
    expect(tokenize("Search Notes!")).toEqual(new Set(["search", "notes"]));
    expect(jaccard(tokenize("a b"), tokenize("a b"))).toBe(1);
    expect(jaccard(tokenize("a b"), tokenize("a c"))).toBeCloseTo(1 / 3);
    expect(jaccard(new Set(), new Set())).toBe(1);
  });
});

describe("trajectoryToMovement", () => {
  const span: TrajectorySpan = {
    id: "traj-1",
    sessionId: "s1",
    createdAt: "2026-01-01T00:00:00.000Z",
    captureTier: "app",
    observations: [
      {
        kind: "observation",
        source: "device",
        summary: "Notes active",
        ts: 0,
        metadata: { appId: "notes", appName: "Notes" },
      },
    ],
    actions: [
      {
        kind: "action",
        tool: "device",
        summary: "tapped search field",
        ts: 1,
        metadata: { appId: "notes", gesture: "tap", target: "search field" },
      },
      {
        kind: "action",
        tool: "device",
        summary: "typed into search field",
        ts: 2,
        metadata: { appId: "notes", gesture: "type", target: "search field", valueSummary: "invoice" },
      },
      // non-device action is ignored
      { kind: "action", tool: "shell", summary: "ran command", ts: 3 },
    ],
  };

  it("projects device actions into movement steps", () => {
    const movement = trajectoryToMovement(span);
    expect(movement).toBeDefined();
    expect(movement?.appId).toBe("notes");
    expect(movement?.goal).toBe("Notes active");
    expect(movement?.steps).toEqual([
      { gesture: "tap", appId: "notes", target: "search field", ts: 1 },
      { gesture: "type", appId: "notes", target: "search field", valueSummary: "invoice", ts: 2 },
    ]);
  });

  it("returns undefined when a span has no movements", () => {
    const empty: TrajectorySpan = { ...span, actions: [] };
    expect(trajectoryToMovement(empty)).toBeUndefined();
  });
});

describe("MovementPolicyEngine", () => {
  it("throws when predicting before fit", () => {
    const engine = new MovementPolicyEngine(new MockMovementPolicyBackend());
    expect(engine.fitted).toBe(false);
    expect(() => engine.predict({ goal: "x" })).toThrow(/before fit/);
  });

  it("fits from captured spans and serves predictions", async () => {
    const span: TrajectorySpan = {
      id: "traj-1",
      sessionId: "s1",
      createdAt: "2026-01-01T00:00:00.000Z",
      captureTier: "app",
      observations: [{ kind: "observation", source: "device", summary: "compose message", ts: 0 }],
      actions: [
        {
          kind: "action",
          tool: "device",
          summary: "typed body",
          ts: 1,
          metadata: { appId: "chat", gesture: "type", target: "body", valueSummary: "hi" },
        },
      ],
    };
    const engine = new MovementPolicyEngine(new MockMovementPolicyBackend());
    const model = await engine.fitFromTrajectories([span]);
    expect(model.trajectoryCount).toBe(1);
    expect(engine.fitted).toBe(true);
    const prediction = engine.predict({ goal: "compose message" });
    expect(prediction.matchedTrajectoryId).toBe("traj-1");
    expect(prediction.steps[0].gesture).toBe("type");
  });
});
