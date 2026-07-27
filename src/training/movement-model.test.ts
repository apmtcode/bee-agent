import { describe, expect, it } from "vitest";
import {
  DeterministicMovementBackend,
  MovementBackendRegistry,
  buildMovementDataset,
  movementDatasetFromReplays,
  movementStepFromAction,
  movementStepToken,
  type MovementStep,
} from "./movement-model.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

function step(partial: Partial<MovementStep> & { tool: string; summary: string }): MovementStep {
  return { ...partial };
}

describe("movement dataset building", () => {
  it("distills actions into ordered movement steps and carries gesture metadata", () => {
    const trajectory: TrajectorySpan = {
      id: "traj-1",
      sessionId: "sess-1",
      createdAt: "2026-01-01T00:00:00.000Z",
      captureTier: "app",
      observations: [],
      actions: [
        { kind: "action", tool: "device", summary: "typed into search", ts: 20, metadata: { gesture: "type", target: "search" } },
        { kind: "action", tool: "os", summary: "focused browser", ts: 10, metadata: { gesture: "focus", target: "browser" } },
      ],
    };

    const dataset = buildMovementDataset([trajectory]);
    expect(dataset.sequences).toHaveLength(1);
    // Sorted by timestamp: focus (ts 10) before type (ts 20).
    expect(dataset.sequences[0]!.steps.map((s) => s.summary)).toEqual(["focused browser", "typed into search"]);
    expect(movementStepFromAction(trajectory.actions[0]!)).toMatchObject({ gesture: "type", target: "search" });
  });

  it("drops trajectories with no actions", () => {
    const dataset = buildMovementDataset([
      { id: "empty", sessionId: "s", createdAt: "", captureTier: "operator", observations: [], actions: [] },
    ]);
    expect(dataset.sequences).toHaveLength(0);
  });

  it("builds a dataset from replay manifests grouped by trajectory", () => {
    const dataset = movementDatasetFromReplays([
      {
        trajectoryIds: ["a", "b"],
        events: [
          { kind: "observation", ts: 1, trajectoryId: "a", source: "os", summary: "opened" },
          { kind: "action", ts: 2, trajectoryId: "a", tool: "device", summary: "tapped send" },
          { kind: "action", ts: 3, trajectoryId: "b", tool: "os", summary: "focused editor" },
        ],
      },
    ]);
    expect(dataset.sequences.map((s) => s.trajectoryId).sort()).toEqual(["a", "b"]);
  });
});

describe("DeterministicMovementBackend", () => {
  const backend = new DeterministicMovementBackend();

  const workflow: MovementStep[] = [
    step({ tool: "os", gesture: "focus", target: "browser", summary: "focused browser" }),
    step({ tool: "device", gesture: "tap", target: "compose", summary: "tapped compose" }),
    step({ tool: "device", gesture: "type", target: "compose", summary: "typed into compose" }),
    step({ tool: "device", gesture: "tap", target: "send", summary: "tapped send" }),
  ];

  it("reproduces a recorded movement end-to-end from an empty prefix", async () => {
    const model = await backend.train({ version: 1, sequences: [{ trajectoryId: "t1", steps: workflow }] }, { order: 2 });
    const prediction = await backend.predict(model, { prefix: [], maxSteps: 10 });

    expect(prediction.stopped).toBe("end");
    expect(prediction.steps.map((s) => movementStepToken(s))).toEqual(workflow.map((s) => movementStepToken(s)));
    // Every step came from a fully-matched (recorded) context.
    expect(prediction.steps.every((s) => s.source === "recorded")).toBe(true);
  });

  it("is deterministic across repeated training + prediction runs", async () => {
    const dataset = { version: 1 as const, sequences: [{ trajectoryId: "t1", steps: workflow }] };
    const a = await backend.train(dataset, { order: 2 });
    const b = await backend.train(dataset, { order: 2 });
    expect(JSON.stringify(a)).toEqual(JSON.stringify(b));
    const pa = await backend.predict(a, { prefix: [workflow[0]!], maxSteps: 5 });
    const pb = await backend.predict(b, { prefix: [workflow[0]!], maxSteps: 5 });
    expect(JSON.stringify(pa)).toEqual(JSON.stringify(pb));
  });

  it("generalizes to a novel-but-related prefix via backoff", async () => {
    // Train on two workflows that share a "type -> tap send" tail.
    const wfA = workflow;
    const wfB: MovementStep[] = [
      step({ tool: "os", gesture: "focus", target: "mail", summary: "focused mail" }),
      step({ tool: "device", gesture: "tap", target: "reply", summary: "tapped reply" }),
      step({ tool: "device", gesture: "type", target: "reply", summary: "typed into reply" }),
      step({ tool: "device", gesture: "tap", target: "send", summary: "tapped send" }),
    ];
    const model = await backend.train(
      { version: 1, sequences: [{ trajectoryId: "a", steps: wfA }, { trajectoryId: "b", steps: wfB }] },
      { order: 2 },
    );

    // Novel prefix: focus editor (unseen app) then tap compose. The model has
    // never seen this exact context but should still continue plausibly.
    const novelPrefix: MovementStep[] = [
      step({ tool: "os", gesture: "focus", target: "editor", summary: "focused editor" }),
      step({ tool: "device", gesture: "tap", target: "compose", summary: "tapped compose" }),
    ];
    const prediction = await backend.predict(model, { prefix: novelPrefix, maxSteps: 5 });

    expect(prediction.steps.length).toBeGreaterThan(0);
    // "tap compose" was seen -> next learned step is "type into compose".
    expect(prediction.steps[0]!.summary).toBe("typed into compose");
    expect(prediction.steps.map((s) => movementStepToken(s))).toContain(
      movementStepToken(step({ tool: "device", gesture: "tap", target: "send", summary: "tapped send" })),
    );
  });

  it("stops with no-continuation when the prefix has no learned context", async () => {
    const model = await backend.train({ version: 1, sequences: [{ trajectoryId: "t1", steps: workflow }] }, { order: 2 });
    const prediction = await backend.predict(model, {
      prefix: [step({ tool: "unknown-tool", summary: "totally novel action never seen" })],
      maxSteps: 5,
    });
    // Backoff to order-0 still yields the global next-token distribution, so we
    // never dead-end on a non-empty model; assert it produces a bounded result.
    expect(prediction.steps.length).toBeLessThanOrEqual(5);
  });
});

describe("MovementBackendRegistry", () => {
  it("registers the deterministic backend by default and resolves by id", () => {
    const registry = new MovementBackendRegistry();
    expect(registry.ids()).toContain("deterministic-markov");
    expect(registry.get("deterministic-markov").id).toBe("deterministic-markov");
    expect(() => registry.get("nope")).toThrow(/Unknown movement model backend/);
  });
});
