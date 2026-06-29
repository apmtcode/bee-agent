import { describe, expect, it } from "vitest";
import { buildReplayManifest } from "../capture/replay.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import {
  MovementBackendRegistry,
  buildMovementDataset,
  buildMovementDatasetFromTrajectories,
  type MovementDataset,
} from "./movement-backend.js";
import { MOCK_MOVEMENT_BACKEND_ID, MockMovementBackend } from "./mock-movement-backend.js";

function syntheticTrajectory(id: string, steps: Array<{ obs: string; action: string }>) {
  let ts = 0;
  const observations = [] as Parameters<typeof buildTrajectorySpan>[0]["observations"];
  const actions = [] as Parameters<typeof buildTrajectorySpan>[0]["actions"];
  for (const step of steps) {
    ts += 10;
    observations!.push({ kind: "observation", source: "device", summary: step.obs, ts });
    ts += 5;
    actions!.push({ kind: "action", tool: "device", summary: step.action, ts });
  }
  return buildTrajectorySpan({ id, sessionId: "sess-1", observations, actions });
}

describe("buildMovementDataset", () => {
  it("groups actions by trajectory, orders by ts, and attaches preceding observation context", () => {
    const trajectory = buildTrajectorySpan({
      id: "traj-1",
      sessionId: "sess-1",
      observations: [
        { kind: "observation", source: "device", summary: "Mail compose open", ts: 10 },
        { kind: "observation", source: "device", summary: "recipient field focused", ts: 30 },
      ],
      actions: [
        { kind: "action", tool: "device", summary: "tapped Compose", ts: 20 },
        { kind: "action", tool: "device", summary: "typed recipient", ts: 40 },
      ],
    });
    const replay = buildReplayManifest({ sessionId: "sess-1", transcript: [], trajectories: [trajectory] });

    const dataset = buildMovementDataset([replay]);

    expect(dataset.sequenceCount).toBe(1);
    expect(dataset.stepCount).toBe(2);
    expect(dataset.sequences[0].steps).toEqual([
      { ts: 20, tool: "device", summary: "tapped Compose", context: "Mail compose open" },
      { ts: 40, tool: "device", summary: "typed recipient", context: "recipient field focused" },
    ]);
  });

  it("matches the trajectory-derived dataset for the same spans", () => {
    const trajectory = syntheticTrajectory("traj-a", [
      { obs: "Settings open", action: "tapped Wi-Fi" },
      { obs: "Wi-Fi list shown", action: "tapped network" },
    ]);
    const replay = buildReplayManifest({ sessionId: "sess-1", transcript: [], trajectories: [trajectory] });

    const fromReplay = buildMovementDataset([replay]);
    const fromTrajectories = buildMovementDatasetFromTrajectories([trajectory]);

    expect(fromTrajectories).toEqual(fromReplay);
  });
});

describe("MovementBackendRegistry", () => {
  it("registers, looks up, and rejects duplicate/unknown backends", () => {
    const registry = new MovementBackendRegistry();
    const backend = new MockMovementBackend();
    registry.register(backend);

    expect(registry.has(MOCK_MOVEMENT_BACKEND_ID)).toBe(true);
    expect(registry.require(MOCK_MOVEMENT_BACKEND_ID)).toBe(backend);
    expect(registry.list().map((entry) => entry.id)).toEqual([MOCK_MOVEMENT_BACKEND_ID]);
    expect(() => registry.register(new MockMovementBackend())).toThrow(/already registered/);
    expect(() => registry.require("nope")).toThrow(/unknown movement backend/);
  });
});

describe("MockMovementBackend", () => {
  const dataset: MovementDataset = buildMovementDatasetFromTrajectories([
    syntheticTrajectory("login-1", [
      { obs: "Login screen shown", action: "tapped username field" },
      { obs: "username field focused", action: "typed username" },
      { obs: "password field focused", action: "tapped Sign In" },
    ]),
    syntheticTrajectory("login-2", [
      { obs: "Login screen shown", action: "tapped username field" },
      { obs: "username field focused", action: "typed username" },
      { obs: "password field focused", action: "tapped Sign In" },
    ]),
    syntheticTrajectory("search-1", [
      { obs: "Search page open", action: "tapped search box" },
      { obs: "search box focused", action: "typed query" },
    ]),
  ]);

  it("replays a recorded trajectory exactly", async () => {
    const model = await new MockMovementBackend().train(dataset);
    const steps = model.replay("login-1");
    expect(steps?.map((step) => step.summary)).toEqual([
      "tapped username field",
      "typed username",
      "tapped Sign In",
    ]);
    expect(model.replay("missing")).toBeUndefined();
  });

  it("predicts the next step from the previous action via the transition model", async () => {
    const model = await new MockMovementBackend().train(dataset);
    const prediction = model.predictNext({ lastSummary: "tapped username field" });
    expect(prediction?.step.summary).toBe("typed username");
    expect(prediction?.source).toBe("transition");
    // Both login trajectories agree, so confidence is full.
    expect(prediction?.confidence).toBe(1);
  });

  it("predicts a first step from a goal context when no prior action is known", async () => {
    const model = await new MockMovementBackend().train(dataset);
    const prediction = model.predictNext({ context: "Login screen shown" });
    expect(prediction?.step.summary).toBe("tapped username field");
    expect(prediction?.source).toBe("context");
    expect(prediction?.confidence).toBeGreaterThan(0);
  });

  it("falls back to the most common first step with no cues", async () => {
    const model = await new MockMovementBackend().train(dataset);
    const prediction = model.predictNext({});
    // "tapped username field" starts 2 of 3 sequences.
    expect(prediction?.step.summary).toBe("tapped username field");
    expect(prediction?.source).toBe("fallback");
    expect(prediction?.confidence).toBeCloseTo(2 / 3, 5);
  });

  it("generalizes to a novel-but-related context by retrieving the nearest sequence", async () => {
    const model = await new MockMovementBackend().train(dataset);
    // Never-seen phrasing, but token-overlaps the login trajectories.
    const plan = model.generate({ context: "sign in screen for login" });
    expect(plan.map((step) => step.summary)).toEqual([
      "tapped username field",
      "typed username",
      "tapped Sign In",
    ]);
    expect(model.generate({ context: "search box for query" }).map((step) => step.summary)).toEqual([
      "tapped search box",
      "typed query",
    ]);
  });

  it("respects maxSteps when generating", async () => {
    const model = await new MockMovementBackend().train(dataset);
    expect(model.generate({ context: "Login screen shown", maxSteps: 1 })).toHaveLength(1);
  });

  it("round-trips through serialize/load preserving behaviour", async () => {
    const backend = new MockMovementBackend();
    const model = await backend.train(dataset);
    const restored = backend.load(model.serialize());

    expect(restored.sequenceCount).toBe(model.sequenceCount);
    expect(restored.stepCount).toBe(model.stepCount);
    expect(restored.replay("login-1")).toEqual(model.replay("login-1"));
    expect(restored.predictNext({ lastSummary: "tapped username field" })).toEqual(
      model.predictNext({ lastSummary: "tapped username field" }),
    );
    expect(() => backend.load({ ...model.serialize(), backendId: "other" })).toThrow(/cannot load model/);
  });

  it("is deterministic across independent trainings", async () => {
    const a = await new MockMovementBackend().train(dataset);
    const b = await new MockMovementBackend().train(dataset);
    expect(a.serialize()).toEqual(b.serialize());
  });

  it("returns undefined / empty for an empty dataset", async () => {
    const model = await new MockMovementBackend().train({ version: 1, sequenceCount: 0, stepCount: 0, sequences: [] });
    expect(model.predictNext({ context: "anything" })).toBeUndefined();
    expect(model.generate({ context: "anything" })).toEqual([]);
  });
});
