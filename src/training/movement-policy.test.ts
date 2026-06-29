import { describe, expect, it } from "vitest";
import {
  deserializeMovementPolicy,
  movementTrajectoryFromReplay,
  predictMovement,
  replayMovement,
  serializeMovementPolicy,
  trainMovementPolicy,
  type MovementTrajectory,
} from "./movement-policy.js";

function trajectory(id: string, events: MovementTrajectory["events"]): MovementTrajectory {
  return { id, events };
}

const deployTrajectory = trajectory("deploy", [
  { kind: "observation", source: "browser", summary: "open the deploy dashboard" },
  { kind: "action", tool: "browser", summary: "click deploy" },
  { kind: "action", tool: "browser", summary: "confirm rollout" },
  { kind: "action", tool: "browser", summary: "watch logs" },
]);

describe("trainMovementPolicy", () => {
  it("learns vocabulary, transitions, and observation cues", () => {
    const policy = trainMovementPolicy([deployTrajectory]);
    expect(policy.trainedTrajectories).toBe(1);
    expect(policy.trainedActions).toBe(3);
    expect(policy.vocabulary).toContain("browser␟click deploy");
    // observation words become generalization cues
    expect(Object.keys(policy.observationCues)).toEqual(
      expect.arrayContaining(["w␟deploy", "src␟browser"]),
    );
  });

  it("ignores trajectories with no actions", () => {
    const policy = trainMovementPolicy([
      trajectory("noop", [{ kind: "observation", source: "browser", summary: "nothing happened" }]),
    ]);
    expect(policy.trainedTrajectories).toBe(0);
    expect(policy.vocabulary).toEqual([]);
  });
});

describe("replayMovement", () => {
  it("repeats a recorded movement sequence from its seed observation", () => {
    const policy = trainMovementPolicy([deployTrajectory]);
    const replayed = replayMovement(policy, {
      observation: { source: "browser", summary: "open the deploy dashboard" },
    });
    expect(replayed).toEqual([
      { tool: "browser", summary: "click deploy" },
      { tool: "browser", summary: "confirm rollout" },
      { tool: "browser", summary: "watch logs" },
    ]);
  });

  it("terminates at the learned end marker instead of looping forever", () => {
    const policy = trainMovementPolicy([deployTrajectory]);
    const replayed = replayMovement(policy, {
      observation: { source: "browser", summary: "open the deploy dashboard" },
      maxSteps: 100,
    });
    expect(replayed.length).toBe(3);
  });
});

describe("predictMovement generalization", () => {
  it("picks the right action for a new but related observation", () => {
    const policy = trainMovementPolicy([
      trajectory("a", [
        { kind: "observation", source: "browser", summary: "open the deploy dashboard" },
        { kind: "action", tool: "browser", summary: "click deploy" },
      ]),
      trajectory("b", [
        { kind: "observation", source: "browser", summary: "open the deploy panel now" },
        { kind: "action", tool: "browser", summary: "click deploy" },
      ]),
      trajectory("c", [
        { kind: "observation", source: "editor", summary: "open the settings file" },
        { kind: "action", tool: "editor", summary: "edit settings" },
      ]),
    ]);

    // never-seen wording, but shares "deploy"/"open" + browser source
    const prediction = predictMovement(policy, {
      observation: { source: "browser", summary: "please open deploy right away" },
    });
    expect(prediction?.action).toEqual({ tool: "browser", summary: "click deploy" });
    expect(prediction?.end).toBe(false);
  });

  it("returns undefined for an empty model", () => {
    const policy = trainMovementPolicy([]);
    expect(predictMovement(policy, { observation: { source: "x", summary: "y" } })).toBeUndefined();
  });

  it("is deterministic across repeated calls", () => {
    const policy = trainMovementPolicy([deployTrajectory]);
    const context = { observation: { source: "browser", summary: "open the deploy dashboard" } };
    const first = predictMovement(policy, context);
    const second = predictMovement(policy, context);
    expect(first).toEqual(second);
  });
});

describe("serialization", () => {
  it("round-trips a policy through JSON", () => {
    const policy = trainMovementPolicy([deployTrajectory]);
    const restored = deserializeMovementPolicy(serializeMovementPolicy(policy));
    expect(restored).toEqual(policy);
    expect(
      replayMovement(restored, { observation: { source: "browser", summary: "open the deploy dashboard" } }),
    ).toHaveLength(3);
  });

  it("rejects an unknown version", () => {
    expect(() => deserializeMovementPolicy(JSON.stringify({ version: 2 }))).toThrow(/version/);
  });
});

describe("movementTrajectoryFromReplay", () => {
  it("maps replay-manifest events to movement events and drops transcript", () => {
    const result = movementTrajectoryFromReplay("traj-1", [
      { kind: "transcript", ts: 1, messageId: "m1", role: "user", content: "go" },
      { kind: "observation", ts: 2, trajectoryId: "traj-1", source: "browser", summary: "open deploy" },
      { kind: "action", ts: 3, trajectoryId: "traj-1", tool: "browser", summary: "click deploy" },
    ]);
    expect(result.events).toEqual([
      { kind: "observation", source: "browser", summary: "open deploy" },
      { kind: "action", tool: "browser", summary: "click deploy" },
    ]);
  });
});
