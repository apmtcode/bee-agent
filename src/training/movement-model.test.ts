import { describe, expect, it } from "vitest";
import { buildMovementDataset } from "./movement-dataset.js";
import {
  DeterministicSequenceModelBackend,
  MovementModelTrainer,
  MovementPolicy,
} from "./movement-model.js";
import { syntheticDeviceTrajectory } from "./movement-test-utils.js";

function composeTrajectory(id: string, recipient: string) {
  return syntheticDeviceTrajectory({
    id,
    sessionId: `session-${id}`,
    appId: "mail",
    platform: "macos",
    screenTitle: "Inbox",
    goal: "compose a message",
    gestures: [
      { kind: "tap", target: "Compose" },
      { kind: "type", target: "To", valueSummary: recipient },
    ],
    approved: true,
  });
}

describe("DeterministicSequenceModelBackend", () => {
  it("repeats a recorded movement verbatim for an exact context match", () => {
    const dataset = buildMovementDataset([composeTrajectory("t1", "alice@example.com")]);
    const policy = new MovementModelTrainer().train(dataset);

    const prediction = policy.infer({
      appId: "mail",
      platform: "macos",
      screenTitle: "Inbox",
      goal: "compose a message",
    });

    expect(prediction.mode).toBe("repeat");
    expect(prediction.confidence).toBe(1);
    expect(prediction.matchedExampleId).toBe("t1");
    expect(prediction.steps).toHaveLength(2);
    expect(prediction.steps[1]).toMatchObject({ target: "To", valueSummary: "alice@example.com" });
  });

  it("induces a variable slot and generalizes to an unseen target value", () => {
    const dataset = buildMovementDataset([
      composeTrajectory("t1", "alice@example.com"),
      composeTrajectory("t2", "bob@example.com"),
    ]);
    const policy = new MovementModelTrainer().train(dataset);

    // valueSummary varied across the two recorded movements -> it is a slot.
    expect(policy.model.slots.some((slot) => slot.field === "valueSummary" && slot.stepIndex === 1)).toBe(true);

    const prediction = policy.infer({
      appId: "mail",
      platform: "macos",
      screenTitle: "Inbox",
      goal: "compose a message",
      slots: { valueSummary: "carol@example.com" },
    });

    expect(prediction.mode).toBe("generalize");
    expect(prediction.filledSlots.valueSummary).toBe("carol@example.com");
    expect(prediction.steps[1]).toMatchObject({ gesture: "type", target: "To", valueSummary: "carol@example.com" });
    // The first step, whose fields never varied, is left untouched.
    expect(prediction.steps[0]).toMatchObject({ gesture: "tap", target: "Compose" });
  });

  it("does not substitute a field that never varied in training", () => {
    // Only one recorded movement -> no slots induced -> target stays fixed even if provided.
    const dataset = buildMovementDataset([composeTrajectory("t1", "alice@example.com")]);
    const policy = new MovementModelTrainer().train(dataset);

    const prediction = policy.infer({
      appId: "mail",
      platform: "macos",
      screenTitle: "Drafts", // different screen -> generalize, not repeat
      goal: "compose a message",
      slots: { target: "Reply" },
    });

    expect(prediction.mode).toBe("generalize");
    expect(prediction.filledSlots.target).toBeUndefined();
    expect(prediction.steps[0]).toMatchObject({ target: "Compose" });
  });

  it("returns unknown for an incompatible app context", () => {
    const dataset = buildMovementDataset([composeTrajectory("t1", "alice@example.com")]);
    const policy = new MovementModelTrainer().train(dataset);

    const prediction = policy.infer({ appId: "terminal", platform: "macos", goal: "compose a message" });

    expect(prediction.mode).toBe("unknown");
    expect(prediction.steps).toHaveLength(0);
    expect(prediction.confidence).toBe(0);
  });

  it("returns unknown when similarity falls below the generalize threshold", () => {
    const dataset = buildMovementDataset([composeTrajectory("t1", "alice@example.com")]);
    const policy = new MovementModelTrainer().train(dataset);

    const prediction = policy.infer({
      appId: "mail",
      platform: "macos",
      screenTitle: "Settings Preferences Accounts Signatures",
      goal: "configure unrelated preferences panel entirely",
    });

    expect(prediction.mode).toBe("unknown");
  });

  it("produces identical predictions after JSON serialization round-trip", () => {
    const dataset = buildMovementDataset([
      composeTrajectory("t1", "alice@example.com"),
      composeTrajectory("t2", "bob@example.com"),
    ]);
    const trained = new MovementModelTrainer().train(dataset);

    const rehydrated = MovementPolicy.fromSerialized(
      JSON.parse(JSON.stringify(trained.model)) as typeof trained.model,
    );

    const query = {
      appId: "mail",
      platform: "macos",
      screenTitle: "Inbox",
      goal: "compose a message",
      slots: { valueSummary: "dave@example.com" },
    };
    expect(rehydrated.infer(query)).toEqual(trained.infer(query));
  });

  it("exposes the backend name on the serialized model", () => {
    const dataset = buildMovementDataset([composeTrajectory("t1", "alice@example.com")]);
    const model = new DeterministicSequenceModelBackend().train(dataset);
    expect(model.backend).toBe("deterministic-sequence");
  });
});
