import { describe, expect, it } from "vitest";
import {
  MOVEMENT_END_TOKEN,
  MarkovMovementBackend,
  evaluateMovementModel,
  movementSequenceFromReplayEvents,
  movementSequenceFromTrajectory,
  movementStepToken,
  type MovementDataset,
} from "./movement-model.js";
import {
  generateSyntheticMovementDataset,
  splitMovementDataset,
} from "./synthetic-stream.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

const linearDataset: MovementDataset = {
  sequences: [
    {
      id: "s1",
      steps: [
        { tool: "pointer", gesture: "click", target: "a", summary: "click a" },
        { tool: "keyboard", gesture: "type", target: "hello", summary: "type hello" },
        { tool: "pointer", gesture: "click", target: "submit", summary: "click submit" },
      ],
    },
    {
      id: "s2",
      steps: [
        { tool: "pointer", gesture: "click", target: "b", summary: "click b" },
        { tool: "keyboard", gesture: "type", target: "world", summary: "type world" },
        { tool: "pointer", gesture: "click", target: "ok", summary: "click ok" },
      ],
    },
  ],
};

describe("movementStepToken", () => {
  it("is structural: independent of target and summary", () => {
    expect(movementStepToken({ tool: "pointer", gesture: "click", target: "a", summary: "x" })).toBe(
      movementStepToken({ tool: "pointer", gesture: "click", target: "z", summary: "y" }),
    );
    expect(movementStepToken({ tool: "Pointer", gesture: "Click", summary: "" })).toBe("pointer:click");
    expect(movementStepToken({ tool: "browser", summary: "" })).toBe("browser:-");
  });
});

describe("MarkovMovementBackend", () => {
  it("learns and predicts the next movement structurally", async () => {
    const model = await new MarkovMovementBackend().train(linearDataset, { order: 3 });
    const prediction = model.predictNext([{ tool: "pointer", gesture: "click", target: "new", summary: "click new" }]);
    expect(prediction?.token).toBe("keyboard:type");
    expect(prediction?.step?.tool).toBe("keyboard");
    expect(prediction?.end).toBe(false);
    expect(prediction?.probability).toBeGreaterThan(0);
  });

  it("predicts an end sentinel at a natural sequence stop", async () => {
    const model = await new MarkovMovementBackend().train(linearDataset, { order: 3 });
    const full = linearDataset.sequences[0]!.steps;
    const prediction = model.predictNext(full);
    expect(prediction?.token).toBe(MOVEMENT_END_TOKEN);
    expect(prediction?.end).toBe(true);
    expect(prediction?.step).toBeUndefined();
  });

  it("rolls out a full sequence from a prefix and terminates", async () => {
    const model = await new MarkovMovementBackend().train(linearDataset, { order: 3 });
    const generated = model.generate([{ tool: "pointer", gesture: "click", target: "start", summary: "click start" }]);
    expect(generated.map(movementStepToken)).toEqual(["keyboard:type", "pointer:click"]);
  });

  it("backs off to shorter context for unseen prefixes", async () => {
    const model = await new MarkovMovementBackend().train(linearDataset, { order: 3 });
    // A never-seen leading step: order-3 context misses, backs off to order-0.
    const prediction = model.predictNext([{ tool: "scroll", gesture: "wheel", summary: "scroll" }]);
    expect(prediction).toBeDefined();
    expect(prediction?.order).toBeLessThanOrEqual(1);
  });

  it("round-trips through serialize/fromSerialized", async () => {
    const model = await new MarkovMovementBackend().train(linearDataset, { order: 2 });
    const restored = MarkovMovementBackend.fromSerialized(model.serialize());
    const context = [{ tool: "pointer", gesture: "click", target: "x", summary: "click x" }];
    expect(restored.predictNext(context)?.token).toBe(model.predictNext(context)?.token);
  });
});

describe("generalization on synthetic streams", () => {
  it("generalizes to held-out but related sequences", async () => {
    const dataset = generateSyntheticMovementDataset({ seed: 42, sequencesPerTask: 12 });
    const { train, heldOut } = splitMovementDataset(dataset, 4);
    expect(train.sequences.length).toBeGreaterThan(0);
    expect(heldOut.sequences.length).toBeGreaterThan(0);

    const model = await new MarkovMovementBackend().train(train, { order: 3 });
    const result = evaluateMovementModel(model, heldOut);
    expect(result.coverage).toBe(1);
    // Structural policy should predict most next-movements on unseen sequences.
    expect(result.accuracy).toBeGreaterThan(0.6);
  });

  it("is deterministic across identical seeds", () => {
    const a = generateSyntheticMovementDataset({ seed: 7, sequencesPerTask: 5 });
    const b = generateSyntheticMovementDataset({ seed: 7, sequencesPerTask: 5 });
    expect(a).toEqual(b);
  });
});

describe("capture bridges", () => {
  it("derives a sequence from replay action events (ignoring observations)", () => {
    const sequence = movementSequenceFromReplayEvents("r1", [
      { kind: "observation", ts: 1, trajectoryId: "t", source: "browser", summary: "opened" },
      { kind: "action", ts: 2, trajectoryId: "t", tool: "browser", summary: "clicked deploy" },
      { kind: "transcript", ts: 3, messageId: "m", role: "user", content: "go" },
      { kind: "action", ts: 4, trajectoryId: "t", tool: "keyboard", summary: "typed yes" },
    ]);
    expect(sequence.steps.map((step) => step.tool)).toEqual(["browser", "keyboard"]);
  });

  it("derives a sequence from a trajectory span using gesture metadata", () => {
    const trajectory: TrajectorySpan = {
      id: "t1",
      sessionId: "sess",
      createdAt: "2026-01-01T00:00:00.000Z",
      captureTier: "operator",
      observations: [],
      actions: [
        { kind: "action", tool: "device", summary: "tapped deploy", ts: 1, metadata: { gesture: "tap", target: "deploy" } },
      ],
    };
    const sequence = movementSequenceFromTrajectory(trajectory);
    expect(sequence.steps[0]).toMatchObject({ tool: "device", gesture: "tap", target: "deploy" });
    expect(movementStepToken(sequence.steps[0]!)).toBe("device:tap");
  });
});
