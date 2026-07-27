import { describe, expect, it } from "vitest";
import {
  buildMovementDataset,
  evaluateMovementModel,
  movementTokenForEvent,
  MOVEMENT_END,
  NgramMovementBackend,
  type MovementDataset,
  type MovementReplayEvent,
} from "./movement-model.js";

function actionEvent(trajectoryId: string, tool: string, summary: string, ts: number): MovementReplayEvent {
  return { kind: "action", trajectoryId, tool, summary, ts };
}

function observationEvent(trajectoryId: string, source: string, summary: string, ts: number): MovementReplayEvent {
  return { kind: "observation", trajectoryId, source, summary, ts };
}

describe("movement dataset", () => {
  it("tokenizes actions and observations but drops transcript events", () => {
    expect(movementTokenForEvent(actionEvent("t1", "device", "tapped Submit", 1))).toEqual({
      type: "action",
      symbol: "action:device:tapped-submit",
      summary: "tapped Submit",
      ts: 1,
    });
    expect(movementTokenForEvent(observationEvent("t1", "device", "Editor active", 2))).toMatchObject({
      type: "observation",
      symbol: "observation:device:editor-active",
    });
    expect(
      movementTokenForEvent({ kind: "transcript", ts: 3, role: "user", content: "hi" } as MovementReplayEvent),
    ).toBeUndefined();
  });

  it("groups events by trajectory and orders them by timestamp", () => {
    const dataset = buildMovementDataset({
      replays: [
        {
          trajectoryIds: ["t1", "t2"],
          events: [
            actionEvent("t1", "device", "tapped Submit", 4),
            observationEvent("t1", "device", "Form open", 1),
            actionEvent("t2", "browser", "clicked Deploy", 2),
          ],
        },
      ],
    });

    expect(dataset.sequences).toHaveLength(2);
    const t1 = dataset.sequences.find((sequence) => sequence.id === "t1")!;
    // Sorted by ts: observation (ts 1) before action (ts 4).
    expect(t1.tokens.map((token) => token.type)).toEqual(["observation", "action"]);
    expect(t1.tokens.map((token) => token.symbol)).toEqual([
      "observation:device:form-open",
      "action:device:tapped-submit",
    ]);
  });
});

describe("NgramMovementBackend — repeat", () => {
  it("reproduces a recorded movement it was trained on", () => {
    const dataset: MovementDataset = {
      version: 1,
      sequences: [
        {
          id: "login-flow",
          tokens: [
            { type: "observation", symbol: "observation:device:login-screen", summary: "login", ts: 1 },
            { type: "action", symbol: "action:device:tap-username", summary: "tap username", ts: 2 },
            { type: "action", symbol: "action:device:type-user", summary: "type user", ts: 3 },
            { type: "action", symbol: "action:device:tap-password", summary: "tap password", ts: 4 },
            { type: "action", symbol: "action:device:type-secret", summary: "type secret", ts: 5 },
            { type: "action", symbol: "action:device:tap-submit", summary: "tap submit", ts: 6 },
          ],
        },
      ],
    };

    const model = new NgramMovementBackend().train(dataset);
    // Generating from scratch reproduces the exact recorded movement.
    expect(model.generate()).toEqual(dataset.sequences[0]!.tokens.map((token) => token.symbol));
  });

  it("stops at the end sentinel rather than looping forever", () => {
    const dataset: MovementDataset = {
      version: 1,
      sequences: [
        {
          id: "single",
          tokens: [{ type: "action", symbol: "action:device:tap-ok", summary: "ok", ts: 1 }],
        },
      ],
    };
    const model = new NgramMovementBackend().train(dataset);
    const generated = model.generate([], { maxSteps: 50 });
    expect(generated).toEqual(["action:device:tap-ok"]);
    expect(generated).not.toContain(MOVEMENT_END);
  });
});

describe("NgramMovementBackend — generalize", () => {
  it("predicts a plausible next action for a novel prefix via backoff", () => {
    // Two flows share the "type into search then submit" tail. A never-seen
    // opening observation should still back off to the shared suffix behaviour.
    const dataset: MovementDataset = {
      version: 1,
      sequences: [
        {
          id: "flow-a",
          tokens: [
            { type: "observation", symbol: "observation:app:mail", summary: "mail", ts: 1 },
            { type: "action", symbol: "action:kbd:type-query", summary: "type query", ts: 2 },
            { type: "action", symbol: "action:kbd:press-enter", summary: "enter", ts: 3 },
          ],
        },
        {
          id: "flow-b",
          tokens: [
            { type: "observation", symbol: "observation:app:notes", summary: "notes", ts: 1 },
            { type: "action", symbol: "action:kbd:type-query", summary: "type query", ts: 2 },
            { type: "action", symbol: "action:kbd:press-enter", summary: "enter", ts: 3 },
          ],
        },
      ],
    };

    const model = new NgramMovementBackend(2).train(dataset, { order: 2 });

    // Novel context: an unseen app observation followed by the shared action.
    // The bigram context (type-query) is known even though the full prefix is not.
    const prediction = model.predict([
      "observation:app:calendar",
      "action:kbd:type-query",
    ]);
    expect(prediction).toBeDefined();
    expect(prediction!.symbol).toBe("action:kbd:press-enter");
    // It generalized: it matched on a shorter context than the full novel prefix.
    expect(prediction!.matchedOrder).toBeLessThanOrEqual(2);
    expect(prediction!.candidates.length).toBeGreaterThan(0);
  });
});

describe("NgramMovementBackend — serialize / load", () => {
  it("round-trips a trained model for later inference", () => {
    const dataset = buildMovementDataset({
      replays: [
        {
          trajectoryIds: ["t1"],
          events: [
            observationEvent("t1", "device", "home", 1),
            actionEvent("t1", "device", "open app", 2),
            actionEvent("t1", "device", "tap settings", 3),
          ],
        },
      ],
    });
    const original = new NgramMovementBackend().train(dataset);
    const restored = NgramMovementBackend.load(original.serialize());

    expect(restored.order).toBe(original.order);
    expect(restored.vocabulary).toEqual(original.vocabulary);
    expect(restored.generate()).toEqual(original.generate());
  });
});

describe("evaluateMovementModel", () => {
  it("scores next-token fidelity on held-out sequences", () => {
    const dataset: MovementDataset = {
      version: 1,
      sequences: [
        {
          id: "train",
          tokens: [
            { type: "observation", symbol: "observation:device:start", summary: "start", ts: 1 },
            { type: "action", symbol: "action:device:step-1", summary: "s1", ts: 2 },
            { type: "action", symbol: "action:device:step-2", summary: "s2", ts: 3 },
          ],
        },
      ],
    };
    const model = new NgramMovementBackend().train(dataset);

    // Held-out = the same deterministic movement: the model should predict it perfectly.
    const evaluation = evaluateMovementModel(model, dataset.sequences);
    expect(evaluation.accuracy).toBe(1);
    expect(evaluation.actionAccuracy).toBe(1);
    expect(evaluation.actionPredictions).toBe(2);
    expect(evaluation.predictions).toBeGreaterThan(0);
  });
});
