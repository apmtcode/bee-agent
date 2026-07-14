import { describe, expect, it } from "vitest";
import type { ReplayTimelineEvent } from "../capture/replay.js";
import {
  MarkovMovementBackend,
  MarkovMovementModel,
  MovementModelRegistry,
  datasetFromReplays,
  defaultMovementModelRegistry,
  evaluateMovementModel,
  movementSequenceFromEvents,
  tokenizeMovementEvent,
  type MovementDataset,
} from "./movement-model.js";

function action(tool: string, ts: number): ReplayTimelineEvent {
  return { kind: "action", ts, trajectoryId: "t", tool, summary: tool };
}

function observation(source: string, ts: number): ReplayTimelineEvent {
  return { kind: "observation", ts, trajectoryId: "t", source, summary: source };
}

describe("tokenizeMovementEvent", () => {
  it("collapses each event kind into a coarse token", () => {
    expect(tokenizeMovementEvent(action("mouse.move", 1))).toBe("action:mouse.move");
    expect(tokenizeMovementEvent(observation("screen", 1))).toBe("observation:screen");
    expect(
      tokenizeMovementEvent({ kind: "transcript", ts: 1, messageId: "m", role: "user", content: "hi" }),
    ).toBe("transcript:user");
  });
});

describe("MarkovMovementBackend", () => {
  const backend = new MarkovMovementBackend();

  it("reproduces a recorded movement verbatim (objective c)", async () => {
    const sequence = movementSequenceFromEvents("rec", [
      observation("screen", 1),
      action("mouse.move", 2),
      action("mouse.click", 3),
      action("keyboard.type", 4),
    ]);
    const model = await backend.train({ sequences: [sequence] }, { maxOrder: 2 });
    const generated = model.generate({ history: [] });
    expect(generated).toEqual([
      "observation:screen",
      "action:mouse.move",
      "action:mouse.click",
      "action:keyboard.type",
    ]);
  });

  it("terminates generation at the learned end-of-movement", async () => {
    const model = await backend.train(
      { sequences: [movementSequenceFromEvents("s", [action("a", 1), action("b", 2)])] },
      { maxOrder: 1 },
    );
    // maxSteps is generous; generation must stop on its own via the END sentinel.
    const generated = model.generate({ history: [] }, { maxSteps: 100 });
    expect(generated).toEqual(["action:a", "action:b"]);
  });

  it("generalizes to a related-but-unseen prefix via backoff (objective d)", async () => {
    // Two movements that share the "mouse.move -> mouse.click" transition.
    const dataset: MovementDataset = {
      sequences: [
        movementSequenceFromEvents("s1", [action("mouse.move", 1), action("mouse.click", 2)]),
        movementSequenceFromEvents("s2", [
          observation("menu", 1),
          action("mouse.move", 2),
          action("mouse.click", 3),
        ]),
      ],
    };
    const model = await backend.train(dataset, { maxOrder: 1 });
    // A context never seen at full order ("observation:menu" was only ever
    // followed by mouse.move, but starting fresh with mouse.move) still predicts
    // the learned click via the order-1 transition.
    const prediction = model.predictNext({ history: ["action:mouse.move"] });
    expect(prediction.token).toBe("action:mouse.click");
    expect(prediction.confidence).toBeGreaterThan(0);
  });

  it("is deterministic across identical training runs", async () => {
    const dataset: MovementDataset = {
      sequences: [movementSequenceFromEvents("s", [action("a", 1), action("b", 2), action("a", 3)])],
    };
    const first = (await backend.train(dataset, { maxOrder: 2 })).serialize();
    const second = (await backend.train(dataset, { maxOrder: 2 })).serialize();
    expect(first).toEqual(second);
  });

  it("survives a serialize -> deserialize round-trip", async () => {
    const dataset: MovementDataset = {
      sequences: [movementSequenceFromEvents("s", [action("a", 1), action("b", 2)])],
    };
    const model = await backend.train(dataset, { maxOrder: 2 });
    const restored = MarkovMovementModel.deserialize(model.serialize());
    expect(restored.serialize()).toEqual(model.serialize());
    expect(restored.generate({ history: [] })).toEqual(model.generate({ history: [] }));
  });
});

describe("evaluateMovementModel", () => {
  it("scores perfect fidelity on the trained movement", async () => {
    const backend = new MarkovMovementBackend();
    const sequence = movementSequenceFromEvents("s", [
      action("mouse.move", 1),
      action("mouse.click", 2),
      action("keyboard.type", 3),
    ]);
    const model = await backend.train({ sequences: [sequence] }, { maxOrder: 2 });
    const evaluation = evaluateMovementModel(model, { sequences: [sequence] });
    expect(evaluation.accuracy).toBe(1);
    expect(evaluation.predictedSteps).toBe(sequence.tokens.length + 1); // +1 terminal END step
    expect(evaluation.perSequence[0]?.correct).toBe(evaluation.perSequence[0]?.steps);
  });

  it("measures partial generalization on a held-out sequence", async () => {
    const backend = new MarkovMovementBackend();
    const train: MovementDataset = {
      sequences: [
        movementSequenceFromEvents("a", [action("mouse.move", 1), action("mouse.click", 2)]),
        movementSequenceFromEvents("b", [action("mouse.move", 1), action("mouse.click", 2)]),
      ],
    };
    const model = await backend.train(train, { maxOrder: 1 });
    const heldOut: MovementDataset = {
      sequences: [movementSequenceFromEvents("held", [action("mouse.move", 1), action("mouse.click", 2)])],
    };
    const evaluation = evaluateMovementModel(model, heldOut);
    // The transition was learned, so the model predicts the click; accuracy is high
    // but need not be perfect (the terminal END and first-token prior may differ).
    expect(evaluation.accuracy).toBeGreaterThan(0.5);
    expect(evaluation.sequenceCount).toBe(1);
  });
});

describe("MovementModelRegistry", () => {
  it("resolves the default backend and rejects unknown ones", () => {
    const registry = defaultMovementModelRegistry();
    expect(registry.list()).toContain("markov");
    expect(registry.has("markov")).toBe(true);
    expect(registry.get("markov")).toBeInstanceOf(MarkovMovementBackend);
    expect(() => registry.get("mlx")).toThrow(/unknown movement-model backend/);
  });

  it("accepts a pluggable custom backend", async () => {
    const custom = {
      name: "always-idle",
      async train() {
        return {
          backend: "always-idle",
          predictNext: () => ({ token: "action:idle", confidence: 1, order: 0, candidates: [] }),
          generate: () => ["action:idle"],
          serialize: () => ({ version: 1 as const, backend: "always-idle", maxOrder: 0, counts: {} }),
        };
      },
    };
    const registry = new MovementModelRegistry().register(custom);
    const model = await registry.get("always-idle").train({ sequences: [] });
    expect(model.generate({ history: [] })).toEqual(["action:idle"]);
  });
});

describe("datasetFromReplays", () => {
  it("builds sequences from replay manifests", () => {
    const dataset = datasetFromReplays([
      { sessionId: "session-1", events: [action("a", 1), observation("b", 2)] },
    ]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]).toEqual({ id: "session-1", tokens: ["action:a", "observation:b"] });
  });
});
