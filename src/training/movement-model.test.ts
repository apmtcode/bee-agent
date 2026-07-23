import { describe, expect, it } from "vitest";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import {
  MarkovMovementBackend,
  MovementModelTrainer,
  deriveMovementSequence,
  eventFromToken,
  familyOfToken,
  generateSyntheticMovementSequences,
  targetFamily,
  tokenOf,
  type MovementModelBackend,
  type MovementModelState,
  type MovementPrediction,
  type MovementSequence,
  type MovementToken,
} from "./movement-model.js";

describe("movement token encoding", () => {
  it("encodes and decodes movements canonically", () => {
    const token = tokenOf({ ts: 1, tool: "Mouse", gesture: "Click", target: "Button:Submit", direction: "Down" });
    expect(token).toBe(tokenOf({ ts: 9, tool: "mouse", gesture: "click", target: "button:submit", direction: "down" }));
    expect(eventFromToken(token)).toMatchObject({ tool: "mouse", gesture: "click", target: "button:submit", direction: "down" });
  });

  it("collapses instance ids into a shared target family", () => {
    expect(targetFamily("row:3")).toBe(targetFamily("row:17"));
    expect(targetFamily("button:submit-1")).toBe("button:submit");
    expect(familyOfToken(tokenOf({ ts: 0, tool: "mouse", gesture: "click", target: "row-3" }))).toBe(
      familyOfToken(tokenOf({ ts: 0, tool: "mouse", gesture: "click", target: "row-99" })),
    );
  });
});

describe("MarkovMovementBackend", () => {
  const trainer = new MovementModelTrainer(new MarkovMovementBackend());

  const seenSequence: MovementSequence = {
    id: "seen",
    events: [
      { ts: 0, tool: "window", gesture: "shortcut", target: "app:open" },
      { ts: 1, tool: "mouse", gesture: "move", target: "panel:list" },
      { ts: 2, tool: "mouse", gesture: "click", target: "row:1" },
      { ts: 3, tool: "keyboard", gesture: "type", target: "field:name" },
      { ts: 4, tool: "mouse", gesture: "click", target: "button:save" },
    ],
  };

  it("repeats recorded movements with full replay fidelity", async () => {
    const state = await trainer.train([seenSequence], { order: 3 });
    const report = await trainer.evaluateReplayFidelity(state, seenSequence);
    expect(report.exactRate).toBe(1);
    expect(report.coverage).toBe(1);
  });

  it("replays a continuation from a seed prefix", async () => {
    const state = await trainer.train([seenSequence], { order: 3 });
    const produced = await trainer.replay(state, seenSequence.events.slice(0, 2), 3);
    expect(produced.map((event) => event.target)).toEqual(["row:1", "field:name", "button:save"]);
  });

  it("is deterministic across retrainings", async () => {
    const a = await trainer.train([seenSequence], { order: 3 });
    const b = await trainer.train([seenSequence], { order: 3 });
    expect(JSON.stringify(a)).toEqual(JSON.stringify(b));
  });

  it("round-trips a persisted model state", async () => {
    const state = await trainer.train([seenSequence], { order: 3 });
    const restored = JSON.parse(JSON.stringify(state)) as MovementModelState;
    const prediction = await new MarkovMovementBackend().predict(restored, [
      tokenOf(seenSequence.events[0]!),
      tokenOf(seenSequence.events[1]!),
    ]);
    expect(prediction?.event.target).toBe("row:1");
  });

  it("generalizes to new but related movements via target-family back-off", async () => {
    // Train on families of related sequences, hold out one variant per family.
    const all = generateSyntheticMovementSequences({ families: 4, variantsPerFamily: 4, stepsPerSequence: 6, seed: 7 });
    const heldOut = all.filter((sequence) => sequence.id.endsWith("variant0"));
    const trainSet = all.filter((sequence) => !sequence.id.endsWith("variant0"));
    const state = await trainer.train(trainSet, { order: 3, enableFamilyBackoff: true });

    let familySum = 0;
    for (const sequence of heldOut) {
      const report = await trainer.evaluateGeneralization(state, sequence);
      familySum += report.familyRate;
      // Held-out variants use unseen concrete targets, so exact match is weak
      // but the model should still recover the correct movement family.
      expect(report.familyRate).toBeGreaterThan(report.exactRate - 1e-9);
    }
    expect(familySum / heldOut.length).toBeGreaterThanOrEqual(0.9);
  });

  it("returns undefined when it has no basis to predict", async () => {
    const state = await trainer.train([seenSequence], { order: 3 });
    const prediction = await new MarkovMovementBackend().predict(state, [
      tokenOf({ ts: 0, tool: "gamepad", gesture: "rumble", target: "unknown:thing" }),
    ]);
    expect(prediction).toBeUndefined();
  });
});

describe("dataset derivation", () => {
  it("derives an ordered movement sequence from a captured trajectory", () => {
    const trajectory = buildTrajectorySpan({
      id: "traj-1",
      sessionId: "sess-1",
      actions: [
        { kind: "action", tool: "device", summary: "clicked save", ts: 20, metadata: { gesture: "tap", target: "button:save" } },
        { kind: "action", tool: "device", summary: "opened list", ts: 10, metadata: { gesture: "swipe", direction: "down" } },
      ],
    });
    const sequence = deriveMovementSequence(trajectory);
    expect(sequence.events.map((event) => event.ts)).toEqual([10, 20]);
    expect(sequence.events[0]).toMatchObject({ tool: "device", gesture: "swipe", direction: "down" });
    expect(sequence.events[1]).toMatchObject({ tool: "device", gesture: "tap", target: "button:save" });
  });
});

describe("pluggable backend seam", () => {
  it("drives the trainer through a custom backend implementation", async () => {
    const calls: string[] = [];
    const fixedBackend: MovementModelBackend = {
      id: "fixed-mock",
      async train(): Promise<MovementModelState> {
        calls.push("train");
        return {
          backendId: "fixed-mock",
          version: 1,
          order: 1,
          familyBackoff: false,
          vocabulary: [],
          transitions: {},
          familyTransitions: {},
          trainedSequences: 0,
          trainedEvents: 0,
        };
      },
      async predict(_state: MovementModelState, context: MovementToken[]): Promise<MovementPrediction | undefined> {
        calls.push("predict");
        return { token: context[0] ?? "-", event: eventFromToken(context[0] ?? "-"), confidence: 1, backoffOrder: 1, source: "exact" };
      },
    };
    const trainer = new MovementModelTrainer(fixedBackend);
    expect(trainer.backendId).toBe("fixed-mock");
    const state = await trainer.train([]);
    const prediction = await trainer.predictNext(state, [{ ts: 0, tool: "mouse", gesture: "click", target: "a" }]);
    expect(prediction?.event.tool).toBe("mouse");
    expect(calls).toEqual(["train", "predict"]);
  });
});
