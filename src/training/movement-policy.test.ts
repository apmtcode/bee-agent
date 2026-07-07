import { describe, expect, it } from "vitest";
import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  MarkovMovementBackend,
  movementDatasetFromSpans,
  movementStepFromAction,
  type MovementDataset,
  type MovementSequence,
  type MovementStep,
} from "./movement-policy.js";

function seq(id: string, steps: MovementStep[]): MovementSequence {
  return { id, steps };
}

describe("MarkovMovementBackend", () => {
  it("repeats a single recorded sequence deterministically via argmax rollout", () => {
    const dataset: MovementDataset = {
      sequences: [
        seq("s1", [
          { gesture: "shortcut", target: "command-palette" },
          { gesture: "type", target: "search" },
          { gesture: "tap", target: "result" },
        ]),
      ],
    };
    const model = new MarkovMovementBackend().train(dataset);
    const rollout = model.generate();
    expect(rollout).toEqual(dataset.sequences[0]!.steps);
  });

  it("reproduces the dominant path when sequences diverge", () => {
    // Two recordings share the start, then one path is recorded twice → dominant.
    const common: MovementStep = { gesture: "shortcut", target: "menu" };
    const dataset: MovementDataset = {
      sequences: [
        seq("a", [common, { gesture: "tap", target: "save" }]),
        seq("b", [common, { gesture: "tap", target: "save" }]),
        seq("c", [common, { gesture: "tap", target: "quit" }]),
      ],
    };
    const model = new MarkovMovementBackend().train(dataset);
    const rollout = model.generate();
    expect(rollout).toEqual([common, { gesture: "tap", target: "save" }]);
  });

  it("generalizes to a novel prefix by backing off to lower-order context", () => {
    // Train so that (order-2) context of the query prefix was never seen, but
    // the last single step ("swipe left") always leads to "tap next" (order 1).
    const dataset: MovementDataset = {
      sequences: [
        seq("a", [
          { gesture: "tap", target: "gallery" },
          { gesture: "swipe", direction: "left" },
          { gesture: "tap", target: "next" },
        ]),
        seq("b", [
          { gesture: "tap", target: "album" },
          { gesture: "swipe", direction: "left" },
          { gesture: "tap", target: "next" },
        ]),
      ],
    };
    const model = new MarkovMovementBackend().train(dataset, { maxOrder: 2 });

    // Novel prefix: a starting target never recorded, followed by "swipe left".
    const novelHistory: MovementStep[] = [
      { gesture: "tap", target: "settings-thumbnail" },
      { gesture: "swipe", direction: "left" },
    ];
    const prediction = model.predictNext(novelHistory);
    expect(prediction).toBeDefined();
    expect(prediction!.step).toEqual({ gesture: "tap", target: "next" });
    // The order-2 context (thumbnail→swipe) was unseen, so it backed off to 1.
    expect(prediction!.order).toBe(1);
    expect(prediction!.probability).toBeGreaterThan(0);
  });

  it("falls back to the unigram marginal (order 0) for a fully novel context", () => {
    const dataset: MovementDataset = {
      sequences: [
        seq("a", [{ gesture: "tap", target: "x" }, { gesture: "tap", target: "x" }]),
        seq("b", [{ gesture: "tap", target: "x" }]),
      ],
    };
    const model = new MarkovMovementBackend().train(dataset, { maxOrder: 2 });
    const prediction = model.predictNext([{ gesture: "shortcut", target: "unheard-of" }]);
    expect(prediction).toBeDefined();
    expect(prediction!.order).toBe(0);
    expect(prediction!.step).toEqual({ gesture: "tap", target: "x" });
  });

  it("terminates rollout at the learned END and respects maxSteps", () => {
    const dataset: MovementDataset = {
      // A self-looping movement: without the END terminator this would run away.
      sequences: [seq("loop", [{ gesture: "scroll", direction: "down" }])],
    };
    const model = new MarkovMovementBackend().train(dataset, { maxOrder: 1 });
    const rollout = model.generate({ maxSteps: 10 });
    expect(rollout).toEqual([{ gesture: "scroll", direction: "down" }]);
  });

  it("round-trips through serialize/loadModel with identical predictions", () => {
    const dataset: MovementDataset = {
      sequences: [
        seq("a", [
          { gesture: "shortcut", target: "menu" },
          { gesture: "type", target: "note" },
          { gesture: "tap", target: "done" },
        ]),
      ],
    };
    const backend = new MarkovMovementBackend();
    const model = backend.train(dataset);
    const serialized = model.serialize();
    const json = JSON.parse(JSON.stringify(serialized));
    const restored = backend.loadModel(json);
    expect(restored.generate()).toEqual(model.generate());
    expect(serialized.backendId).toBe("markov");
    expect(serialized.version).toBe(1);
  });
});

describe("movementStepFromAction / movementDatasetFromSpans", () => {
  it("derives steps from device, browser, and generic actions", () => {
    expect(
      movementStepFromAction({
        kind: "action",
        tool: "device",
        summary: "swiped left",
        ts: 1,
        metadata: { gesture: "swipe", direction: "left", target: "carousel" },
      }),
    ).toEqual({ gesture: "swipe", direction: "left", target: "carousel" });

    expect(
      movementStepFromAction({
        kind: "action",
        tool: "browser",
        summary: "clicked submit",
        ts: 1,
        metadata: { action: "click", target: "submit" },
      }),
    ).toEqual({ gesture: "click", target: "submit" });

    // No gesture/action metadata → falls back to the tool name.
    expect(
      movementStepFromAction({ kind: "action", tool: "shell", summary: "ran", ts: 1 }),
    ).toEqual({ gesture: "shell" });
  });

  it("builds an ordered dataset from spans and trains end-to-end", () => {
    const span: TrajectorySpan = {
      id: "span-1",
      sessionId: "sess",
      createdAt: "2026-07-07T00:00:00.000Z",
      captureTier: "full",
      observations: [],
      actions: [
        { kind: "action", tool: "device", summary: "tap", ts: 30, metadata: { gesture: "tap", target: "z" } },
        { kind: "action", tool: "device", summary: "shortcut", ts: 10, metadata: { gesture: "shortcut", target: "menu" } },
        { kind: "action", tool: "device", summary: "type", ts: 20, metadata: { gesture: "type", target: "q" } },
      ],
    };
    const dataset = movementDatasetFromSpans([span]);
    // Actions are re-ordered by timestamp (10, 20, 30).
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]!.steps).toEqual([
      { gesture: "shortcut", target: "menu" },
      { gesture: "type", target: "q" },
      { gesture: "tap", target: "z" },
    ]);

    const model = new MarkovMovementBackend().train(dataset);
    expect(model.generate()).toEqual(dataset.sequences[0]!.steps);
  });

  it("skips spans with no derivable movement", () => {
    const span: TrajectorySpan = {
      id: "empty",
      sessionId: "sess",
      createdAt: "2026-07-07T00:00:00.000Z",
      captureTier: "operator",
      observations: [],
      actions: [],
    };
    expect(movementDatasetFromSpans([span]).sequences).toHaveLength(0);
  });
});
