import { describe, expect, it } from "vitest";
import { buildTrajectorySpan, type TrajectoryAction } from "../capture/trajectory.js";
import {
  buildMovementDataset,
  extractMovementSequence,
  extractMovementSequenceFromReplay,
  movementToken,
  NgramMovementBackend,
} from "./movement-model.js";

function action(overrides: Partial<TrajectoryAction> & { ts: number }): TrajectoryAction {
  return {
    kind: "action",
    tool: "device",
    summary: "did something",
    ...overrides,
  };
}

function trajectory(id: string, actions: TrajectoryAction[]) {
  return buildTrajectorySpan({ id, sessionId: `sess-${id}`, actions });
}

describe("movementToken", () => {
  it("captures structure (tool:gesture:direction) without the target by default", () => {
    const token = movementToken(
      action({ ts: 1, metadata: { gesture: "swipe", direction: "down", target: "file-list" } }),
    );
    expect(token).toBe("device:swipe:down");
  });

  it("includes a slugged target when requested", () => {
    const token = movementToken(
      action({ ts: 1, metadata: { gesture: "tap", target: "Search Field!" } }),
      { includeTarget: true },
    );
    expect(token).toBe("device:tap:search-field");
  });

  it("falls back to tool:summary when no gesture metadata is present", () => {
    expect(movementToken(action({ ts: 1, tool: "browser", summary: "Clicked Deploy" }))).toBe("browser:clicked-deploy");
  });
});

describe("extractMovementSequence", () => {
  it("orders tokens by timestamp", () => {
    const sequence = extractMovementSequence(
      trajectory("t1", [
        action({ ts: 30, metadata: { gesture: "scroll", direction: "down" } }),
        action({ ts: 10, metadata: { gesture: "tap" } }),
        action({ ts: 20, metadata: { gesture: "type" } }),
      ]),
    );
    expect(sequence.tokens).toEqual(["device:tap", "device:type", "device:scroll:down"]);
  });

  it("reads action events out of a replay manifest", () => {
    const sequence = extractMovementSequenceFromReplay({
      trajectoryIds: ["t9"],
      sessionId: "s9",
      events: [
        { kind: "observation", ts: 1, trajectoryId: "t9", source: "device", summary: "open" },
        { kind: "action", ts: 3, trajectoryId: "t9", tool: "device", summary: "b" },
        { kind: "action", ts: 2, trajectoryId: "t9", tool: "device", summary: "a" },
      ],
    });
    expect(sequence.tokens).toEqual(["device:a", "device:b"]);
  });
});

describe("NgramMovementBackend", () => {
  const dataset = buildMovementDataset([
    trajectory("a", [
      action({ ts: 1, metadata: { gesture: "tap" } }),
      action({ ts: 2, metadata: { gesture: "type" } }),
      action({ ts: 3, metadata: { gesture: "shortcut" } }),
    ]),
    trajectory("b", [
      action({ ts: 1, metadata: { gesture: "tap" } }),
      action({ ts: 2, metadata: { gesture: "type" } }),
      action({ ts: 3, metadata: { gesture: "shortcut" } }),
    ]),
  ]);

  it("predicts the learned next movement from an exact context", () => {
    const model = new NgramMovementBackend().train(dataset, { order: 3 });
    const prediction = model.predict(["device:tap", "device:type"]);
    expect(prediction.token).toBe("device:shortcut");
    expect(prediction.viaBackoff).toBe(false);
    expect(prediction.confidence).toBe(1);
  });

  it("backs off to a shorter context when the exact context is unseen", () => {
    const model = new NgramMovementBackend().train(dataset, { order: 3 });
    // "device:scroll" was never observed as a context, so order-2 backs off to order-1.
    const prediction = model.predict(["device:scroll", "device:tap"]);
    expect(prediction.token).toBe("device:type");
    expect(prediction.viaBackoff).toBe(true);
    expect(prediction.order).toBe(1);
  });

  it("is deterministic and breaks ties by count then lexicographically", () => {
    const tie = buildMovementDataset([
      trajectory("x", [action({ ts: 1, metadata: { gesture: "tap" } }), action({ ts: 2, metadata: { gesture: "zoom" } })]),
      trajectory("y", [action({ ts: 1, metadata: { gesture: "tap" } }), action({ ts: 2, metadata: { gesture: "aim" } })]),
    ]);
    const model = new NgramMovementBackend().train(tie);
    // Both continuations have count 1 → lexicographic winner is "device:aim".
    expect(model.predict(["device:tap"]).token).toBe("device:aim");
    expect(model.predict(["device:tap"]).token).toBe("device:aim");
  });

  it("returns a null prediction for an empty model", () => {
    const model = new NgramMovementBackend().train({ sequences: [] });
    expect(model.predict([]).token).toBeNull();
  });

  it("round-trips through serialize/load with identical predictions", () => {
    const backend = new NgramMovementBackend();
    const model = backend.train(dataset, { order: 3 });
    const restored = backend.load(model.serialize());
    expect(restored.order).toBe(3);
    expect(restored.vocabulary()).toEqual(model.vocabulary());
    expect(restored.predict(["device:tap", "device:type"])).toEqual(
      model.predict(["device:tap", "device:type"]),
    );
  });
});
