import { describe, expect, it } from "vitest";
import {
  buildMovementSequencesFromReplays,
  deserializeMovementModel,
  MARKOV_BACKEND_ID,
  MarkovMovementBackend,
  movementFeatureKey,
  movementShapeKey,
  parseMovementFeatureFromSummary,
  type MovementSequence,
} from "./movement-model.js";

const backend = new MarkovMovementBackend();

function seq(id: string, ...features: MovementSequence["features"]): MovementSequence {
  return { id, features };
}

describe("MarkovMovementBackend", () => {
  it("learns and repeats an exact recorded movement sequence", () => {
    const dataset = [
      seq(
        "t1",
        { tool: "device", gesture: "tap", target: "mail" },
        { tool: "device", gesture: "type", target: "subject" },
        { tool: "device", gesture: "tap", target: "send" },
      ),
    ];
    const model = backend.train(dataset);
    expect(model.backendId).toBe(MARKOV_BACKEND_ID);

    const next = model.predictNext([{ tool: "device", gesture: "tap", target: "mail" }]);
    expect(next.strategy).toBe("exact");
    expect(next.feature).toEqual({ tool: "device", gesture: "type", target: "subject" });
    expect(next.confidence).toBe(1);
  });

  it("autoregressively replays the full recorded movement", () => {
    const dataset = [
      seq(
        "t1",
        { tool: "device", gesture: "tap", target: "mail" },
        { tool: "device", gesture: "type", target: "subject" },
        { tool: "device", gesture: "tap", target: "send" },
      ),
    ];
    const model = backend.train(dataset);
    const replay = model.predictSequence([{ tool: "device", gesture: "tap", target: "mail" }], 2);
    expect(replay.map((p) => p.feature.target)).toEqual(["subject", "send"]);
  });

  it("generalizes to a related-but-unseen target via shape backoff", () => {
    // Trained: tapping any app is followed by a "type" gesture.
    const dataset = [
      seq("a", { tool: "device", gesture: "tap", target: "mail" }, { tool: "device", gesture: "type", target: "body" }),
      seq("b", { tool: "device", gesture: "tap", target: "notes" }, { tool: "device", gesture: "type", target: "note" }),
    ];
    const model = backend.train(dataset);
    // "calendar" was never tapped in training, so no exact context exists.
    const next = model.predictNext([{ tool: "device", gesture: "tap", target: "calendar" }]);
    expect(next.strategy).toBe("shape");
    expect(next.feature.gesture).toBe("type");
  });

  it("falls back to the global unigram when context is unknown", () => {
    const dataset = [
      seq("a", { tool: "device", gesture: "scroll", direction: "down" }, { tool: "device", gesture: "scroll", direction: "down" }),
    ];
    const model = backend.train(dataset);
    const next = model.predictNext([{ tool: "browser", gesture: "click", target: "x" }]);
    expect(next.strategy).toBe("unigram");
    expect(next.feature.gesture).toBe("scroll");
  });

  it("returns an empty prediction for an empty dataset", () => {
    const model = backend.train([]);
    const next = model.predictNext([{ tool: "device", gesture: "tap" }]);
    expect(next.strategy).toBe("empty");
    expect(next.confidence).toBe(0);
  });

  it("prefers the longest matching context (higher order wins)", () => {
    // After [tap mail, type body] the next is always "send"; but after just
    // [type body] alone the model has seen both "send" and "save". The order-2
    // context must win.
    const dataset = [
      seq(
        "a",
        { tool: "device", gesture: "tap", target: "mail" },
        { tool: "device", gesture: "type", target: "body" },
        { tool: "device", gesture: "tap", target: "send" },
      ),
      seq(
        "b",
        { tool: "device", gesture: "tap", target: "chat" },
        { tool: "device", gesture: "type", target: "body" },
        { tool: "device", gesture: "tap", target: "save" },
      ),
      seq(
        "c",
        { tool: "device", gesture: "tap", target: "chat" },
        { tool: "device", gesture: "type", target: "body" },
        { tool: "device", gesture: "tap", target: "save" },
      ),
    ];
    const model = backend.train(dataset, { maxOrder: 3 });
    const next = model.predictNext([
      { tool: "device", gesture: "tap", target: "mail" },
      { tool: "device", gesture: "type", target: "body" },
    ]);
    expect(next.order).toBe(2);
    expect(next.feature.target).toBe("send");
  });
});

describe("serialization", () => {
  it("round-trips a trained model deterministically", () => {
    const dataset = [
      seq(
        "t1",
        { tool: "device", gesture: "tap", target: "mail" },
        { tool: "device", gesture: "type", target: "subject" },
        { tool: "device", gesture: "tap", target: "send" },
      ),
    ];
    const model = backend.train(dataset);
    const serialized = model.serialize();
    // Serialization is stable (sorted keys), so JSON is reproducible.
    expect(JSON.stringify(serialized)).toBe(JSON.stringify(model.serialize()));

    const restored = deserializeMovementModel(serialized);
    const context = [{ tool: "device", gesture: "tap", target: "mail" }];
    expect(restored.predictNext(context)).toEqual(model.predictNext(context));
  });
});

describe("feature keying", () => {
  it("distinguishes exact vs shape keys", () => {
    const a = { tool: "device", gesture: "tap", target: "send" } as const;
    const b = { tool: "device", gesture: "tap", target: "save" } as const;
    expect(movementFeatureKey(a)).not.toBe(movementFeatureKey(b));
    expect(movementShapeKey(a)).toBe(movementShapeKey(b));
  });
});

describe("parseMovementFeatureFromSummary", () => {
  it("recovers gesture, direction, and target from adapter summaries", () => {
    expect(parseMovementFeatureFromSummary("device", "tapped Send button")).toEqual({
      tool: "device",
      gesture: "tap",
      target: "send button",
    });
    expect(parseMovementFeatureFromSummary("device", "swiped down")).toEqual({
      tool: "device",
      gesture: "swipe",
      direction: "down",
    });
    expect(parseMovementFeatureFromSummary("device", "typed into subject")).toEqual({
      tool: "device",
      gesture: "type",
      target: "subject",
    });
  });
});

describe("buildMovementSequencesFromReplays", () => {
  it("derives ordered movement sequences from replay action events", () => {
    const replays = [
      {
        sessionId: "s1",
        trajectoryIds: ["traj-1"],
        eventCount: 3,
        events: [
          { kind: "observation", ts: 5, trajectoryId: "traj-1", source: "device", summary: "mail active" },
          { kind: "action", ts: 20, trajectoryId: "traj-1", tool: "device", summary: "tapped send" },
          { kind: "action", ts: 10, trajectoryId: "traj-1", tool: "device", summary: "typed into subject" },
        ],
      },
    ];
    const sequences = buildMovementSequencesFromReplays(replays);
    expect(sequences).toHaveLength(1);
    expect(sequences[0]!.id).toBe("traj-1");
    // Sorted by ts: type (10) before tap send (20); observation dropped.
    expect(sequences[0]!.features.map((f) => f.gesture)).toEqual(["type", "tap"]);
  });

  it("drops replays with no action events", () => {
    const replays = [
      {
        sessionId: "s1",
        trajectoryIds: ["traj-1"],
        eventCount: 1,
        events: [{ kind: "observation", ts: 1 }],
      },
    ];
    expect(buildMovementSequencesFromReplays(replays)).toHaveLength(0);
  });
});
