import { describe, expect, it } from "vitest";
import { buildTrajectorySpan, type TrajectoryAction } from "../capture/trajectory.js";
import {
  MarkovMovementBackend,
  MOVEMENT_END,
  loadMovementModel,
  movementTokenKey,
  tokenizeAction,
  tokenizeTrajectory,
  trainMovementModel,
  type MovementToken,
} from "./movement-model.js";

function action(tool: string, summary: string, ts: number, metadata?: Record<string, unknown>): TrajectoryAction {
  return { kind: "action", tool, summary, ts, ...(metadata ? { metadata } : {}) };
}

function seq(...tokens: Array<[string, string, string?]>): MovementToken[] {
  return tokens.map(([tool, actionName, target]) => ({
    tool,
    action: actionName,
    ...(target ? { target } : {}),
  }));
}

describe("tokenization", () => {
  it("derives gesture/target/direction from action metadata", () => {
    const token = tokenizeAction(
      action("device", "tapped send-button", 5, { gesture: "tap", target: "send-button" }),
    );
    expect(token).toEqual({ tool: "device", action: "tap", target: "send-button" });
  });

  it("falls back to the first summary word when no gesture metadata", () => {
    const token = tokenizeAction(action("os", "Focused MainWindow", 1));
    expect(token).toEqual({ tool: "os", action: "focused" });
  });

  it("orders trajectory actions by timestamp", () => {
    const span = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [
        action("device", "tapped b", 20, { gesture: "tap", target: "b" }),
        action("device", "tapped a", 10, { gesture: "tap", target: "a" }),
      ],
    });
    expect(tokenizeTrajectory(span).map((token) => token.target)).toEqual(["a", "b"]);
  });

  it("produces collision-resistant keys across distinct tokens", () => {
    const a = movementTokenKey({ tool: "device", action: "tap", target: "x" });
    const b = movementTokenKey({ tool: "device", action: "tapx", target: "" });
    expect(a).not.toEqual(b);
  });
});

describe("MarkovMovementBackend training + replay", () => {
  it("deterministically replays a single memorized sequence", async () => {
    const sequence = seq(["device", "tap", "compose"], ["device", "type", "body"], ["device", "tap", "send"]);
    const model = await new MarkovMovementBackend().train([sequence], { order: 3 });

    const generated = model.generate();
    expect(generated).toEqual(sequence);
    expect(model.stats.sequenceCount).toBe(1);
    expect(model.stats.tokenCount).toBe(3);
  });

  it("predicts the learned next token from a prefix", async () => {
    const sequence = seq(["device", "tap", "compose"], ["device", "type", "body"], ["device", "tap", "send"]);
    const model = await new MarkovMovementBackend().train([sequence], { order: 2 });

    const prediction = model.predictNext([sequence[0]!]);
    expect(prediction.token).toEqual(sequence[1]);
    expect(prediction.confidence).toBeGreaterThan(0);
    expect(prediction.backoffOrder).toBeGreaterThanOrEqual(1);
    expect(prediction.generalized).toBe(false);
  });

  it("predicts end-of-sequence after the final token", async () => {
    const sequence = seq(["device", "tap", "a"], ["device", "tap", "b"]);
    const model = await new MarkovMovementBackend().train([sequence], { order: 2 });
    const prediction = model.predictNext(sequence);
    expect(prediction.key).toBe(MOVEMENT_END);
  });

  it("is reproducible: identical datasets yield identical snapshots", async () => {
    const dataset = [seq(["device", "tap", "a"], ["device", "type", "b"])];
    const a = (await new MarkovMovementBackend().train(dataset)).toJSON();
    const b = (await new MarkovMovementBackend().train(dataset)).toJSON();
    expect(a).toEqual(b);
  });
});

describe("backoff + generalization", () => {
  it("backs off to a shorter context when the full context is unseen", async () => {
    // Two sequences share the bigram (type -> tap:save) but differ earlier.
    const s1 = seq(["device", "tap", "open"], ["device", "type", "doc"], ["device", "tap", "save"]);
    const s2 = seq(["device", "tap", "new"], ["device", "type", "doc"], ["device", "tap", "save"]);
    const model = await new MarkovMovementBackend().train([s1, s2], { order: 3 });

    // A never-seen 2-token prefix ending in the shared "type doc" should still
    // predict "tap save" via backoff.
    const novelPrefix = seq(["device", "swipe", "left"], ["device", "type", "doc"]);
    const prediction = model.predictNext(novelPrefix);
    expect(prediction.token).toEqual({ tool: "device", action: "tap", target: "save" });
  });

  it("generalizes to an unseen token via feature similarity", async () => {
    const trained = seq(["device", "tap", "send-button"], ["device", "type", "body"]);
    const model = await new MarkovMovementBackend().train([trained], { order: 2 });

    // "tap submit-button" was never seen, but is feature-similar to "tap send-button".
    const prediction = model.predictNext([{ tool: "device", action: "tap", target: "submit-button" }]);
    expect(prediction.generalized).toBe(true);
    expect(prediction.token).toEqual({ tool: "device", action: "type", target: "body" });
    expect(prediction.backoffOrder).toBe(0);
  });
});

describe("snapshot round-trip", () => {
  it("reloads a model that predicts identically", async () => {
    const dataset = [
      seq(["device", "tap", "a"], ["device", "type", "b"], ["device", "tap", "c"]),
    ];
    const model = await new MarkovMovementBackend().train(dataset, { order: 3 });
    const reloaded = loadMovementModel(model.toJSON());

    expect(reloaded.generate()).toEqual(model.generate());
    expect(reloaded.stats).toEqual(model.stats);
  });
});

describe("trainMovementModel convenience", () => {
  it("trains directly from trajectory spans and drops empty sequences", async () => {
    const withActions = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [action("device", "tapped a", 1, { gesture: "tap", target: "a" })],
    });
    const empty = buildTrajectorySpan({ id: "t2", sessionId: "s2", actions: [] });
    const model = await trainMovementModel([withActions, empty]);
    expect(model.stats.sequenceCount).toBe(1);
  });
});
