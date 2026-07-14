import { describe, expect, it } from "vitest";
import { buildTrajectorySpan, type TrajectoryAction } from "../capture/trajectory.js";
import {
  buildMovementDataset,
  createMovementPolicyBackend,
  deriveMovementToken,
  evaluateMovementGeneralization,
  movementTokenKey,
  NgramMovementPolicyBackend,
  registerMovementPolicyBackend,
  type MovementDataset,
  type MovementPolicyBackend,
  type MovementSequence,
  type MovementToken,
} from "./movement-policy.js";

function action(tool: string, gesture: string, ts: number, extra: Partial<{ target: string; direction: string }> = {}): TrajectoryAction {
  return {
    kind: "action",
    tool,
    summary: `${gesture} ${extra.target ?? ""}`.trim(),
    ts,
    metadata: { gesture, ...extra },
  };
}

function seq(source: string, tokens: MovementToken[]): MovementSequence {
  return { source, tokens };
}

describe("deriveMovementToken", () => {
  it("pulls gesture/target/direction from action metadata", () => {
    const token = deriveMovementToken(action("device", "swipe", 1, { direction: "left", target: "gallery" }));
    expect(token).toEqual({ tool: "device", gesture: "swipe", target: "gallery", direction: "left" });
  });

  it("falls back to the tool name when no gesture metadata is present", () => {
    const token = deriveMovementToken({ kind: "action", tool: "browser", summary: "click", ts: 1 });
    expect(token).toEqual({ tool: "browser", gesture: "browser" });
  });
});

describe("buildMovementDataset", () => {
  it("orders actions by timestamp and drops empty trajectories", () => {
    const withActions = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [action("device", "tap", 30, { target: "send" }), action("device", "tap", 10, { target: "compose" })],
    });
    const empty = buildTrajectorySpan({ id: "t2", sessionId: "s1" });
    const dataset = buildMovementDataset([withActions, empty]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]!.tokens.map((token) => token.target)).toEqual(["compose", "send"]);
  });
});

describe("NgramMovementPolicyBackend", () => {
  it("repeats a recorded movement sequence deterministically (objective 2c)", () => {
    const tap = { tool: "device", gesture: "tap", target: "compose" };
    const type = { tool: "device", gesture: "type", target: "body" };
    const send = { tool: "device", gesture: "tap", target: "send" };
    const dataset: MovementDataset = { sequences: [seq("s1", [tap, type, send])] };

    const model = new NgramMovementPolicyBackend().train(dataset, { order: 2 });
    const rollout = model.generate([tap], 2);
    expect(rollout).toEqual([type, send]);

    const prediction = model.predictNext([tap]);
    expect(prediction?.token).toEqual(type);
    expect(prediction?.backoff).toBe(false);
    expect(prediction?.probability).toBe(1);
  });

  it("generalizes to a new-but-related context via backoff (objective 2d)", () => {
    const openApp = { tool: "device", gesture: "tap", target: "mail-icon" };
    const swipe = { tool: "device", gesture: "swipe", target: "inbox", direction: "down" };
    const archive = { tool: "device", gesture: "tap", target: "archive" };
    // Two trajectories: a swipe on the inbox is always followed by archiving.
    const dataset: MovementDataset = {
      sequences: [seq("s1", [openApp, swipe, archive]), seq("s2", [swipe, archive])],
    };
    const model = new NgramMovementPolicyBackend().train(dataset, { order: 2 });

    // Held-out context never seen as a full n-gram: a different opener then swipe.
    const novelOpener = { tool: "device", gesture: "tap", target: "notification" };
    const prediction = model.predictNext([novelOpener, swipe]);
    expect(prediction?.token).toEqual(archive);
    expect(prediction?.backoff).toBe(true);
    expect(prediction?.contextOrder).toBe(1);
  });

  it("returns undefined for an empty untrained model", () => {
    const model = new NgramMovementPolicyBackend().train({ sequences: [] });
    expect(model.predictNext([{ tool: "device", gesture: "tap" }])).toBeUndefined();
    expect(model.generate([], 3)).toEqual([]);
  });

  it("ranks candidates by frequency with deterministic tie-breaking", () => {
    const root = { tool: "device", gesture: "tap", target: "home" };
    const a = { tool: "device", gesture: "tap", target: "a" };
    const b = { tool: "device", gesture: "tap", target: "b" };
    const dataset: MovementDataset = {
      sequences: [seq("s1", [root, a]), seq("s2", [root, a]), seq("s3", [root, b])],
    };
    const model = new NgramMovementPolicyBackend().train(dataset, { order: 1 });
    const prediction = model.predictNext([root]);
    expect(prediction?.token).toEqual(a);
    expect(prediction?.candidates.map((candidate) => candidate.token.target)).toEqual(["a", "b"]);
    expect(prediction?.candidates[0]!.count).toBe(2);
  });

  it("round-trips through serialize/restore with identical predictions", () => {
    const root = { tool: "device", gesture: "tap", target: "home" };
    const next = { tool: "device", gesture: "swipe", target: "feed", direction: "up" };
    const dataset: MovementDataset = { sequences: [seq("s1", [root, next])] };
    const backend = new NgramMovementPolicyBackend();
    const model = backend.train(dataset, { order: 2 });
    const snapshot = model.serialize();
    const restored = backend.restore(snapshot);

    expect(restored.predictNext([root])?.token).toEqual(next);
    expect(restored.serialize()).toEqual(snapshot);
    expect(snapshot.observedTokens).toBe(2);
    expect(snapshot.observedSequences).toBe(1);
  });
});

describe("createMovementPolicyBackend registry", () => {
  it("resolves the default n-gram backend", () => {
    expect(createMovementPolicyBackend().id).toBe("ngram-backoff");
  });

  it("throws on an unknown backend id", () => {
    expect(() => createMovementPolicyBackend("nope")).toThrow(/Unknown movement-policy backend/);
  });

  it("supports registering a pluggable backend", () => {
    const fake: MovementPolicyBackend = new NgramMovementPolicyBackend();
    registerMovementPolicyBackend("test-backend", () => fake);
    expect(createMovementPolicyBackend("test-backend")).toBe(fake);
  });
});

describe("evaluateMovementGeneralization", () => {
  it("scores held-out predictions and counts backoff-driven generalizations", () => {
    const swipe = { tool: "device", gesture: "swipe", target: "inbox", direction: "down" };
    const archive = { tool: "device", gesture: "tap", target: "archive" };
    const opener = { tool: "device", gesture: "tap", target: "mail-icon" };
    const dataset: MovementDataset = { sequences: [seq("s1", [opener, swipe, archive])] };
    const model = new NgramMovementPolicyBackend().train(dataset, { order: 2 });

    const report = evaluateMovementGeneralization(model, [
      // Seen exact bigram -> memorized (no backoff).
      { context: [opener, swipe], expected: archive },
      // Unseen prefix, same swipe suffix -> generalized via backoff.
      { context: [{ tool: "device", gesture: "tap", target: "widget" }, swipe], expected: archive },
    ]);

    expect(report.total).toBe(2);
    expect(report.matched).toBe(2);
    expect(report.backoffMatched).toBe(1);
    expect(report.accuracy).toBe(1);
  });
});

describe("movementTokenKey", () => {
  it("is stable and distinguishes tokens by every field", () => {
    const base = { tool: "device", gesture: "swipe", target: "x", direction: "up" };
    expect(movementTokenKey(base)).toBe(movementTokenKey({ ...base }));
    expect(movementTokenKey(base)).not.toBe(movementTokenKey({ ...base, direction: "down" }));
  });
});
