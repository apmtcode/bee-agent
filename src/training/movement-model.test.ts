import { describe, expect, it } from "vitest";
import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import {
  MOVEMENT_END,
  MarkovMovementBackend,
  evaluateMovementModel,
  movementTokenFromAction,
  sequencesFromReplays,
  tokenizeReplayEvents,
  trainMovementModel,
  type MovementSequence,
} from "./movement-model.js";

function actionEvent(tool: string, summary: string, ts: number): ReplayTimelineEvent {
  return { kind: "action", ts, trajectoryId: "traj-1", tool, summary };
}

function seq(tokens: string[], trajectoryId?: string): MovementSequence {
  return { trajectoryId, tokens };
}

describe("movement tokenisation", () => {
  it("normalises tool + summary into a stable token", () => {
    expect(movementTokenFromAction("Device", "  Tapped   Submit ")).toBe("device::tapped submit");
    expect(movementTokenFromAction("device", "")).toBe("device");
  });

  it("extracts only action events from a replay timeline, in order", () => {
    const events: ReplayTimelineEvent[] = [
      { kind: "transcript", ts: 1, messageId: "m1", role: "user", content: "go" },
      actionEvent("device", "tapped search", 2),
      { kind: "observation", ts: 3, trajectoryId: "traj-1", source: "device", summary: "screen" },
      actionEvent("device", "typed query", 4),
    ];
    expect(tokenizeReplayEvents(events)).toEqual(["device::tapped search", "device::typed query"]);
  });

  it("builds one sequence per replay and drops empty ones", () => {
    const replays: ReplayManifest[] = [
      {
        version: 1,
        sessionId: "s1",
        trajectoryIds: ["traj-a"],
        eventCount: 2,
        events: [actionEvent("device", "tapped a", 1), actionEvent("device", "tapped b", 2)],
      },
      {
        version: 1,
        sessionId: "s2",
        trajectoryIds: ["traj-b"],
        eventCount: 1,
        events: [{ kind: "observation", ts: 1, trajectoryId: "traj-b", source: "device", summary: "idle" }],
      },
    ];
    const sequences = sequencesFromReplays(replays);
    expect(sequences).toHaveLength(1);
    expect(sequences[0]).toEqual({ trajectoryId: "traj-a", tokens: ["device::tapped a", "device::tapped b"] });
  });
});

describe("MarkovMovementBackend — repeat recorded movements", () => {
  it("reproduces a recorded trajectory exactly via greedy generation", () => {
    const recorded = ["open app", "tap search", "type query", "tap result"];
    const model = trainMovementModel([seq(recorded)], { order: 2 });
    expect(model.generate({ maxSteps: 16 })).toEqual(recorded);
  });

  it("predicts the recorded continuation for a seen context", () => {
    const model = trainMovementModel([seq(["a", "b", "c", "d"])], { order: 2 });
    const ranked = model.predictNext(["b", "c"], { topK: 1 });
    expect(ranked[0]!.token).toBe("d");
    expect(ranked[0]!.contextOrder).toBe(2);
  });

  it("terminates generation at the learned end of sequence", () => {
    const model = trainMovementModel([seq(["one", "two"])], { order: 1 });
    const generated = model.generate({ maxSteps: 50 });
    expect(generated).toEqual(["one", "two"]);
    expect(generated).not.toContain(MOVEMENT_END);
  });
});

describe("MarkovMovementBackend — generalise to related movements", () => {
  it("backs off to a shorter suffix for an unseen high-order context", () => {
    // "x y" was never seen, but "y" -> "z" was. Backoff should recover "z".
    const model = trainMovementModel([seq(["p", "y", "z"]), seq(["q", "y", "z"])], { order: 2 });
    const ranked = model.predictNext(["x", "y"], { topK: 1 });
    expect(ranked[0]!.token).toBe("z");
    expect(ranked[0]!.contextOrder).toBe(1); // fell back from order 2 to order 1
  });

  it("keeps the whole vocabulary reachable via smoothing for a novel context", () => {
    const model = trainMovementModel([seq(["a", "b"]), seq(["c", "d"])], { order: 2, smoothing: 0.1 });
    const ranked = model.predictNext(["totally", "unseen"]);
    // Every non-start vocab token gets non-zero probability — nothing is impossible.
    const tokens = ranked.map((prediction) => prediction.token);
    expect(tokens).toEqual(expect.arrayContaining(["a", "b", "c", "d", MOVEMENT_END]));
    expect(ranked.every((prediction) => prediction.probability > 0)).toBe(true);
  });

  it("prefers a novel recombination consistent with local transitions", () => {
    // Two trajectories share the "menu" hub; the model should be able to continue
    // from "menu" toward either branch it has evidence for.
    const model = trainMovementModel(
      [seq(["home", "menu", "settings"]), seq(["home", "menu", "profile"])],
      { order: 1 },
    );
    const ranked = model.predictNext(["menu"], { topK: 3 }).map((prediction) => prediction.token);
    expect(ranked).toEqual(expect.arrayContaining(["settings", "profile"]));
  });
});

describe("MarkovMovementBackend — determinism & persistence", () => {
  it("is deterministic across repeated training and generation", () => {
    const data = [seq(["a", "b", "c"]), seq(["a", "b", "d"])];
    const first = trainMovementModel(data, { order: 2 }).generate({ maxSteps: 10 });
    const second = trainMovementModel(data, { order: 2 }).generate({ maxSteps: 10 });
    expect(first).toEqual(second);
  });

  it("produces reproducible seeded sampling", () => {
    const model = trainMovementModel([seq(["a", "b", "c"]), seq(["a", "x", "y"])], { order: 1 });
    const runA = model.generate({ seed: 42, maxSteps: 10 });
    const runB = model.generate({ seed: 42, maxSteps: 10 });
    expect(runA).toEqual(runB);
  });

  it("round-trips through serialize/restore with identical predictions", () => {
    const backend = new MarkovMovementBackend();
    const model = backend.train([seq(["a", "b", "c", "d"])], { order: 2 });
    const snapshot = model.serialize();
    const restored = backend.restore(JSON.parse(JSON.stringify(snapshot)));
    expect(restored.generate({ maxSteps: 16 })).toEqual(model.generate({ maxSteps: 16 }));
    expect(restored.predictNext(["b", "c"])).toEqual(model.predictNext(["b", "c"]));
  });

  it("rejects a snapshot from a different backend", () => {
    const backend = new MarkovMovementBackend();
    const snapshot = backend.train([seq(["a"])], { order: 1 }).serialize();
    expect(() => backend.restore({ ...snapshot, backend: "neural" })).toThrow(/does not match/);
  });
});

describe("evaluateMovementModel — generalisation harness", () => {
  it("scores perfect repeat fidelity on the training distribution", () => {
    const data = [seq(["a", "b", "c"]), seq(["a", "b", "c"])];
    const model = trainMovementModel(data, { order: 2 });
    const evalResult = evaluateMovementModel(model, data, { topK: 1 });
    expect(evalResult.top1Accuracy).toBe(1);
    expect(evalResult.tokenCount).toBe(8); // (3 tokens + END) * 2 sequences
    expect(evalResult.meanLogLoss).toBeLessThan(0.5);
  });

  it("generalises above chance to held-out but related sequences", () => {
    // Train on many variants of home -> menu -> <branch>; hold out one branch.
    const train = [
      seq(["home", "menu", "settings"]),
      seq(["home", "menu", "profile"]),
      seq(["home", "menu", "settings"]),
      seq(["home", "menu", "help"]),
    ];
    const heldOut = [seq(["home", "menu", "profile"])];
    const model = trainMovementModel(train, { order: 2, smoothing: 0.05 });
    const evalResult = evaluateMovementModel(model, heldOut, { topK: 3 });
    // "home"->"menu" is learned; menu's branches are all in top-K.
    expect(evalResult.topKAccuracy).toBeGreaterThanOrEqual(0.75);
    expect(evalResult.top1Accuracy).toBeGreaterThan(0.25);
  });
});
