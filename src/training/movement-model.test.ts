import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  MarkovMovementBackend,
  MOVEMENT_END,
  MovementTrainingService,
  evaluateMovementModel,
  getMovementBackend,
  listMovementBackends,
  registerMovementBackend,
  sequenceFromReplay,
  sequencesFromReplays,
  tokenizeReplayEvent,
  type MovementModelArtifact,
  type MovementModelBackend,
  type MovementSequence,
} from "./movement-model.js";
import {
  DEFAULT_TASK_GRAMMAR,
  generateSyntheticReplays,
} from "./movement-synthetic.js";
import type { ReplayManifest } from "../capture/replay.js";

function seq(id: string, tokens: string[]): MovementSequence {
  return { id, tokens };
}

describe("tokenization", () => {
  it("maps actions and observations to movement tokens, skips transcript by default", () => {
    expect(
      tokenizeReplayEvent({ kind: "action", ts: 1, trajectoryId: "t", tool: "mouse-click", summary: "" }),
    ).toBe("action:mouse-click");
    expect(
      tokenizeReplayEvent({ kind: "observation", ts: 1, trajectoryId: "t", source: "window", summary: "" }),
    ).toBe("observation:window");
    expect(
      tokenizeReplayEvent({ kind: "transcript", ts: 1, messageId: "m", role: "user", content: "" }),
    ).toBeUndefined();
  });

  it("builds a timeline-ordered sequence and drops empty replays", () => {
    const replay: ReplayManifest = {
      version: 1,
      sessionId: "s1",
      trajectoryIds: ["s1"],
      eventCount: 2,
      events: [
        { kind: "observation", ts: 1, trajectoryId: "s1", source: "window", summary: "" },
        { kind: "action", ts: 2, trajectoryId: "s1", tool: "mouse-click", summary: "" },
      ],
    };
    expect(sequenceFromReplay(replay).tokens).toEqual(["observation:window", "action:mouse-click"]);

    const empty: ReplayManifest = { version: 1, sessionId: "e", trajectoryIds: [], eventCount: 0, events: [] };
    expect(sequencesFromReplays([replay, empty])).toHaveLength(1);
  });
});

describe("MarkovMovementBackend", () => {
  const backend = new MarkovMovementBackend();

  it("faithfully replays a recorded movement sequence via free roll-out", () => {
    const tokens = ["action:a", "action:b", "action:c", "action:d"];
    const model = backend.train([seq("s", tokens)], { order: 3 });
    expect(backend.generate(model)).toEqual(tokens);
  });

  it("predicts the recorded next movement with full confidence and stops at END", () => {
    const tokens = ["action:a", "action:b", "action:c"];
    const model = backend.train([seq("s", tokens)], { order: 2 });
    const p1 = backend.predict(model, ["action:a"]);
    expect(p1.token).toBe("action:b");
    expect(p1.probability).toBe(1);
    expect(p1.matched).toBe(true);

    const end = backend.predict(model, tokens);
    expect(end.token).toBe(MOVEMENT_END);
  });

  it("is deterministic under ties (argmax broken by token order)", () => {
    // From context [x], both "action:m" and "action:n" follow once each.
    const model = backend.train(
      [seq("s1", ["action:x", "action:n"]), seq("s2", ["action:x", "action:m"])],
      { order: 1 },
    );
    const a = backend.predict(model, ["action:x"]);
    const b = backend.predict(model, ["action:x"]);
    expect(a.token).toBe("action:m"); // sorts before "action:n"
    expect(a.token).toBe(b.token);
  });

  it("generalizes to an unseen prefix by backing off to a shorter context", () => {
    // Two trajectories share the tail [...-> save]. A novel high-order prefix
    // that was never seen should still predict "save" via back-off.
    const model = backend.train(
      [
        seq("s1", ["action:open", "action:click", "action:type", "action:save"]),
        seq("s2", ["action:launch", "action:focus", "action:type", "action:save"]),
      ],
      { order: 3 },
    );
    // Prefix ending in "action:type" was seen at order-1 → predicts save.
    const prediction = backend.predict(model, ["action:brand-new", "action:also-new", "action:type"]);
    expect(prediction.token).toBe("action:save");
    expect(prediction.matched).toBe(true);
    expect(prediction.backoffOrder).toBeLessThan(model.order);
  });

  it("reports no match on an empty model", () => {
    const model = backend.train([], { order: 2 });
    const prediction = backend.predict(model, ["action:whatever"]);
    expect(prediction.matched).toBe(false);
    expect(model.tokenCount).toBe(0);
    expect(model.sequenceCount).toBe(0);
  });

  it("records trainedAt only when a clock is provided (deterministic by default)", () => {
    const withoutClock = backend.train([seq("s", ["action:a"])]);
    expect(withoutClock.trainedAt).toBeUndefined();
    const withClock = backend.train([seq("s", ["action:a"])], { now: () => "2026-07-10T00:00:00Z" });
    expect(withClock.trainedAt).toBe("2026-07-10T00:00:00Z");
  });
});

describe("evaluateMovementModel", () => {
  const backend = new MarkovMovementBackend();

  it("scores held-out next-token accuracy and exact roll-out", () => {
    const train = generateSyntheticReplays({ grammar: DEFAULT_TASK_GRAMMAR, count: 6, seed: 7 });
    const model = backend.train(sequencesFromReplays(train), { order: 3 });
    const heldOut = sequencesFromReplays(train);

    const evaluation = evaluateMovementModel(backend, model, heldOut);
    expect(evaluation.sequenceCount).toBe(heldOut.length);
    // The canonical (no-variation) task is fully memorized → perfect fidelity.
    expect(evaluation.accuracy).toBe(1);
    expect(evaluation.exactRolloutRate).toBe(1);
  });

  it("generalizes above chance to related-but-unseen variations", () => {
    const train = sequencesFromReplays(
      generateSyntheticReplays({ grammar: DEFAULT_TASK_GRAMMAR, count: 8, seed: 1 }),
    );
    const model = backend.train(train, { order: 3 });
    // Held-out set introduces tool substitutions the model never saw.
    const heldOut = sequencesFromReplays(
      generateSyntheticReplays({ grammar: DEFAULT_TASK_GRAMMAR, count: 8, seed: 99, variationRate: 1 }),
    );
    const evaluation = evaluateMovementModel(backend, model, heldOut);
    // Varied phases can't be predicted, but shared phases still land via back-off.
    expect(evaluation.accuracy).toBeGreaterThan(0.5);
    expect(evaluation.accuracy).toBeLessThan(1);
  });

  it("handles an empty held-out set", () => {
    const model = backend.train([seq("s", ["action:a"])]);
    const evaluation = evaluateMovementModel(backend, model, []);
    expect(evaluation.accuracy).toBe(0);
    expect(evaluation.exactRolloutRate).toBe(0);
  });
});

describe("backend registry", () => {
  it("exposes the default markov backend and allows registering new ones", () => {
    expect(listMovementBackends()).toContain("markov");
    expect(getMovementBackend("markov")).toBeInstanceOf(MarkovMovementBackend);
    expect(() => getMovementBackend("nope")).toThrow(/Unknown movement-model backend/);

    const stub: MovementModelBackend = {
      name: "stub-test-backend",
      train: () => ({
        version: 1,
        backend: "stub-test-backend",
        order: 1,
        vocabulary: [],
        transitions: {},
        sequenceCount: 0,
        tokenCount: 0,
      }),
      predict: () => ({ token: MOVEMENT_END, probability: 0, backoffOrder: 0, matched: false }),
      generate: () => [],
    };
    registerMovementBackend(stub);
    expect(getMovementBackend("stub-test-backend")).toBe(stub);
  });
});

describe("MovementTrainingService", () => {
  let root: string;

  beforeEach(async () => {
    root = await mkdtemp(path.join(tmpdir(), "movement-model-"));
  });

  afterEach(async () => {
    await rm(root, { recursive: true, force: true });
  });

  it("trains from replays, persists, reloads, and infers", async () => {
    const service = new MovementTrainingService(root);
    const replays = generateSyntheticReplays({ grammar: DEFAULT_TASK_GRAMMAR, count: 4, seed: 3 });

    const model = service.trainFromReplays(replays, { order: 3, now: () => "2026-07-10T00:00:00Z" });
    expect(model.trainedAt).toBe("2026-07-10T00:00:00Z");
    expect(model.sequenceCount).toBe(4);

    const savedPath = await service.saveModel("models/movement.json", model);
    expect(savedPath).toContain(root);

    const reloaded = (await service.loadModel("models/movement.json")) as MovementModelArtifact;
    expect(reloaded.transitions).toEqual(model.transitions);

    // Inference from the reloaded artifact reproduces the recorded movement.
    const rollout = service.generate(reloaded);
    const expected = sequencesFromReplays(replays)[0]!.tokens;
    expect(rollout).toEqual(expected);
  });

  it("returns undefined when loading a missing model", async () => {
    const service = new MovementTrainingService(root);
    expect(await service.loadModel("models/absent.json")).toBeUndefined();
  });
});
