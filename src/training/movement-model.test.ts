import { describe, expect, it } from "vitest";
import { buildReplayManifest } from "../capture/replay.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import {
  MarkovMovementBackend,
  MovementBackendRegistry,
  evaluateNextTokenAccuracy,
  movementSequencesFromReplay,
  movementTokenFromAction,
  movementTokensFromActions,
  type MovementSequence,
} from "./movement-model.js";

const backend = new MarkovMovementBackend();

function seq(id: string, ...tokens: string[]): MovementSequence {
  return { id, tokens };
}

describe("MarkovMovementBackend", () => {
  it("replays a recorded movement exactly via greedy rollout", () => {
    const recorded = seq("r1", "open:Finder", "click:Documents", "type:report", "save:file");
    const model = backend.train([recorded], { order: 3 });

    const replayed = model.generate([recorded.tokens[0]!]);
    expect([recorded.tokens[0]!, ...replayed]).toEqual(recorded.tokens);
  });

  it("is deterministic — same data, same predictions and serialization", () => {
    const data = [seq("a", "x", "y", "z"), seq("b", "x", "y", "w")];
    const first = backend.train(data).serialize();
    const second = backend.train(data).serialize();
    expect(first).toEqual(second);
  });

  it("predicts the majority continuation and reports the full distribution", () => {
    // After "x y", "z" was seen twice and "w" once → z is argmax at prob 2/3.
    const model = backend.train([seq("a", "x", "y", "z"), seq("b", "x", "y", "z"), seq("c", "x", "y", "w")]);
    const prediction = model.predictNext(["x", "y"]);
    expect(prediction.token).toBe("z");
    expect(prediction.probability).toBeCloseTo(2 / 3);
    // Full order-3 context (START-padded [<start>, x, y]) was seen in training.
    expect(prediction.order).toBe(3);
    expect(prediction.backedOff).toBe(false);
    expect(prediction.distribution.map((c) => c.token)).toEqual(["z", "w"]);
  });

  it("generalizes to an unseen prefix by backing off to a shorter context", () => {
    // Train so the bigram (b→c) is well attested, but the exact trigram
    // context "q a b" was never recorded. A pure trigram model would have
    // nothing to say; backoff resolves it through the seen bigram.
    const model = backend.train([
      seq("t1", "a", "b", "c"),
      seq("t2", "x", "b", "c"),
      seq("t3", "y", "b", "c"),
    ]);
    const prediction = model.predictNext(["q", "a", "b"]);
    expect(prediction.token).toBe("c");
    expect(prediction.backedOff).toBe(true);
    expect(prediction.order).toBeLessThan(3);
  });

  it("signals end-of-movement separately from real tokens", () => {
    const model = backend.train([seq("s", "a", "b")]);
    const afterB = model.predictNext(["a", "b"]);
    expect(afterB.token).toBeUndefined();
    expect(afterB.endProbability).toBeCloseTo(1);
    expect(afterB.distribution).toHaveLength(0);
  });

  it("round-trips through serialize/load with identical behaviour", () => {
    const data = [seq("a", "p", "q", "r"), seq("b", "p", "q", "s"), seq("c", "p", "t")];
    const original = backend.train(data);
    const restored = backend.load(original.serialize());
    for (const prefix of [[], ["p"], ["p", "q"], ["p", "t"]]) {
      expect(restored.predictNext(prefix)).toEqual(original.predictNext(prefix));
    }
    expect(restored.serialize()).toEqual(original.serialize());
  });

  it("returns an empty prediction for a model with no data", () => {
    const model = backend.train([]);
    const prediction = model.predictNext(["anything"]);
    expect(prediction.token).toBeUndefined();
    expect(prediction.distribution).toHaveLength(0);
    expect(model.vocabularySize).toBe(0);
  });
});

describe("evaluateNextTokenAccuracy", () => {
  it("scores perfect fidelity on the training distribution", () => {
    const data = [seq("a", "l", "o", "g", "i", "n"), seq("b", "l", "o", "g", "i", "n")];
    const model = backend.train(data);
    const report = evaluateNextTokenAccuracy(model, data);
    expect(report.accuracy).toBe(1);
    expect(report.coverage).toBe(1);
    expect(report.predictions).toBeGreaterThan(0);
  });

  it("measures generalization on held-out but related sequences", () => {
    // Held-out prefix shares suffix structure but a novel first token.
    const train = [seq("a", "menu", "file", "save"), seq("b", "menu", "file", "save")];
    const held = [seq("h", "toolbar", "file", "save")];
    const model = backend.train(train);
    const report = evaluateNextTokenAccuracy(model, held);
    // "file"→"save" generalizes via backoff even though "toolbar file" is unseen.
    expect(report.coverage).toBeGreaterThan(0);
    expect(report.backoffRate).toBeGreaterThan(0);
  });
});

describe("MovementBackendRegistry", () => {
  it("resolves the default markov backend and loads its models", () => {
    const registry = new MovementBackendRegistry();
    expect(registry.ids()).toContain("markov");
    const model = registry.resolve("markov").train([seq("a", "one", "two")]);
    const reloaded = registry.load(model.serialize());
    expect(reloaded.backendId).toBe("markov");
    expect(reloaded.predictNext(["one"]).token).toBe("two");
  });

  it("throws for an unknown backend id", () => {
    const registry = new MovementBackendRegistry();
    expect(() => registry.resolve("nope")).toThrow(/unknown movement model backend/);
  });
});

describe("dataset adapters", () => {
  it("tokenizes trajectory actions in timestamp order", () => {
    const tokens = movementTokensFromActions([
      { kind: "action", tool: "device", summary: "tapped Send", ts: 20 },
      { kind: "action", tool: "device", summary: "typed message", ts: 10 },
    ]);
    expect(tokens).toEqual([
      movementTokenFromAction("device", "typed message"),
      movementTokenFromAction("device", "tapped Send"),
    ]);
  });

  it("builds one sequence per trajectory from a replay manifest", () => {
    const trajectory = buildTrajectorySpan({
      id: "traj-1",
      sessionId: "sess-1",
      actions: [
        { kind: "action", tool: "device", summary: "open app", ts: 1 },
        { kind: "action", tool: "device", summary: "tap button", ts: 2 },
      ],
    });
    const manifest = buildReplayManifest({ sessionId: "sess-1", transcript: [], trajectories: [trajectory] });
    const sequences = movementSequencesFromReplay(manifest);
    expect(sequences).toHaveLength(1);
    expect(sequences[0]!.id).toBe("traj-1");
    expect(sequences[0]!.tokens).toEqual([
      movementTokenFromAction("device", "open app"),
      movementTokenFromAction("device", "tap button"),
    ]);

    // Sanity: a model trained on the replay reproduces the movement.
    const model = backend.train(sequences);
    expect([sequences[0]!.tokens[0]!, ...model.generate([sequences[0]!.tokens[0]!])]).toEqual(sequences[0]!.tokens);
  });
});
