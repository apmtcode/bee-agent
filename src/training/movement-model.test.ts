import { describe, expect, it } from "vitest";
import {
  MarkovMovementBackend,
  MOVEMENT_END,
  buildMovementDataset,
  buildMovementDatasetFromTrajectories,
  createDefaultMovementBackendRegistry,
  evaluateMovementModel,
  tokenizeReplayEvent,
  type MovementDataset,
} from "./movement-model.js";
import {
  generateSyntheticMovementDataset,
  generateSyntheticMovementSequences,
  syntheticDatasetViaReplay,
  synthesizeReplayManifest,
} from "./synthetic-movement.js";
import { buildReplayManifest } from "../capture/replay.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";

function dataset(sequences: string[][]): MovementDataset {
  const vocabulary = new Set<string>();
  for (const sequence of sequences) {
    for (const token of sequence) {
      vocabulary.add(token);
    }
  }
  return { version: 1, sequences, vocabulary: [...vocabulary].sort() };
}

describe("tokenizeReplayEvent", () => {
  it("maps actions and observations to canonical tokens and skips transcript", () => {
    expect(tokenizeReplayEvent({ kind: "action", ts: 1, trajectoryId: "t", tool: "mouse.move", summary: "" })).toBe(
      "act:mouse.move",
    );
    expect(tokenizeReplayEvent({ kind: "observation", ts: 1, trajectoryId: "t", source: "window", summary: "" })).toBe(
      "obs:window",
    );
    expect(
      tokenizeReplayEvent({ kind: "transcript", ts: 1, messageId: "m", role: "user", content: "hi" }),
    ).toBeUndefined();
  });
});

describe("buildMovementDataset", () => {
  it("groups movement events per trajectory in timeline order", () => {
    const trajectory = buildTrajectorySpan({
      id: "traj-1",
      sessionId: "sess-1",
      observations: [{ kind: "observation", source: "window", summary: "focus", ts: 5 }],
      actions: [
        { kind: "action", tool: "mouse.move", summary: "", ts: 10 },
        { kind: "action", tool: "mouse.click", summary: "", ts: 20 },
      ],
    });
    const replay = buildReplayManifest({ sessionId: "sess-1", transcript: [], trajectories: [trajectory] });
    const built = buildMovementDataset([replay]);
    expect(built.sequences).toEqual([["obs:window", "act:mouse.move", "act:mouse.click"]]);
    expect(built.vocabulary).toEqual(["act:mouse.click", "act:mouse.move", "obs:window"]);
  });

  it("builds directly from trajectory spans", () => {
    const built = buildMovementDatasetFromTrajectories([
      buildTrajectorySpan({
        id: "traj-1",
        sessionId: "sess-1",
        actions: [
          { kind: "action", tool: "key.press", summary: "", ts: 2 },
          { kind: "action", tool: "key.press", summary: "", ts: 1 },
        ],
      }),
    ]);
    expect(built.sequences).toEqual([["act:key.press", "act:key.press"]]);
  });
});

describe("MarkovMovementBackend", () => {
  it("learns transitions and predicts the next movement deterministically", () => {
    const backend = new MarkovMovementBackend();
    const model = backend.train(dataset([["a", "b", "c"], ["a", "b", "c"], ["a", "b", "d"]]), { order: 2 });
    const prediction = model.predictNext(["a", "b"]);
    expect(prediction?.token).toBe("c");
    expect(prediction?.backoffOrder).toBe(2);
    // c appears twice, d once after "a b".
    expect(prediction?.distribution).toEqual([
      { token: "c", probability: 2 / 3 },
      { token: "d", probability: 1 / 3 },
    ]);
  });

  it("backs off to shorter contexts for unseen prefixes (generalization)", () => {
    const backend = new MarkovMovementBackend();
    const model = backend.train(dataset([["x", "a", "b"], ["y", "a", "b"]]), { order: 2 });
    // The bigram context ["z","a"] was never seen; must back off to unigram ["a"].
    const prediction = model.predictNext(["z", "a"]);
    expect(prediction?.token).toBe("b");
    expect(prediction?.backoffOrder).toBe(1);
  });

  it("generates a continuation that terminates at END", () => {
    const backend = new MarkovMovementBackend();
    const model = backend.train(dataset([["a", "b", "c"], ["a", "b", "c"]]), { order: 2 });
    expect(model.generate([], 10)).toEqual(["a", "b", "c"]);
  });

  it("round-trips through serialization", () => {
    const backend = new MarkovMovementBackend();
    const model = backend.train(dataset([["a", "b", "c"]]), { order: 2 });
    const restored = backend.load(model.toJSON());
    expect(restored.predictNext(["a", "b"])?.token).toBe(model.predictNext(["a", "b"])?.token);
    expect(restored.generate([], 5)).toEqual(model.generate([], 5));
  });

  it("returns undefined from an untrained model", () => {
    const model = new MarkovMovementBackend().train(dataset([]), { order: 2 });
    expect(model.predictNext(["a"])).toBeUndefined();
  });

  it("can predict the END sentinel after a known terminal token", () => {
    const model = new MarkovMovementBackend().train(dataset([["a"], ["a"]]), { order: 1 });
    expect(model.predictNext(["a"])?.token).toBe(MOVEMENT_END);
  });
});

describe("createDefaultMovementBackendRegistry", () => {
  it("preloads the markov backend and rejects unknown ids", () => {
    const registry = createDefaultMovementBackendRegistry();
    expect(registry.list()).toEqual(["markov"]);
    expect(registry.has("markov")).toBe(true);
    expect(registry.get("markov").id).toBe("markov");
    expect(() => registry.get("nope")).toThrow(/unknown movement backend/);
  });
});

describe("synthetic movement generator", () => {
  it("is deterministic for a fixed seed", () => {
    const options = { motif: ["a", "b", "c", "d"], episodes: 20, seed: 7 };
    expect(generateSyntheticMovementSequences(options)).toEqual(generateSyntheticMovementSequences(options));
  });

  it("produces related-but-varied episodes (not all identical)", () => {
    const sequences = generateSyntheticMovementSequences({
      motif: ["a", "b", "c", "d"],
      episodes: 40,
      seed: 3,
      mutationRate: 0.4,
    });
    const distinct = new Set(sequences.map((sequence) => sequence.join(",")));
    expect(distinct.size).toBeGreaterThan(1);
    expect(sequences.every((sequence) => sequence.length > 0)).toBe(true);
  });

  it("round-trips synthetic sequences through a replay manifest into a dataset", () => {
    const sequences = generateSyntheticMovementSequences({
      motif: ["act:mouse.move", "obs:window"],
      episodes: 3,
      seed: 1,
      mutationRate: 0,
    });
    const manifest = synthesizeReplayManifest(sequences);
    expect(manifest.eventCount).toBe(6);
    const viaReplay = buildMovementDataset([manifest]);
    expect(viaReplay.sequences).toEqual(sequences);
    expect(syntheticDatasetViaReplay({ motif: ["act:mouse.move", "obs:window"], episodes: 3, seed: 1, mutationRate: 0 }).sequences).toEqual(
      sequences,
    );
  });
});

describe("evaluateMovementModel — generalization harness", () => {
  it("scores next-token accuracy and perplexity on held-out sequences", () => {
    const backend = new MarkovMovementBackend();
    const train = generateSyntheticMovementDataset({ motif: ["a", "b", "c", "d", "e"], episodes: 200, seed: 11, mutationRate: 0.1 });
    const heldOut = generateSyntheticMovementDataset({ motif: ["a", "b", "c", "d", "e"], episodes: 50, seed: 99, mutationRate: 0.1 });
    const model = backend.train(train, { order: 3 });
    const result = evaluateMovementModel(model, heldOut);
    // The model has never seen the held-out episodes verbatim, but the motif is
    // shared — backoff should recover most next movements.
    expect(result.sequences).toBe(50);
    expect(result.nextTokenAccuracy).toBeGreaterThan(0.7);
    expect(result.perplexity).toBeGreaterThan(0);
    expect(Number.isFinite(result.perplexity)).toBe(true);
  });

  it("reports zeroed metrics for an empty held-out set", () => {
    const model = new MarkovMovementBackend().train(dataset([["a", "b"]]), { order: 2 });
    const result = evaluateMovementModel(model, dataset([]));
    expect(result).toEqual({ sequences: 0, predictions: 0, correct: 0, nextTokenAccuracy: 0, perplexity: 0 });
  });
});
