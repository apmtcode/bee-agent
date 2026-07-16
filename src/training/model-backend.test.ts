import { describe, expect, it } from "vitest";
import { buildReplayManifest } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  NgramMovementBackend,
  buildMovementDataset,
  buildMovementDatasetFromReplays,
  createMovementModelBackend,
  listMovementModelBackends,
  movementTokenKey,
  registerMovementModelBackend,
  type MovementDataset,
  type MovementToken,
} from "./model-backend.js";

function span(id: string, tokens: Array<{ tool: string; summary: string; gesture?: string; target?: string }>): TrajectorySpan {
  return {
    id,
    sessionId: `session-${id}`,
    createdAt: "2026-07-16T00:00:00.000Z",
    captureTier: "full",
    observations: [],
    actions: tokens.map((token, index) => ({
      kind: "action" as const,
      tool: token.tool,
      summary: token.summary,
      ts: index + 1,
      metadata: {
        ...(token.gesture ? { gesture: token.gesture } : {}),
        ...(token.target ? { target: token.target } : {}),
      },
    })),
  };
}

const FIXED_CLOCK = () => new Date("2026-07-16T12:00:00.000Z");

function backend(order = 2): NgramMovementBackend {
  return new NgramMovementBackend(order, FIXED_CLOCK);
}

describe("movement tokenization + dataset building", () => {
  it("derives structured tokens from action metadata and orders by timestamp", () => {
    const dataset = buildMovementDataset([
      span("t1", [
        { tool: "device", summary: "tapped Login", gesture: "tap", target: "Login" },
        { tool: "device", summary: "typed into Email", gesture: "type", target: "Email" },
      ]),
    ]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]!.tokens[0]).toMatchObject({ tool: "device", gesture: "tap", target: "Login" });
  });

  it("drops trajectories with no actions", () => {
    const dataset = buildMovementDataset([span("empty", [])]);
    expect(dataset.sequences).toHaveLength(0);
  });

  it("builds a dataset from replay manifests", () => {
    const manifest = buildReplayManifest({
      sessionId: "s1",
      transcript: [],
      trajectories: [span("t1", [{ tool: "device", summary: "tapped Login", gesture: "tap", target: "Login" }])],
    });
    const dataset = buildMovementDatasetFromReplays([manifest]);
    expect(dataset.sequences[0]!.tokens).toEqual([{ tool: "device", summary: "tapped Login" }]);
  });

  it("collapses structurally identical movements and separates free-text ones", () => {
    const a: MovementToken = { tool: "device", summary: "tapped A", gesture: "tap", target: "Login" };
    const b: MovementToken = { tool: "device", summary: "different text", gesture: "tap", target: "Login" };
    expect(movementTokenKey(a)).toBe(movementTokenKey(b));

    const c: MovementToken = { tool: "cli", summary: "ran build" };
    const d: MovementToken = { tool: "cli", summary: "ran tests" };
    expect(movementTokenKey(c)).not.toBe(movementTokenKey(d));
  });
});

describe("NgramMovementBackend training + inference", () => {
  const dataset: MovementDataset = {
    version: 1,
    sequences: [
      { id: "s1", tokens: seq("open", "search", "select", "confirm") },
      { id: "s2", tokens: seq("open", "search", "select", "confirm") },
      { id: "s3", tokens: seq("open", "search", "cancel") },
    ],
  };

  it("learns vocabulary and sequence counts", async () => {
    const model = await backend().train(dataset);
    expect(model.backend).toBe("ngram-markov");
    expect(model.sequenceCount).toBe(3);
    expect(model.vocabularySize).toBe(5); // open, search, select, confirm, cancel
    expect(model.trainedAt).toBe("2026-07-16T12:00:00.000Z");
  });

  it("predicts the most frequent continuation deterministically", async () => {
    const model = await backend().train(dataset);
    // After open>search, "select" (2) beats "cancel" (1).
    const prediction = model.predictNext(seq("open", "search"));
    expect(prediction?.token.summary).toBe("select");
    expect(prediction?.contextUsed).toBe(2);
    expect(prediction?.confidence).toBeCloseTo(2 / 3);
  });

  it("is deterministic across repeated calls", async () => {
    const model = await backend().train(dataset);
    const first = model.predictNext(seq("open", "search"));
    const second = model.predictNext(seq("open", "search"));
    expect(first).toEqual(second);
  });

  it("generalizes to an unseen context by backing off to a shorter suffix", async () => {
    const model = await backend().train(dataset);
    // "warmup" was never seen; the pair (warmup, search) is unknown, but the
    // model backs off to context "search" and still predicts a plausible move.
    const prediction = model.predictNext([token("warmup"), token("search")]);
    expect(prediction).toBeDefined();
    expect(prediction!.contextUsed).toBeLessThan(2);
    expect(["select", "cancel"]).toContain(prediction!.token.summary);
  });

  it("falls back to global frequency for a fully unseen context", async () => {
    const model = await backend().train(dataset);
    const prediction = model.predictNext([token("totally-unknown")]);
    expect(prediction).toBeDefined();
    expect(prediction!.contextUsed).toBe(0);
  });

  it("generates a multi-step movement sequence from a seed", async () => {
    const model = await backend().train(dataset);
    const generated = model.generate(seq("open"), 3);
    expect(generated.map((token) => token.summary)).toEqual(["search", "select", "confirm"]);
  });

  it("returns no prediction for an empty vocabulary", async () => {
    const model = await backend().train({ version: 1, sequences: [] });
    expect(model.predictNext(seq("open"))).toBeUndefined();
    expect(model.generate(seq("open"), 5)).toEqual([]);
  });

  it("rejects a non-positive order", () => {
    expect(() => new NgramMovementBackend(0)).toThrow(/positive integer/);
  });
});

describe("model serialization", () => {
  it("round-trips a trained model with identical predictions", async () => {
    const trainer = backend();
    const dataset: MovementDataset = {
      version: 1,
      sequences: [
        { id: "s1", tokens: seq("a", "b", "c") },
        { id: "s2", tokens: seq("a", "b", "d") },
        { id: "s3", tokens: seq("a", "b", "c") },
      ],
    };
    const model = await trainer.train(dataset);
    const serialized = model.serialize();

    // Serialized form must be JSON-safe and stable.
    const roundTripped = JSON.parse(JSON.stringify(serialized));
    const reloaded = trainer.load(roundTripped);

    expect(reloaded.vocabularySize).toBe(model.vocabularySize);
    expect(reloaded.predictNext(seq("a", "b"))).toEqual(model.predictNext(seq("a", "b")));
    expect(reloaded.generate(seq("a"), 2)).toEqual(model.generate(seq("a"), 2));
  });
});

describe("backend registry", () => {
  it("lists and resolves the default n-gram backend", () => {
    expect(listMovementModelBackends()).toContain("ngram-markov");
    expect(createMovementModelBackend().name).toBe("ngram-markov");
  });

  it("throws for an unknown backend name", () => {
    expect(() => createMovementModelBackend("nope")).toThrow(/unknown movement-model backend/);
  });

  it("supports registering a custom backend behind the seam", async () => {
    registerMovementModelBackend("test-const", () => ({
      name: "test-const",
      async train() {
        return {
          backend: "test-const",
          order: 1,
          trainedAt: "2026-07-16T12:00:00.000Z",
          sequenceCount: 0,
          vocabularySize: 0,
          predictNext: () => undefined,
          generate: () => [],
          serialize: () => ({
            backend: "test-const",
            version: 1 as const,
            order: 1,
            trainedAt: "2026-07-16T12:00:00.000Z",
            sequenceCount: 0,
            vocabulary: [],
            transitions: [],
          }),
        };
      },
    }));
    expect(listMovementModelBackends()).toContain("test-const");
    const resolved = createMovementModelBackend("test-const");
    const model = await resolved.train({ version: 1, sequences: [] });
    expect(model.backend).toBe("test-const");
  });
});

function token(summary: string): MovementToken {
  return { tool: "cli", summary };
}

function seq(...summaries: string[]): MovementToken[] {
  return summaries.map((summary) => token(summary));
}
