import { describe, expect, it } from "vitest";
import type { ReplayTimelineEvent } from "../capture/replay.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import {
  DEFAULT_MOVEMENT_BACKEND_ID,
  MarkovMovementBackend,
  createMovementModelBackend,
  evaluateMovementModel,
  listMovementModelBackends,
  movementDatasetFromReplays,
  movementDatasetFromTrajectories,
  registerMovementModelBackend,
  tokenizeReplayEvents,
  type MovementDataset,
} from "./movement-model.js";

function dataset(sequences: Array<{ id: string; tokens: string[] }>): MovementDataset {
  return { version: 1, sequences };
}

describe("MarkovMovementBackend", () => {
  it("reproduces a recorded movement sequence exactly (replay)", () => {
    const backend = new MarkovMovementBackend();
    const artifact = backend.train(
      dataset([{ id: "t1", tokens: ["open", "focus", "click", "type", "submit"] }]),
    );

    expect(backend.generate(artifact)).toEqual(["open", "focus", "click", "type", "submit"]);
  });

  it("generalizes to a novel-but-related prefix via context back-off", () => {
    const backend = new MarkovMovementBackend();
    const artifact = backend.train(
      dataset([
        { id: "a", tokens: ["open", "focus", "click", "type", "submit"] },
        { id: "b", tokens: ["launch", "focus", "click", "type", "save"] },
      ]),
      { maxOrder: 3 },
    );

    // "resume" was never seen; the model backs off to the shared [focus]/[click]
    // context and still produces the learned continuation.
    const continuation = backend.generate(artifact, { seed: ["resume", "focus"] });
    expect(continuation).toEqual(["click", "type", "save"]);
  });

  it("ranks predictions by observed frequency with deterministic tie-break", () => {
    const backend = new MarkovMovementBackend();
    const artifact = backend.train(
      dataset([
        { id: "a", tokens: ["click", "type"] },
        { id: "b", tokens: ["click", "type"] },
        { id: "c", tokens: ["click", "scroll"] },
      ]),
      { maxOrder: 2 },
    );

    const predictions = backend.predict(artifact, ["click"]);
    expect(predictions.map((p) => p.token)).toEqual(["type", "scroll"]);
    expect(predictions[0]?.count).toBe(2);
    expect(predictions[0]?.probability).toBeCloseTo(2 / 3);
    expect(predictions[0]?.order).toBe(1);
  });

  it("falls back to the unigram distribution for fully novel context", () => {
    const backend = new MarkovMovementBackend();
    const artifact = backend.train(dataset([{ id: "a", tokens: ["a", "b", "b"] }]), { maxOrder: 2 });

    const predictions = backend.predict(artifact, ["zzz-unseen"]);
    // unigram: "b" appears twice, "a" once.
    expect(predictions[0]?.token).toBe("b");
    expect(predictions[0]?.order).toBe(0);
  });

  it("excludes the end sentinel from predictions unless requested", () => {
    const backend = new MarkovMovementBackend();
    const artifact = backend.train(dataset([{ id: "a", tokens: ["only"] }]), { maxOrder: 2 });

    // With the sentinel included, the matched order-1 context [only] → <end>.
    expect(backend.predict(artifact, ["only"], { includeEnd: true })[0]?.token).toBe("<end>");
    // Without it, that order yields nothing emittable, so it backs off to the
    // unigram distribution — never surfacing <end> as a movement.
    const predicted = backend.predict(artifact, ["only"]);
    expect(predicted.every((p) => p.token !== "<end>")).toBe(true);
    expect(predicted.map((p) => p.token)).toEqual(["only"]);
  });

  it("produces a JSON-serializable artifact that round-trips", () => {
    const backend = new MarkovMovementBackend();
    const artifact = backend.train(dataset([{ id: "a", tokens: ["x", "y", "z"] }]), {
      trainedAt: "2026-06-28T00:00:00.000Z",
    });

    const roundTripped = JSON.parse(JSON.stringify(artifact));
    expect(backend.generate(roundTripped)).toEqual(["x", "y", "z"]);
    expect(artifact.trainedAt).toBe("2026-06-28T00:00:00.000Z");
    expect(artifact.vocabulary).toEqual(["x", "y", "z"]);
    expect(artifact.sequenceCount).toBe(1);
    expect(artifact.tokenCount).toBe(3);
  });
});

describe("dataset builders", () => {
  it("tokenizes replay events into movement tokens", () => {
    const events: ReplayTimelineEvent[] = [
      { kind: "transcript", ts: 1, messageId: "m1", role: "user", content: "go" },
      { kind: "observation", ts: 2, trajectoryId: "t", source: "screen", summary: "frame" },
      { kind: "action", ts: 3, trajectoryId: "t", tool: "click", summary: "button" },
    ];

    expect(tokenizeReplayEvents(events)).toEqual(["observation:screen", "action:click"]);
    expect(tokenizeReplayEvents(events, { includeObservations: false })).toEqual(["action:click"]);
    expect(tokenizeReplayEvents(events, { includeTranscript: true })).toEqual([
      "msg:user",
      "observation:screen",
      "action:click",
    ]);
  });

  it("builds a dataset from replay manifests", () => {
    const dataset = movementDatasetFromReplays([
      {
        sessionId: "s1",
        events: [
          { kind: "action", ts: 1, trajectoryId: "t", tool: "open", summary: "" },
          { kind: "action", ts: 2, trajectoryId: "t", tool: "close", summary: "" },
        ],
      },
    ]);

    expect(dataset.sequences).toEqual([{ id: "s1", tokens: ["action:open", "action:close"] }]);
  });

  it("builds a dataset from trajectory spans ordered by timestamp", () => {
    const span = buildTrajectorySpan({
      id: "tj1",
      sessionId: "s1",
      observations: [{ kind: "observation", source: "ui", summary: "", ts: 1 }],
      actions: [
        { kind: "action", tool: "type", summary: "", ts: 3 },
        { kind: "action", tool: "click", summary: "", ts: 2 },
      ],
    });

    const dataset = movementDatasetFromTrajectories([span]);
    expect(dataset.sequences[0]?.tokens).toEqual([
      "observation:ui",
      "action:click",
      "action:type",
    ]);
  });
});

describe("backend registry", () => {
  it("creates the default Markov backend and lists registered ids", () => {
    expect(listMovementModelBackends()).toContain(DEFAULT_MOVEMENT_BACKEND_ID);
    expect(createMovementModelBackend().id).toBe(DEFAULT_MOVEMENT_BACKEND_ID);
  });

  it("supports registering a pluggable alternate backend", () => {
    registerMovementModelBackend("noop-test-backend", () => ({
      id: "noop-test-backend",
      train: () => ({
        version: 1,
        backend: "noop-test-backend",
        maxOrder: 0,
        vocabulary: [],
        sequenceCount: 0,
        tokenCount: 0,
        transitions: {},
      }),
      predict: () => [],
      generate: () => [],
    }));

    expect(listMovementModelBackends()).toContain("noop-test-backend");
    expect(createMovementModelBackend("noop-test-backend").generate({
      version: 1,
      backend: "noop-test-backend",
      maxOrder: 0,
      vocabulary: [],
      sequenceCount: 0,
      tokenCount: 0,
      transitions: {},
    })).toEqual([]);
  });

  it("throws on an unknown backend id", () => {
    expect(() => createMovementModelBackend("does-not-exist")).toThrow(/Unknown movement-model backend/);
  });
});

describe("evaluateMovementModel", () => {
  it("scores perfect fidelity on the training sequence", () => {
    const backend = new MarkovMovementBackend();
    const sequence = { id: "a", tokens: ["open", "click", "type", "submit"] };
    const artifact = backend.train(dataset([sequence]));

    const result = evaluateMovementModel(backend, artifact, [sequence]);
    expect(result.accuracy).toBe(1);
    expect(result.exactSequenceMatch).toBe(1);
    expect(result.correct).toBe(result.predictions);
  });

  it("measures partial generalization on a held-out related sequence", () => {
    const backend = new MarkovMovementBackend();
    const artifact = backend.train(
      dataset([
        { id: "a", tokens: ["open", "focus", "click", "type", "submit"] },
        { id: "b", tokens: ["launch", "focus", "click", "type", "save"] },
      ]),
      { maxOrder: 3 },
    );

    const heldOut = [{ id: "c", tokens: ["resume", "focus", "click", "type", "save"] }];
    const result = evaluateMovementModel(backend, artifact, heldOut);
    // The shared [focus → click → type] core is predicted; the novel first
    // token cannot be, so accuracy is high but below 1.
    expect(result.accuracy).toBeGreaterThan(0.5);
    expect(result.accuracy).toBeLessThan(1);
  });
});
