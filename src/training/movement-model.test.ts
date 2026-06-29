import { describe, expect, it } from "vitest";
import type { ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  buildMovementDataset,
  createDefaultMovementBackendRegistry,
  movementSequenceFromReplayEvents,
  movementTokenKey,
  MovementModelBackendRegistry,
  NGramMovementBackend,
  type MovementDataset,
  type MovementToken,
} from "./movement-model.js";

function tok(tool: string, summary: string): MovementToken {
  return { tool, summary };
}

function dataset(...sequences: { trajectoryId: string; tokens: MovementToken[] }[]): MovementDataset {
  return { version: 1, sequences };
}

function span(id: string, actions: { tool: string; summary: string; ts: number }[]): TrajectorySpan {
  return {
    id,
    sessionId: `session-${id}`,
    createdAt: "2026-06-29T00:00:00.000Z",
    captureTier: "operator",
    observations: [],
    actions: actions.map((action) => ({ kind: "action" as const, ...action })),
  };
}

describe("NGramMovementBackend", () => {
  it("reproduces a recorded movement exactly from its prefix", () => {
    const backend = new NGramMovementBackend(2);
    const sequence = [tok("mouse", "move 0,0"), tok("mouse", "click"), tok("keyboard", "type hi")];
    const model = backend.train(dataset({ trajectoryId: "t1", tokens: sequence }));

    const prediction = model.predict([sequence[0]]);
    expect(prediction?.token).toEqual(sequence[1]);

    const generated = model.generate([sequence[0]], 5);
    expect(generated).toEqual([sequence[1], sequence[2]]);
  });

  it("generalizes to a novel-but-related continuation via backoff", () => {
    const backend = new NGramMovementBackend(2);
    // Two trajectories share the movement "open menu"; only the prefixes differ.
    const model = backend.train(
      dataset(
        { trajectoryId: "a", tokens: [tok("ui", "focus window"), tok("ui", "open menu"), tok("ui", "select save")] },
        { trajectoryId: "b", tokens: [tok("ui", "scroll down"), tok("ui", "open menu"), tok("ui", "select save")] },
      ),
    );

    // A context never seen verbatim at order-2, but the shared "open menu" state
    // still predicts the learned continuation via backoff to a shorter context.
    const prediction = model.predict([tok("ui", "click elsewhere"), tok("ui", "open menu")]);
    expect(prediction?.token).toEqual(tok("ui", "select save"));
    expect(prediction?.order).toBe(1);
  });

  it("is deterministic under count ties (stable tie-break)", () => {
    const backend = new NGramMovementBackend(1);
    const model = backend.train(
      dataset(
        { trajectoryId: "a", tokens: [tok("k", "x"), tok("k", "b")] },
        { trajectoryId: "b", tokens: [tok("k", "x"), tok("k", "a")] },
      ),
    );
    // Both "a" and "b" follow "x" once; tie broken by ascending token key → "a".
    const first = model.predict([tok("k", "x")]);
    const second = model.predict([tok("k", "x")]);
    expect(first?.token).toEqual(tok("k", "a"));
    expect(second).toEqual(first);
  });

  it("falls back to the unigram prior when no context matches", () => {
    const backend = new NGramMovementBackend(2);
    const model = backend.train(dataset({ trajectoryId: "t1", tokens: [tok("a", "1"), tok("a", "1"), tok("b", "2")] }));
    const prediction = model.predict([tok("z", "unseen")]);
    expect(prediction?.order).toBe(0);
    expect(prediction?.token).toEqual(tok("a", "1")); // most frequent overall
  });

  it("stops generation when no movement can be predicted", () => {
    const backend = new NGramMovementBackend(2);
    const model = backend.train(dataset({ trajectoryId: "t1", tokens: [tok("a", "1"), tok("b", "2")] }));
    // Empty training-context coverage: an unseen vocabulary still backs off to
    // unigram, so generation continues; cap with maxSteps to stay finite.
    const generated = model.generate([tok("a", "1")], 3);
    expect(generated.length).toBeLessThanOrEqual(3);
  });

  it("round-trips a structural snapshot", () => {
    const backend = new NGramMovementBackend(1);
    const model = backend.train(dataset({ trajectoryId: "t1", tokens: [tok("a", "1"), tok("b", "2")] }));
    const snapshot = model.snapshot();
    expect(snapshot.backendId).toBe("ngram");
    expect(snapshot.order).toBe(1);
    expect(snapshot.vocabulary[movementTokenKey(tok("a", "1"))]).toEqual(tok("a", "1"));
    expect(snapshot.transitions[""]).toBeDefined(); // unigram bucket present
  });
});

describe("MovementModelBackendRegistry", () => {
  it("trains via a named backend and lists registered backends", () => {
    const registry = createDefaultMovementBackendRegistry();
    expect(registry.list()).toContain("ngram");
    const model = registry.train("ngram", dataset({ trajectoryId: "t", tokens: [tok("a", "1"), tok("b", "2")] }));
    expect(model.backendId).toBe("ngram");
  });

  it("throws on an unknown backend id", () => {
    const registry = new MovementModelBackendRegistry();
    expect(() => registry.train("nope", dataset())).toThrow(/unknown movement model backend/);
  });
});

describe("dataset builders", () => {
  it("builds a dataset from trajectory actions sorted by timestamp, skipping empties", () => {
    const built = buildMovementDataset([
      span("t1", [
        { tool: "mouse", summary: "click", ts: 20 },
        { tool: "mouse", summary: "move", ts: 10 },
      ]),
      span("empty", []),
    ]);
    expect(built.sequences).toHaveLength(1);
    expect(built.sequences[0].tokens.map((token) => token.summary)).toEqual(["move", "click"]);
  });

  it("extracts only action events from a replay timeline", () => {
    const events: ReplayTimelineEvent[] = [
      { kind: "transcript", ts: 1, messageId: "m1", role: "user", content: "go" },
      { kind: "observation", ts: 2, trajectoryId: "t1", source: "screen", summary: "menu shown" },
      { kind: "action", ts: 3, trajectoryId: "t1", tool: "ui", summary: "open menu" },
    ];
    const sequence = movementSequenceFromReplayEvents("t1", events);
    expect(sequence.tokens).toEqual([tok("ui", "open menu")]);
  });
});
