import { describe, expect, it } from "vitest";
import type { ReplayTimelineEvent } from "../capture/replay.js";
import {
  MarkovMovementBackend,
  buildMovementDataset,
  extractMovementSequence,
  loadMovementModel,
  scoreMovementReplay,
  synthesizeMovementSequences,
  tokenizeMovement,
  type MovementDataset,
} from "./movement-model.js";

function actionEvent(tool: string, summary: string, ts: number): ReplayTimelineEvent {
  return { kind: "action", ts, trajectoryId: "t1", tool, summary };
}

describe("tokenizeMovement / extractMovementSequence", () => {
  it("normalizes tool + summary into a stable token", () => {
    expect(tokenizeMovement({ tool: "  Device ", summary: "Tapped   Submit " })).toBe("device::tapped submit");
  });

  it("keeps only action events, ordered by timestamp", () => {
    const events: ReplayTimelineEvent[] = [
      actionEvent("device", "second", 20),
      { kind: "observation", ts: 5, trajectoryId: "t1", source: "device", summary: "ignored" },
      actionEvent("device", "first", 10),
      { kind: "transcript", ts: 1, messageId: "m1", role: "user", content: "ignored" },
    ];
    expect(extractMovementSequence(events)).toEqual(["device::second", "device::first"].sort());
    // Order is by ts, so first (10) precedes second (20):
    expect(extractMovementSequence(events)).toEqual(["device::first", "device::second"]);
  });
});

describe("buildMovementDataset", () => {
  it("builds one sequence per non-empty replay and drops empty ones", () => {
    const dataset = buildMovementDataset([
      { sessionId: "s1", events: [actionEvent("device", "a", 1), actionEvent("device", "b", 2)] },
      { sessionId: "s2", events: [{ kind: "observation", ts: 1, trajectoryId: "t", source: "d", summary: "x" }] },
    ]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]).toEqual({ id: "s1", tokens: ["device::a", "device::b"] });
  });
});

describe("MarkovMovementBackend — repeat recorded movements (objective 2c)", () => {
  const dataset: MovementDataset = {
    sequences: [{ id: "s1", tokens: ["open", "type", "save", "close"] }],
  };

  it("reproduces the recorded continuation from a seed", () => {
    const model = new MarkovMovementBackend().train(dataset, { order: 2 });
    expect(model.generate(["open"], { maxLength: 10 })).toEqual(["type", "save", "close"]);
  });

  it("predicts the exact next recorded movement with probability 1", () => {
    const model = new MarkovMovementBackend().train(dataset, { order: 2 });
    const prediction = model.predictNext(["open", "type"]);
    expect(prediction?.token).toBe("save");
    expect(prediction?.probability).toBe(1);
    expect(prediction?.backedOff).toBe(false);
    expect(prediction?.contextOrder).toBe(2);
  });
});

describe("MarkovMovementBackend — generalize to new-but-related movements (objective 2d)", () => {
  it("backs off to a shorter learned context for a novel prefix", () => {
    // Both sequences teach "after B comes C". A novel prefix (Q, B) has never
    // been seen at order 2, so the model must back off to the order-1 context.
    const dataset: MovementDataset = {
      sequences: [
        { id: "s1", tokens: ["A", "B", "C"] },
        { id: "s2", tokens: ["X", "B", "C"] },
      ],
    };
    const model = new MarkovMovementBackend().train(dataset, { order: 2 });

    const prediction = model.predictNext(["Q", "B"]);
    expect(prediction?.token).toBe("C");
    expect(prediction?.backedOff).toBe(true);
    expect(prediction?.contextOrder).toBe(1);
  });

  it("ranks candidates by learned frequency and breaks ties deterministically", () => {
    const dataset: MovementDataset = {
      sequences: [
        { id: "s1", tokens: ["m", "a"] },
        { id: "s2", tokens: ["m", "a"] },
        { id: "s3", tokens: ["m", "b"] },
      ],
    };
    const model = new MarkovMovementBackend().train(dataset, { order: 1 });
    const ranked = model.rankNext(["m"]);
    expect(ranked.map((prediction) => prediction.token)).toEqual(["a", "b"]);
    expect(ranked[0]?.probability).toBeCloseTo(2 / 3);
  });
});

describe("serialization round-trip", () => {
  it("reloads to an identical-behaving model", () => {
    const dataset: MovementDataset = {
      sequences: [{ id: "s1", tokens: ["open", "type", "save"] }],
    };
    const model = new MarkovMovementBackend().train(dataset, { order: 2 });
    const reloaded = loadMovementModel(model.serialize());

    expect(reloaded.order).toBe(model.order);
    expect(reloaded.backendId).toBe(model.backendId);
    expect(reloaded.vocabulary()).toEqual(model.vocabulary());
    expect(reloaded.generate(["open"])).toEqual(model.generate(["open"]));
  });
});

describe("synthesizeMovementSequences", () => {
  it("is deterministic for a fixed seed", () => {
    const a = synthesizeMovementSequences({ count: 5, seed: 42, noise: 0.3 });
    const b = synthesizeMovementSequences({ count: 5, seed: 42, noise: 0.3 });
    expect(a).toEqual(b);
    expect(a.length).toBeGreaterThan(0);
  });

  it("produces clean templates when noise is zero", () => {
    const templates = [["a", "b", "c"]];
    const sequences = synthesizeMovementSequences({ count: 3, templates, noise: 0 });
    expect(sequences).toHaveLength(3);
    for (const sequence of sequences) {
      expect(sequence.tokens).toEqual(["a", "b", "c"]);
    }
  });
});

describe("scoreMovementReplay — generalization eval harness", () => {
  it("scores perfect fidelity on clean training data with distinct-prefix templates", () => {
    // Distinct first tokens so a length-1 seed unambiguously selects a template.
    const templates = [
      ["p1", "b", "c"],
      ["q1", "d", "e"],
      ["r1", "f", "g"],
    ];
    const sequences = synthesizeMovementSequences({ count: 6, templates, noise: 0 });
    const model = new MarkovMovementBackend().train({ sequences }, { order: 2 });
    const result = scoreMovementReplay(model, sequences, { seedLength: 1 });
    expect(result.meanFidelity).toBe(1);
  });

  it("still predicts something for held-out related sequences via backoff", () => {
    const train = synthesizeMovementSequences({ count: 8, seed: 1, noise: 0 });
    const model = new MarkovMovementBackend().train({ sequences: train }, { order: 2 });
    // Held-out noisy variants of the same templates — related but unseen.
    const heldOut = synthesizeMovementSequences({ count: 8, seed: 999, noise: 0.4 });
    const result = scoreMovementReplay(model, heldOut, { seedLength: 1 });
    expect(result.meanFidelity).toBeGreaterThan(0);
    expect(result.sequences.length).toBeGreaterThan(0);
  });
});
