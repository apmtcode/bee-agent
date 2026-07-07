import { describe, expect, it } from "vitest";
import type { ReplayManifest } from "../capture/replay.js";
import {
  MarkovMovementBackend,
  MOVEMENT_END,
  actionToMovementToken,
  buildMovementDataset,
  extractMovementSequences,
  normalizeMovementSummary,
  synthesizeMovementSequences,
  type MovementSequence,
} from "./movement-model.js";

function manifest(events: ReplayManifest["events"]): ReplayManifest {
  return {
    version: 1,
    sessionId: "session-1",
    trajectoryIds: [...new Set(events.flatMap((e) => ("trajectoryId" in e ? [e.trajectoryId] : [])))],
    eventCount: events.length,
    events,
  };
}

describe("movement tokenization", () => {
  it("collapses volatile digits so related movements share a token", () => {
    expect(normalizeMovementSummary("Click button at (120, 340)")).toBe("click button at (#, #)");
    expect(actionToMovementToken("mouse", "click at 17, 8")).toBe(
      actionToMovementToken("mouse", "click at 999, 4242"),
    );
  });

  it("extracts one ordered sequence per trajectory from a replay manifest", () => {
    const sequences = extractMovementSequences(
      manifest([
        { kind: "action", ts: 3, trajectoryId: "t1", tool: "mouse", summary: "click 3" },
        { kind: "action", ts: 1, trajectoryId: "t1", tool: "mouse", summary: "move 1" },
        { kind: "transcript", ts: 2, messageId: "m", role: "user", content: "hi" },
        { kind: "action", ts: 5, trajectoryId: "t2", tool: "key", summary: "type 5" },
      ]),
    );
    const t1 = sequences.find((s) => s.id === "t1");
    expect(t1?.tokens).toEqual(["mouse|move #", "mouse|click #"]);
    expect(sequences.find((s) => s.id === "t2")?.tokens).toEqual(["key|type #"]);
  });
});

describe("MarkovMovementBackend", () => {
  const dataset: MovementSequence[] = [
    { id: "a", tokens: ["focus", "click", "type", "submit"] },
    { id: "b", tokens: ["focus", "click", "type", "submit"] },
  ];

  it("repeats a recorded movement sequence exactly via generate()", () => {
    const model = new MarkovMovementBackend({ order: 2 }).train(dataset);
    expect(model.generate(["focus"], 10)).toEqual(["click", "type", "submit"]);
  });

  it("predicts the next movement deterministically from context", () => {
    const model = new MarkovMovementBackend({ order: 2 }).train(dataset);
    const prediction = model.predictNext(["focus", "click"]);
    expect(prediction.token).toBe("type");
    expect(prediction.order).toBe(2);
    expect(prediction.confidence).toBeCloseTo(1);
  });

  it("predicts <end> as a null token when the episode should stop", () => {
    const model = new MarkovMovementBackend({ order: 2 }).train(dataset);
    expect(model.predictNext(["type", "submit"]).token).toBeNull();
  });

  it("generalizes to unseen contexts by backing off to shorter history", () => {
    const model = new MarkovMovementBackend({ order: 2 }).train(dataset);
    // "submit" never precedes "click" in training, so the bigram context is
    // unseen; the model must back off to the unigram/lower order and still
    // produce the globally-consistent successor of "click".
    const prediction = model.predictNext(["submit", "click"]);
    expect(prediction.token).toBe("type");
    expect(prediction.order).toBeLessThan(2);
  });

  it("is fully deterministic — same dataset yields identical predictions", () => {
    const backend = new MarkovMovementBackend({ order: 2 });
    const a = backend.train(dataset).predictNext(["focus"]);
    const b = backend.train(dataset).predictNext(["focus"]);
    expect(a).toEqual(b);
    expect(a.token).toBe("click");
  });
});

describe("synthesizeMovementSequences", () => {
  it("produces reproducible sequences for a fixed seed", () => {
    const spec = { vocabulary: ["a", "b", "c"], sequences: 4, minLength: 3, maxLength: 6, seed: 42 };
    expect(synthesizeMovementSequences(spec)).toEqual(synthesizeMovementSequences(spec));
  });

  it("respects length bounds and vocabulary", () => {
    const seqs = synthesizeMovementSequences({
      vocabulary: ["x", "y"],
      sequences: 5,
      minLength: 2,
      maxLength: 4,
      seed: 7,
    });
    expect(seqs).toHaveLength(5);
    for (const seq of seqs) {
      expect(seq.tokens.length).toBeGreaterThanOrEqual(2);
      expect(seq.tokens.length).toBeLessThanOrEqual(4);
      expect(seq.tokens.every((t) => t === "x" || t === "y")).toBe(true);
    }
  });

  it("returns nothing for an empty vocabulary", () => {
    expect(
      synthesizeMovementSequences({ vocabulary: [], sequences: 3, minLength: 1, maxLength: 2, seed: 1 }),
    ).toEqual([]);
  });
});

describe("buildMovementDataset", () => {
  it("flattens sequences across manifests", () => {
    const dataset = buildMovementDataset([
      manifest([{ kind: "action", ts: 1, trajectoryId: "t1", tool: "a", summary: "one" }]),
      manifest([{ kind: "action", ts: 1, trajectoryId: "t2", tool: "b", summary: "two" }]),
    ]);
    expect(dataset.map((s) => s.id).sort()).toEqual(["t1", "t2"]);
  });
});

it("re-exports the end sentinel", () => {
  expect(MOVEMENT_END).toBe("<end>");
});
