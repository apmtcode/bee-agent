import { describe, expect, it } from "vitest";
import {
  DeterministicMarkovBackend,
  buildMovementDatasetFromTrajectories,
  buildMovementSequenceFromReplayEvents,
  evaluateMovementModel,
  tokenizeAction,
  tokenizeObservation,
  type MovementDataset,
  type MovementSequence,
} from "./model-backend.js";

function action(tool: string, summary: string, metadata?: Record<string, unknown>) {
  return { kind: "action" as const, tool, summary, ts: 0, ...(metadata ? { metadata } : {}) };
}

describe("tokenization", () => {
  it("prefers structured gesture + direction metadata over free text", () => {
    expect(tokenizeAction({ tool: "device", summary: "swiped down", metadata: { gesture: "swipe", direction: "down" } })).toBe(
      "device:swipe:down",
    );
    expect(tokenizeAction({ tool: "device", summary: "tapped Submit", metadata: { gesture: "tap" } })).toBe("device:tap");
  });

  it("falls back to the leading verb of the summary", () => {
    expect(tokenizeAction({ tool: "keyboard", summary: "Typed the report title" })).toBe("keyboard:typed");
    expect(tokenizeAction({ tool: "", summary: "" })).toBe("action:event");
  });

  it("tokenizes observations from source + event metadata", () => {
    expect(tokenizeObservation({ source: "os", summary: "focused Editor", metadata: { event: "focus-changed" } })).toBe(
      "os:focus-changed",
    );
    expect(tokenizeObservation({ source: "os", summary: "focused Editor" })).toBe("os:focused");
  });
});

describe("dataset construction", () => {
  it("builds one sequence per trajectory and prefers redacted actions", () => {
    const dataset = buildMovementDatasetFromTrajectories([
      {
        id: "t1",
        sessionId: "s1",
        actions: [action("device", "raw", { gesture: "tap" }), action("keyboard", "typed secret")],
        outcome: { status: "success" },
        review: { redactedActions: [{ tool: "device", summary: "tapped", metadata: { gesture: "tap" } }] },
      },
    ]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]!.tokens).toEqual(["device:tap"]);
    expect(dataset.sequences[0]!.outcome).toBe("success");
    expect(dataset.vocabulary).toEqual(["device:tap"]);
  });

  it("builds a replay sequence from action + observation events, skipping transcript", () => {
    const sequence = buildMovementSequenceFromReplayEvents("s1", [
      { kind: "transcript" },
      { kind: "observation", source: "os", summary: "focused Editor", metadata: { event: "focus-changed" } },
      { kind: "action", tool: "device", summary: "tapped", metadata: { gesture: "tap" } },
      { kind: "transcript" },
    ]);
    expect(sequence.tokens).toEqual(["os:focus-changed", "device:tap"]);
  });
});

const DATASET: MovementDataset = {
  version: 1,
  sequences: [
    { id: "a", tokens: ["open", "select", "type", "save"] },
    { id: "b", tokens: ["open", "select", "type", "save"] },
    { id: "c", tokens: ["open", "select", "delete", "save"] },
  ],
  vocabulary: ["delete", "open", "save", "select", "type"],
};

describe("DeterministicMarkovBackend", () => {
  const backend = new DeterministicMarkovBackend();

  it("repeats a recorded movement at the highest matching order", async () => {
    const model = await backend.train(DATASET, { maxOrder: 3 });
    // "open" -> "select" is deterministic in every sequence.
    const prediction = backend.predictNext(model, ["open"]);
    expect(prediction.token).toBe("select");
    expect(prediction.confidence).toBe(1);
    expect(prediction.order).toBe(1);
  });

  it("resolves ambiguity by frequency with a deterministic tie-break", async () => {
    const model = await backend.train(DATASET, { maxOrder: 3 });
    // After "select" comes "type" (2x) vs "delete" (1x) at order 1.
    const prediction = backend.predictNext(model, ["select"]);
    expect(prediction.token).toBe("type");
    expect(prediction.candidates.map((c) => c.token)).toEqual(["type", "delete"]);
    expect(prediction.candidates[0]!.probability).toBeCloseTo(2 / 3);
  });

  it("uses the longest context available (higher order wins)", async () => {
    const model = await backend.train(DATASET, { maxOrder: 3 });
    // Trigram "open select type" -> "save" is fully determined.
    const prediction = backend.predictNext(model, ["open", "select", "type"]);
    expect(prediction.token).toBe("save");
    expect(prediction.order).toBe(3);
  });

  it("generalizes to a novel context via backoff", async () => {
    const model = await backend.train(DATASET, { maxOrder: 3 });
    // "paste select" was never seen, but "select" -> ... backs off to order 1.
    const prediction = backend.predictNext(model, ["paste", "select"]);
    expect(prediction.token).toBe("type");
    expect(prediction.order).toBe(1);
  });

  it("falls back to the unigram distribution for an entirely novel context", async () => {
    const model = await backend.train(DATASET, { maxOrder: 3 });
    const prediction = backend.predictNext(model, ["totally-unseen-token"]);
    expect(prediction.order).toBe(0);
    // "save" and "open" and "select" each appear 3x; tie-break is lexicographic.
    expect(prediction.token).toBe("open");
  });

  it("returns an empty prediction for an empty model", async () => {
    const model = await backend.train({ version: 1, sequences: [], vocabulary: [] });
    const prediction = backend.predictNext(model, ["anything"]);
    expect(prediction.token).toBeNull();
    expect(prediction.order).toBe(-1);
  });

  it("generates a recorded movement continuation and honors the step cap", async () => {
    const model = await backend.train(DATASET, { maxOrder: 3 });
    const generated = backend.generate(model, ["open"], { maxSteps: 3 });
    expect(generated).toEqual(["select", "type", "save"]);
  });

  it("stops generation at the stop token", async () => {
    const model = await backend.train(DATASET, { maxOrder: 3 });
    const generated = backend.generate(model, ["open"], { maxSteps: 10, stopToken: "type" });
    expect(generated).toEqual(["select", "type"]);
  });

  it("produces byte-identical models across runs (determinism)", async () => {
    const first = await backend.train(DATASET, { maxOrder: 3 });
    const second = await backend.train(DATASET, { maxOrder: 3 });
    expect(JSON.stringify(first)).toBe(JSON.stringify(second));
  });
});

describe("evaluateMovementModel", () => {
  const backend = new DeterministicMarkovBackend();

  it("scores perfect fidelity when repeating a training sequence", async () => {
    const model = await backend.train(DATASET, { maxOrder: 3 });
    const heldOut: MovementSequence[] = [{ id: "eval", tokens: ["open", "select", "type", "save"] }];
    const result = evaluateMovementModel(backend, model, heldOut);
    expect(result.predictions).toBe(3);
    expect(result.correct).toBe(3);
    expect(result.accuracy).toBe(1);
    expect(result.meanOrder).toBeGreaterThan(0);
  });

  it("still scores partial fidelity on a related-but-novel sequence", async () => {
    const model = await backend.train(DATASET, { maxOrder: 3 });
    // Novel ordering built from familiar tokens: generalization, not repetition.
    const heldOut: MovementSequence[] = [{ id: "eval", tokens: ["select", "type", "open"] }];
    const result = evaluateMovementModel(backend, model, heldOut);
    expect(result.predictions).toBe(2);
    expect(result.accuracy).toBeGreaterThan(0);
    expect(result.accuracy).toBeLessThanOrEqual(1);
  });

  it("returns zeroed metrics with no predictions for trivial sequences", async () => {
    const model = await backend.train(DATASET, { maxOrder: 3 });
    const result = evaluateMovementModel(backend, model, [{ id: "eval", tokens: ["open"] }]);
    expect(result.predictions).toBe(0);
    expect(result.accuracy).toBe(0);
    expect(result.meanOrder).toBe(0);
  });
});
