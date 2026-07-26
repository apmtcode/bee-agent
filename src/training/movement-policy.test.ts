import { describe, expect, it } from "vitest";
import { buildTrajectorySpan, type TrajectoryAction } from "../capture/trajectory.js";
import {
  NgramMovementPolicyBackend,
  buildMovementDataset,
  evaluateMovementPolicy,
  generateSyntheticMovementDataset,
  movementSequenceFromReplay,
  movementSequenceFromTrajectory,
  type MovementDataset,
} from "./movement-policy.js";

function action(tool: string, ts: number): TrajectoryAction {
  return { kind: "action", tool, summary: `${tool}@${ts}`, ts };
}

describe("movement dataset extraction", () => {
  it("orders actions by timestamp and tokenizes by tool", () => {
    const span = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [action("click", 30), action("move", 10), action("type", 20)],
    });
    const sequence = movementSequenceFromTrajectory(span);
    expect(sequence.tokens).toEqual(["move", "type", "click"]);
  });

  it("uses redacted actions when a review redaction is present", () => {
    const span = buildTrajectorySpan({ id: "t2", sessionId: "s1", actions: [action("secret", 1)] });
    span.review = {
      status: "approved",
      reviewedAt: "2026-01-01T00:00:00Z",
      reviewedBy: "reviewer",
      redactedActions: [
        { ts: 5, tool: "move", summary: "safe" },
        { ts: 6, tool: "click", summary: "safe" },
      ],
    };
    expect(movementSequenceFromTrajectory(span).tokens).toEqual(["move", "click"]);
  });

  it("drops empty sequences when building a dataset", () => {
    const withActions = buildTrajectorySpan({ id: "a", sessionId: "s", actions: [action("move", 1)] });
    const empty = buildTrajectorySpan({ id: "b", sessionId: "s", actions: [] });
    const dataset = buildMovementDataset([withActions, empty]);
    expect(dataset.sequences.map((sequence) => sequence.id)).toEqual(["a"]);
  });

  it("extracts action tokens from a replay timeline", () => {
    const sequence = movementSequenceFromReplay("r1", [
      { kind: "transcript", ts: 1, messageId: "m", role: "user", content: "hi" },
      { kind: "observation", ts: 2, trajectoryId: "t", source: "screen", summary: "obs" },
      { kind: "action", ts: 3, trajectoryId: "t", tool: "click", summary: "c" },
      { kind: "action", ts: 4, trajectoryId: "t", tool: "type", summary: "t" },
    ]);
    expect(sequence.tokens).toEqual(["click", "type"]);
  });
});

describe("NgramMovementPolicyBackend", () => {
  const backend = new NgramMovementPolicyBackend();

  it("repeats a recorded movement exactly for a seen context", () => {
    const dataset: MovementDataset = {
      sequences: [{ id: "seq", tokens: ["move", "click", "type", "submit"] }],
    };
    const model = backend.train(dataset, { maxOrder: 3 });

    const prediction = backend.predict(model, ["move", "click", "type"]);
    expect(prediction.token).toBe("submit");
    expect(prediction.probability).toBe(1);
    expect(prediction.fromBackoff).toBe(false);
    expect(backend.hasContext(model, ["move", "click", "type"])).toBe(true);
  });

  it("generalizes via backoff when the full context is novel", () => {
    // "type" is always followed by "submit" across examples, but the exact
    // 3-token prefix below was never seen — the model must back off to it.
    const dataset: MovementDataset = {
      sequences: [
        { id: "a", tokens: ["move", "type", "submit"] },
        { id: "b", tokens: ["click", "type", "submit"] },
        { id: "c", tokens: ["scroll", "type", "submit"] },
      ],
    };
    const model = backend.train(dataset, { maxOrder: 3 });

    const novelContext = ["drag", "focus", "type"];
    expect(backend.hasContext(model, novelContext)).toBe(false);
    const prediction = backend.predict(model, novelContext);
    expect(prediction.token).toBe("submit");
    expect(prediction.fromBackoff).toBe(true);
    expect(prediction.contextOrder).toBe(1);
  });

  it("falls back to the global next-move frequency for an unseen suffix", () => {
    const dataset: MovementDataset = {
      sequences: [{ id: "seq", tokens: ["move", "move", "move", "click"] }],
    };
    const model = backend.train(dataset, { maxOrder: 2 });
    const prediction = backend.predict(model, ["never-seen-token"]);
    expect(prediction.token).toBe("move");
    expect(prediction.contextOrder).toBe(0);
    expect(prediction.fromBackoff).toBe(true);
  });

  it("ranks candidates deterministically with a stable tie-break", () => {
    const dataset: MovementDataset = {
      sequences: [
        { id: "a", tokens: ["start", "beta"] },
        { id: "b", tokens: ["start", "alpha"] },
      ],
    };
    const model = backend.train(dataset, { maxOrder: 1 });
    const prediction = backend.predict(model, ["start"]);
    // Equal counts -> lexicographic tie-break -> "alpha" first.
    expect(prediction.ranked.map((candidate) => candidate.token)).toEqual(["alpha", "beta"]);
    expect(prediction.token).toBe("alpha");
  });

  it("returns an empty prediction for an untrained model", () => {
    const model = backend.train({ sequences: [] });
    const prediction = backend.predict(model, ["anything"]);
    expect(prediction.token).toBeUndefined();
    expect(prediction.ranked).toEqual([]);
  });
});

describe("evaluateMovementPolicy on synthetic streams", () => {
  const backend = new NgramMovementPolicyBackend();

  it("generates deterministic datasets for a fixed seed", () => {
    const a = generateSyntheticMovementDataset({ seed: 42, sequenceCount: 5 });
    const b = generateSyntheticMovementDataset({ seed: 42, sequenceCount: 5 });
    expect(a).toEqual(b);
    expect(a.sequences).toHaveLength(5);
    expect(a.sequences.every((sequence) => sequence.tokens.length >= 2)).toBe(true);
  });

  it("learns motif structure: high repeat fidelity and non-trivial generalization", () => {
    const train = generateSyntheticMovementDataset({ seed: 7, sequenceCount: 40, motifCount: 3 });
    const test = generateSyntheticMovementDataset({ seed: 99, sequenceCount: 20, motifCount: 3 });
    const model = backend.train(train, { maxOrder: 3 });
    const result = evaluateMovementPolicy(backend, model, test);

    expect(result.total).toBeGreaterThan(0);
    expect(result.seenContext.total + result.novelContext.total).toBe(result.total);
    // A model that memorized recurring motifs should reproduce seen contexts
    // near-perfectly...
    expect(result.seenContext.accuracy).toBeGreaterThan(0.8);
    // ...and still beat chance on novel contexts by generalizing via backoff.
    expect(result.novelContext.total).toBeGreaterThan(0);
    expect(result.novelContext.accuracy).toBeGreaterThan(0.2);
    // Overall the trained policy is meaningfully better than the ~1/8 uniform base rate.
    expect(result.accuracy).toBeGreaterThan(0.4);
  });

  it("is pluggable: any backend implementing the interface can be evaluated", () => {
    // A trivial constant backend proves the eval harness is backend-agnostic.
    const constant = {
      id: "constant",
      train: () => ({ token: "move" }),
      predict: (model: { token: string }) => ({
        token: model.token,
        probability: 1,
        contextOrder: 0,
        fromBackoff: false,
        ranked: [{ token: model.token, probability: 1 }],
      }),
      hasContext: () => false,
    };
    const dataset: MovementDataset = { sequences: [{ id: "s", tokens: ["move", "move", "click"] }] };
    const model = constant.train();
    const result = evaluateMovementPolicy(constant, model, dataset);
    // Predicts "move" for both positions; correct once (move) wrong once (click).
    expect(result.total).toBe(2);
    expect(result.correct).toBe(1);
    expect(result.novelContext.total).toBe(2);
  });
});
