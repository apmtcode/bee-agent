import { describe, expect, it } from "vitest";
import {
  MarkovMovementBackend,
  decodeMovementToken,
  evaluateNextTokenAccuracy,
  evaluateReplayFidelity,
  extractMovementDataset,
  extractMovementSequence,
  loadMovementModel,
  movementActionToken,
  movementSequenceFromReplay,
  type MovementDataset,
} from "./movement-model.js";
import { buildReplayManifest } from "../capture/replay.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";

function seq(id: string, tokens: string[], context?: string) {
  return context ? { id, tokens, context } : { id, tokens };
}

describe("movement token codec", () => {
  it("round-trips tool/summary through a reversible token", () => {
    const token = movementActionToken({ tool: "device", summary: "tapped Submit" });
    expect(decodeMovementToken(token)).toEqual({ tool: "device", summary: "tapped Submit" });
  });
});

describe("dataset extraction", () => {
  it("orders trajectory actions by timestamp and drops empty sequences", () => {
    const withActions = buildTrajectorySpan({
      id: "traj-1",
      sessionId: "sess-1",
      actions: [
        { kind: "action", tool: "device", summary: "click B", ts: 30 },
        { kind: "action", tool: "device", summary: "click A", ts: 10 },
      ],
      outcome: { status: "success", summary: "saved", reward: 1 },
    });
    const empty = buildTrajectorySpan({ id: "traj-2", sessionId: "sess-1" });

    const dataset = extractMovementDataset([withActions, empty]);
    expect(dataset.sequences).toHaveLength(1);
    const extracted = extractMovementSequence(withActions);
    expect(extracted.context).toBe("saved");
    expect(extracted.tokens.map((token) => decodeMovementToken(token).summary)).toEqual(["click A", "click B"]);
  });

  it("derives a movement sequence from a replay manifest action timeline", () => {
    const trajectory = buildTrajectorySpan({
      id: "traj-1",
      sessionId: "sess-1",
      observations: [{ kind: "observation", source: "browser", summary: "noise", ts: 5 }],
      actions: [
        { kind: "action", tool: "browser", summary: "open page", ts: 10 },
        { kind: "action", tool: "browser", summary: "click deploy", ts: 20 },
      ],
    });
    const manifest = buildReplayManifest({ sessionId: "sess-1", transcript: [], trajectories: [trajectory] });
    const sequence = movementSequenceFromReplay(manifest);
    expect(sequence.tokens.map((token) => decodeMovementToken(token).summary)).toEqual(["open page", "click deploy"]);
  });
});

describe("MarkovMovementBackend", () => {
  const A = "a";
  const B = "b";
  const C = "c";
  const D = "d";

  it("replays a recorded sequence exactly (objective c)", async () => {
    const dataset: MovementDataset = { version: 1, sequences: [seq("s1", [A, B, C, D])] };
    const model = await new MarkovMovementBackend({ order: 2 }).train(dataset);
    expect(model.generate([A])).toEqual([A, B, C, D]);
    expect(evaluateReplayFidelity(model, dataset.sequences[0]!).fidelity).toBe(1);
  });

  it("is deterministic: predictNext is ranked by count then token, sums to ~1", async () => {
    const dataset: MovementDataset = {
      version: 1,
      sequences: [seq("s1", [A, B]), seq("s2", [A, B]), seq("s3", [A, C])],
    };
    const model = await new MarkovMovementBackend({ order: 1 }).train(dataset);
    const predictions = model.predictNext([A]);
    expect(predictions.map((p) => p.token)).toEqual([B, C]);
    expect(predictions[0]!.probability).toBeCloseTo(2 / 3, 10);
    expect(predictions.reduce((sum, p) => sum + p.probability, 0)).toBeCloseTo(1, 10);
  });

  it("generalizes by recombining learned transitions into a new sequence (objective d)", async () => {
    // Train on related movements that share the middle hop B -> C -> D.
    const dataset: MovementDataset = {
      version: 1,
      sequences: [seq("s1", [A, B, C]), seq("s2", [B, C, D]), seq("s3", [B, C, D])],
    };
    const model = await new MarkovMovementBackend({ order: 1 }).train(dataset);
    // From A the order-1 policy recombines A->B->C->D, a sequence never seen verbatim.
    const generated = model.generate([A]);
    expect(generated).toEqual([A, B, C, D]);
    // Next-token generalization to an unseen prefix ending in a known context.
    expect(model.predictNext([D, B], { topK: 1 })[0]?.token).toBe(C);
  });

  it("stops generation at the learned end-of-sequence", async () => {
    const dataset: MovementDataset = { version: 1, sequences: [seq("s1", [A, B])] };
    const model = await new MarkovMovementBackend({ order: 2 }).train(dataset);
    expect(model.generate([A], { maxLength: 50 })).toEqual([A, B]);
  });

  it("exposes a clean vocabulary without boundary markers", async () => {
    const dataset: MovementDataset = { version: 1, sequences: [seq("s1", [B, A, C])] };
    const model = await new MarkovMovementBackend({ order: 2 }).train(dataset);
    expect(model.vocabulary).toEqual([A, B, C]);
  });

  it("serializes and reloads to an identical policy", async () => {
    const dataset: MovementDataset = {
      version: 1,
      sequences: [seq("s1", [A, B, C, D]), seq("s2", [A, B, D])],
    };
    const model = await new MarkovMovementBackend({ order: 2 }).train(dataset);
    const reloaded = loadMovementModel(model.serialize());
    expect(reloaded.order).toBe(model.order);
    expect(reloaded.vocabulary).toEqual(model.vocabulary);
    expect(reloaded.generate([A])).toEqual(model.generate([A]));
    expect(reloaded.predictNext([A, B])).toEqual(model.predictNext([A, B]));
    expect(reloaded.serialize()).toEqual(model.serialize());
  });

  it("clamps invalid order and falls back gracefully on unknown context", async () => {
    const model = await new MarkovMovementBackend({ order: 0 }).train({
      version: 1,
      sequences: [seq("s1", [A, B])],
    });
    expect(model.order).toBe(1);
    // Unknown context backs off to the unigram distribution rather than failing.
    expect(model.predictNext(["never-seen"]).length).toBeGreaterThan(0);
  });
});

describe("evaluation harness", () => {
  it("reports perfect next-token accuracy on training data and partial on held-out", async () => {
    const train: MovementDataset = {
      version: 1,
      sequences: [seq("s1", ["a", "b", "c"]), seq("s2", ["a", "b", "c"])],
    };
    const model = await new MarkovMovementBackend({ order: 2 }).train(train);
    const onTrain = evaluateNextTokenAccuracy(model, train.sequences);
    expect(onTrain.accuracy).toBe(1);

    const heldOut = [seq("h1", ["a", "b", "z"])];
    const evaluation = evaluateNextTokenAccuracy(model, heldOut);
    // a, b predicted correctly; final z unseen -> one miss recorded.
    expect(evaluation.total).toBe(3);
    expect(evaluation.misses).toHaveLength(1);
    expect(evaluation.misses[0]).toMatchObject({ sequenceId: "h1", position: 2, expected: "z" });
  });
});
