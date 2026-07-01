import { describe, expect, it } from "vitest";
import {
  DeterministicMarkovBackend,
  MOVEMENT_END,
  MOVEMENT_START,
  buildMovementDataset,
  buildMovementSequencesFromReplays,
  buildMovementSequencesFromTrajectories,
  loadMovementModel,
  movementToken,
} from "./movement-model.js";
import {
  evaluateDatasetReplayFidelity,
  evaluateNextActionAccuracy,
  evaluateReplayFidelity,
  generateSyntheticMovementDataset,
  generateSyntheticMovementSequences,
  partitionMovementDataset,
  splitMovementDataset,
} from "./movement-eval.js";
import type { ExportedReplayManifest } from "./export-manifest.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

function span(id: string, actions: Array<{ tool: string; summary: string; ts: number }>): TrajectorySpan {
  return {
    id,
    sessionId: `session-${id}`,
    createdAt: "2026-07-01T00:00:00.000Z",
    captureTier: "operator",
    observations: [],
    actions: actions.map((action) => ({ kind: "action", ...action })),
  };
}

describe("movementToken", () => {
  it("normalizes summaries into stable, comparable tokens", () => {
    expect(movementToken("device", "Tapped  Submit!")).toBe("device::tapped submit");
    expect(movementToken("device", "tapped submit")).toBe(movementToken("device", "Tapped Submit"));
    expect(movementToken("device", "")).toBe("device::");
  });
});

describe("buildMovementSequences", () => {
  it("extracts ordered action steps from trajectory spans", () => {
    const sequences = buildMovementSequencesFromTrajectories([
      span("t1", [
        { tool: "device", summary: "tapped send", ts: 20 },
        { tool: "os", summary: "focused mail", ts: 10 },
      ]),
      span("empty", []),
    ]);
    expect(sequences).toHaveLength(1);
    expect(sequences[0]!.steps.map((step) => step.summary)).toEqual(["focused mail", "tapped send"]);
  });

  it("groups replay action events by trajectory id and orders by ts", () => {
    const replays: ExportedReplayManifest[] = [
      {
        sessionId: "s1",
        trajectoryIds: ["a", "b"],
        eventCount: 4,
        events: [
          { kind: "transcript", ts: 1, messageId: "m", role: "user", content: "hi" },
          { kind: "action", ts: 5, trajectoryId: "a", tool: "device", summary: "tapped one" },
          { kind: "action", ts: 2, trajectoryId: "a", tool: "device", summary: "tapped zero" },
          { kind: "action", ts: 3, trajectoryId: "b", tool: "os", summary: "focused app" },
        ],
      },
    ];
    const sequences = buildMovementSequencesFromReplays(replays);
    const a = sequences.find((sequence) => sequence.trajectoryId === "a");
    expect(a!.steps.map((step) => step.summary)).toEqual(["tapped zero", "tapped one"]);
    expect(sequences.find((sequence) => sequence.trajectoryId === "b")!.steps).toHaveLength(1);
  });
});

describe("DeterministicMarkovBackend", () => {
  it("is deterministic: same dataset yields identical serialized models", async () => {
    const dataset = generateSyntheticMovementDataset({ seed: 7, sequenceCount: 9 });
    const backend = new DeterministicMarkovBackend();
    const a = await backend.train(dataset);
    const b = await backend.train(dataset);
    expect(a.serialize()).toEqual(b.serialize());
  });

  it("memorizes a recorded movement and replays it exactly (objective 2c)", async () => {
    const sequences = buildMovementSequencesFromTrajectories([
      span("t1", [
        { tool: "os", summary: "focused mail", ts: 1 },
        { tool: "device", summary: "tapped compose", ts: 2 },
        { tool: "device", summary: "typed body", ts: 3 },
        { tool: "device", summary: "tapped send", ts: 4 },
      ]),
    ]);
    const model = await new DeterministicMarkovBackend().train(buildMovementDataset(sequences));
    const fidelity = evaluateReplayFidelity(model, sequences[0]!);
    expect(fidelity.exact).toBe(true);
    expect(fidelity.fidelity).toBe(1);
    expect(fidelity.produced).toEqual(sequences[0]!.steps.map((step) => step.token));
  });

  it("predicts the next movement and exposes ranked candidates", async () => {
    const dataset = buildMovementDataset(
      buildMovementSequencesFromTrajectories([
        span("t1", [
          { tool: "os", summary: "focused mail", ts: 1 },
          { tool: "device", summary: "tapped send", ts: 2 },
        ]),
        span("t2", [
          { tool: "os", summary: "focused mail", ts: 1 },
          { tool: "device", summary: "tapped send", ts: 2 },
        ]),
      ]),
    );
    const model = await new DeterministicMarkovBackend().train(dataset);
    const prediction = model.predictNext([MOVEMENT_START, movementToken("os", "focused mail")]);
    expect(prediction.token).toBe(movementToken("device", "tapped send"));
    expect(prediction.confidence).toBeGreaterThan(0);
    expect(prediction.candidates[0]!.token).toBe(movementToken("device", "tapped send"));
  });

  it("backs off to a shorter shared context to generalize to unseen prefixes (objective 2d)", async () => {
    // Train only on prefix A -> mid -> tail. Query with a novel prefix B -> mid.
    const dataset = buildMovementDataset(
      buildMovementSequencesFromTrajectories([
        span("seen", [
          { tool: "os", summary: "focused mail", ts: 1 },
          { tool: "device", summary: "opened editor", ts: 2 },
          { tool: "device", summary: "tapped send", ts: 3 },
        ]),
      ]),
    );
    const model = await new DeterministicMarkovBackend().train(dataset);
    const novelContext = [movementToken("browser", "opened checkout"), movementToken("device", "opened editor")];
    const prediction = model.predictNext(novelContext);
    // Full 2-gram context is unseen, but the length-1 suffix "opened editor" is known.
    expect(prediction.token).toBe(movementToken("device", "tapped send"));
    expect(prediction.matchedContextLength).toBe(1);
  });

  it("round-trips through serialize/load", async () => {
    const dataset = generateSyntheticMovementDataset({ seed: 3 });
    const model = await new DeterministicMarkovBackend().train(dataset);
    const restored = loadMovementModel(model.serialize());
    expect(restored.serialize()).toEqual(model.serialize());
    const context = [MOVEMENT_START];
    expect(restored.predictNext(context)).toEqual(model.predictNext(context));
  });

  it("halts rollout at the end sentinel rather than looping forever", async () => {
    const dataset = buildMovementDataset(
      buildMovementSequencesFromTrajectories([span("t1", [{ tool: "device", summary: "tapped ok", ts: 1 }])]),
    );
    const model = await new DeterministicMarkovBackend().train(dataset);
    const produced = model.rollout([MOVEMENT_START], { maxSteps: 50 });
    expect(produced).toEqual([movementToken("device", "tapped ok")]);
    expect(produced).not.toContain(MOVEMENT_END);
  });
});

describe("synthetic generator", () => {
  it("is deterministic for a fixed seed", () => {
    const a = generateSyntheticMovementSequences({ seed: 42, sequenceCount: 6 });
    const b = generateSyntheticMovementSequences({ seed: 42, sequenceCount: 6 });
    expect(a).toEqual(b);
  });

  it("produces different data for different seeds", () => {
    const a = generateSyntheticMovementSequences({ seed: 1, sequenceCount: 6 });
    const b = generateSyntheticMovementSequences({ seed: 2, sequenceCount: 6 });
    expect(a).not.toEqual(b);
  });
});

describe("eval harness", () => {
  it("splits deterministically and trains only on the train split", () => {
    const dataset = generateSyntheticMovementDataset({ seed: 5, sequenceCount: 12 });
    const split = splitMovementDataset(dataset, { testStride: 4 });
    expect(split.train.sequences.length + split.test.sequences.length).toBe(12);
    expect(split.test.sequences).toHaveLength(3);
    // Reproducible split membership.
    expect(splitMovementDataset(dataset, { testStride: 4 })).toEqual(split);
  });

  it("partitions by explicit hold-out trajectory ids", () => {
    const dataset = generateSyntheticMovementDataset({ seed: 5, sequenceCount: 6 });
    const holdOut = [dataset.sequences[0]!.trajectoryId, dataset.sequences[1]!.trajectoryId];
    const split = partitionMovementDataset(dataset, holdOut);
    expect(split.test.sequences.map((sequence) => sequence.trajectoryId).sort()).toEqual([...holdOut].sort());
    expect(split.train.sequences).toHaveLength(4);
  });

  it("generalizes: held-out related trajectories score better than chance", async () => {
    // Many related sequences drawn from the same flow grammar.
    const dataset = generateSyntheticMovementDataset({ seed: 11, sequenceCount: 60 });
    const split = splitMovementDataset(dataset, { testStride: 5 });
    const model = await new DeterministicMarkovBackend(2).train(split.train);
    const accuracy = evaluateNextActionAccuracy(model, split.test.sequences, { topK: 3 });
    expect(accuracy.positions).toBeGreaterThan(0);
    // Shared flow structure + back-off should beat a low baseline on held-out data.
    expect(accuracy.topKAccuracy).toBeGreaterThan(0.5);
    expect(accuracy.top1Accuracy).toBeGreaterThan(accuracy.unseenContexts / accuracy.positions);
  });

  it("reproduces distinct (non-conflicting) recorded movements exactly", async () => {
    // Each sequence uses unique tokens, so greedy rollout has no ambiguity.
    const sequences = buildMovementSequencesFromTrajectories(
      Array.from({ length: 12 }, (_, index) =>
        span(`t${index}`, [
          { tool: "os", summary: `focused window ${index}`, ts: 1 },
          { tool: "device", summary: `tapped field ${index}`, ts: 2 },
          { tool: "device", summary: `typed value ${index}`, ts: 3 },
          { tool: "device", summary: `tapped submit ${index}`, ts: 4 },
        ]),
      ),
    );
    const model = await new DeterministicMarkovBackend(3).train(buildMovementDataset(sequences));
    const report = evaluateDatasetReplayFidelity(model, sequences);
    expect(report.sequences).toBe(12);
    expect(report.exactMatches).toBe(12);
    expect(report.meanFidelity).toBe(1);
  });

  it("regresses to the dominant path on shared, conflicting flow prefixes", async () => {
    // Shared-prefix synthetic sequences conflict, so per-sequence recall is
    // partial — greedy rollout follows the most-common continuation.
    const dataset = generateSyntheticMovementDataset({ seed: 9, sequenceCount: 12 });
    const model = await new DeterministicMarkovBackend(3).train(dataset);
    const report = evaluateDatasetReplayFidelity(model, dataset.sequences);
    expect(report.meanFidelity).toBeGreaterThan(0.4);
    expect(report.meanFidelity).toBeLessThanOrEqual(1);
  });
});
