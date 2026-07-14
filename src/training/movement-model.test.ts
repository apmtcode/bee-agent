import { describe, expect, it } from "vitest";
import { buildReplayManifest } from "../capture/replay.js";
import { buildTrajectorySpan, type TrajectoryAction } from "../capture/trajectory.js";
import {
  NGramMovementBackend,
  datasetFromReplayManifests,
  datasetFromTrajectories,
  evaluateReplayFidelity,
  type MovementDataset,
} from "./movement-model.js";

function action(tool: string, ts: number, summary = tool): TrajectoryAction {
  return { kind: "action", tool, summary, ts };
}

function sequenceDataset(sequences: string[][]): MovementDataset {
  return {
    sequences: sequences.map((tokens, index) => ({ id: `seq-${index}`, tokens })),
  };
}

describe("datasetFromTrajectories", () => {
  it("orders actions by timestamp and drops empty trajectories", () => {
    const dataset = datasetFromTrajectories([
      buildTrajectorySpan({
        id: "t1",
        sessionId: "s1",
        actions: [action("click", 30), action("open", 10), action("type", 20)],
      }),
      buildTrajectorySpan({ id: "empty", sessionId: "s1", actions: [] }),
    ]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]!.tokens).toEqual(["open", "type", "click"]);
  });

  it("supports tool+summary granularity for finer replay", () => {
    const dataset = datasetFromTrajectories(
      [
        buildTrajectorySpan({
          id: "t1",
          sessionId: "s1",
          actions: [action("click", 10, "button-a"), action("click", 20, "button-b")],
        }),
      ],
      { granularity: "tool+summary" },
    );
    expect(new Set(dataset.sequences[0]!.tokens).size).toBe(2);
  });
});

describe("datasetFromReplayManifests", () => {
  it("extracts action events from replay manifests", () => {
    const manifest = buildReplayManifest({
      sessionId: "s1",
      transcript: [],
      trajectories: [
        buildTrajectorySpan({
          id: "t1",
          sessionId: "s1",
          observations: [{ kind: "observation", source: "screen", summary: "window", ts: 5 }],
          actions: [action("open", 10), action("type", 20)],
        }),
      ],
    });
    const dataset = datasetFromReplayManifests([manifest]);
    expect(dataset.sequences[0]!.tokens).toEqual(["open", "type"]);
  });
});

describe("NGramMovementBackend", () => {
  it("repeats a recorded movement exactly via greedy generation", async () => {
    const recorded = ["open", "focus", "type", "submit"];
    const model = await new NGramMovementBackend().train(sequenceDataset([recorded]));
    expect(model.generate()).toEqual(recorded);
  });

  it("is deterministic across repeated rollouts", async () => {
    const model = await new NGramMovementBackend().train(
      sequenceDataset([
        ["a", "b", "c"],
        ["a", "b", "d"],
        ["a", "b", "c"],
      ]),
    );
    expect(model.generate()).toEqual(model.generate());
    // "c" appears twice after "a b", "d" once -> argmax picks "c".
    const prediction = model.predictNext(["a", "b"]);
    expect(prediction.token).toBe("c");
    expect(prediction.probability).toBeCloseTo(2 / 3);
  });

  it("generalizes to an unseen-but-related prefix via backoff", async () => {
    // The bigram "type -> submit" is learned; a novel prefix that has never
    // preceded "type" still yields "submit" as the next movement via backoff.
    const model = await new NGramMovementBackend().train(
      sequenceDataset([
        ["open", "type", "submit"],
        ["focus", "type", "submit"],
      ]),
      { order: 3 },
    );
    const prediction = model.predictNext(["navigate", "type"]);
    expect(prediction.token).toBe("submit");
    expect(prediction.order).toBeLessThan(3); // used a backed-off context
  });

  it("exposes vocabulary without leaking sentinels", async () => {
    const model = await new NGramMovementBackend().train(sequenceDataset([["open", "close"]]));
    expect(model.vocabulary).toEqual(["close", "open"]);
    expect(model.vocabulary.some((token) => token.includes("start") || token.includes("end"))).toBe(false);
  });

  it("round-trips through serialize/restore with identical behaviour", async () => {
    const backend = new NGramMovementBackend();
    const model = await backend.train(
      sequenceDataset([
        ["open", "type", "save"],
        ["open", "type", "save"],
        ["open", "scroll", "save"],
      ]),
    );
    const restored = backend.restore(model.serialize());
    expect(restored.generate()).toEqual(model.generate());
    expect(restored.predictNext(["open"]).token).toBe(model.predictNext(["open"]).token);
    expect(restored.serialize()).toEqual(model.serialize());
  });

  it("returns an empty prediction for an untrained model", async () => {
    const model = await new NGramMovementBackend().train({ sequences: [] });
    expect(model.predictNext(["anything"]).token).toBeUndefined();
    expect(model.generate()).toEqual([]);
  });

  it("honours the maxTokens generation bound", async () => {
    // A cyclic corpus would generate forever without the bound.
    const model = await new NGramMovementBackend().train(sequenceDataset([["a", "b", "a", "b", "a", "b"]]), {
      order: 2,
    });
    expect(model.generate([], { maxTokens: 4 }).length).toBeLessThanOrEqual(4);
  });
});

describe("evaluateReplayFidelity", () => {
  it("reports perfect fidelity when evaluating on the training set", async () => {
    const sequences = [
      ["open", "type", "submit"],
      ["open", "type", "submit"],
    ];
    const dataset = sequenceDataset(sequences);
    const model = await new NGramMovementBackend().train(dataset);
    const report = evaluateReplayFidelity(model, dataset.sequences);
    expect(report.tokenAccuracy).toBe(1);
    expect(report.sequenceExactMatch).toBe(1);
    expect(report.sequences).toBe(2);
  });

  it("measures partial generalization on held-out related sequences", async () => {
    // Train on a dominant pattern; hold out a related variant.
    const model = await new NGramMovementBackend().train(
      sequenceDataset([
        ["open", "type", "submit"],
        ["open", "type", "submit"],
        ["open", "type", "submit"],
      ]),
      { order: 3 },
    );
    const heldOut = [{ id: "held", tokens: ["open", "type", "cancel"] }];
    const report = evaluateReplayFidelity(model, heldOut);
    // Predicts open/type correctly; misses the novel final "cancel".
    expect(report.tokenAccuracy).toBeGreaterThan(0);
    expect(report.tokenAccuracy).toBeLessThan(1);
  });

  it("handles an empty held-out set", async () => {
    const model = await new NGramMovementBackend().train(sequenceDataset([["a"]]));
    const report = evaluateReplayFidelity(model, []);
    expect(report).toEqual({ tokenAccuracy: 0, sequenceExactMatch: 0, predictions: 0, sequences: 0 });
  });
});
