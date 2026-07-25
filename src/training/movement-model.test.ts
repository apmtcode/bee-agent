import { describe, expect, it } from "vitest";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import {
  NGramMovementBackend,
  buildMovementDataset,
  evaluateMovementModel,
  type MovementDataset,
  type MovementSequence,
} from "./movement-model.js";

function seq(trajectoryId: string, tools: string[]): MovementSequence {
  return {
    trajectoryId,
    tokens: tools.map((tool, index) => ({ tool, summary: `${tool}#${index}` })),
  };
}

describe("buildMovementDataset", () => {
  it("orders actions by timestamp and prefers redacted actions", () => {
    const withRedaction = buildTrajectorySpan({
      id: "traj-1",
      sessionId: "sess-1",
      actions: [
        { kind: "action", tool: "mouse", summary: "raw move", ts: 30 },
        { kind: "action", tool: "keyboard", summary: "raw type", ts: 10 },
      ],
    });
    withRedaction.review = {
      status: "approved",
      reviewedAt: "2026-01-01T00:00:00.000Z",
      reviewedBy: "reviewer",
      redactedActions: [
        { ts: 20, tool: "keyboard", summary: "type X" },
        { ts: 10, tool: "mouse", summary: "click open" },
      ],
    };

    const dataset = buildMovementDataset([withRedaction]);
    expect(dataset.sequences).toHaveLength(1);
    // Redacted actions are used, sorted by ts (10 before 20).
    expect(dataset.sequences[0]?.tokens.map((token) => token.tool)).toEqual(["mouse", "keyboard"]);
  });

  it("falls back to raw actions and drops trajectories with no actions", () => {
    const populated = buildTrajectorySpan({
      id: "traj-a",
      sessionId: "sess-a",
      actions: [
        { kind: "action", tool: "b", summary: "b", ts: 2 },
        { kind: "action", tool: "a", summary: "a", ts: 1 },
      ],
    });
    const empty = buildTrajectorySpan({ id: "traj-b", sessionId: "sess-b" });

    const dataset = buildMovementDataset([populated, empty]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]?.tokens.map((token) => token.tool)).toEqual(["a", "b"]);
  });
});

describe("NGramMovementBackend", () => {
  it("repeats a recorded movement sequence exactly via rollout", () => {
    const dataset: MovementDataset = {
      version: 1,
      sequences: [seq("traj-1", ["open", "type", "select", "save"])],
    };
    const model = new NGramMovementBackend().train(dataset, { order: 2 });

    const rolled = model.rollout([{ tool: "open", summary: "open#0" }], 10);
    expect(rolled.map((token) => token.tool)).toEqual(["type", "select", "save"]);
  });

  it("recovers a concrete replayable summary, not just the tool", () => {
    const dataset: MovementDataset = {
      version: 1,
      sequences: [
        {
          trajectoryId: "traj-1",
          tokens: [
            { tool: "mouse", summary: "move to (10,20)" },
            { tool: "mouse", summary: "click left" },
          ],
        },
      ],
    };
    const model = new NGramMovementBackend().train(dataset, { order: 1 });
    const prediction = model.predict([{ tool: "mouse", summary: "move to (10,20)" }]);
    expect(prediction?.tool).toBe("mouse");
    expect(prediction?.summary).toBe("click left");
  });

  it("terminates rollout at the learned end of a sequence", () => {
    const dataset: MovementDataset = {
      version: 1,
      sequences: [seq("traj-1", ["a", "b"])],
    };
    const model = new NGramMovementBackend().train(dataset, { order: 2 });
    // Ask for far more steps than the sequence length; it should stop after "b".
    const rolled = model.rollout([{ tool: "a", summary: "a#0" }], 25);
    expect(rolled.map((token) => token.tool)).toEqual(["b"]);
  });

  it("generalizes to an unseen-but-related prefix via back-off", () => {
    // Both training sequences share the sub-pattern "type -> submit".
    const dataset: MovementDataset = {
      version: 1,
      sequences: [
        seq("traj-1", ["open", "type", "submit"]),
        seq("traj-2", ["focus", "type", "submit"]),
      ],
    };
    const model = new NGramMovementBackend().train(dataset, { order: 2 });

    // The prefix ["scroll", "type"] was never seen at order-2; the model must
    // back off to the order-1 context "type" and still predict "submit".
    const prediction = model.predict([
      { tool: "scroll", summary: "scroll down" },
      { tool: "type", summary: "type hello" },
    ]);
    expect(prediction?.tool).toBe("submit");
    expect(prediction?.order).toBe(1);
    expect(prediction && prediction.confidence).toBeGreaterThan(0);
  });

  it("is deterministic: identical datasets yield byte-identical models", () => {
    const dataset: MovementDataset = {
      version: 1,
      sequences: [
        seq("traj-1", ["a", "b", "c"]),
        seq("traj-2", ["a", "b", "d"]),
        seq("traj-3", ["a", "b", "c"]),
      ],
    };
    const backend = new NGramMovementBackend();
    const first = backend.train(dataset, { order: 2 }).serialize();
    const second = backend.train(dataset, { order: 2 }).serialize();
    expect(JSON.stringify(first)).toBe(JSON.stringify(second));
  });

  it("breaks frequency ties toward the more common successor", () => {
    // After "a", "b" appears twice and "c" once -> "b" wins.
    const dataset: MovementDataset = {
      version: 1,
      sequences: [seq("t1", ["a", "b"]), seq("t2", ["a", "b"]), seq("t3", ["a", "c"])],
    };
    const model = new NGramMovementBackend().train(dataset, { order: 1 });
    const prediction = model.predict([{ tool: "a", summary: "a#0" }]);
    expect(prediction?.tool).toBe("b");
    expect(prediction?.confidence).toBeCloseTo(2 / 3, 5);
  });

  it("round-trips through serialize/load with identical predictions", () => {
    const dataset: MovementDataset = {
      version: 1,
      sequences: [seq("t1", ["open", "type", "save"]), seq("t2", ["open", "type", "close"])],
    };
    const backend = new NGramMovementBackend();
    const model = backend.train(dataset, { order: 2 });
    const reloaded = backend.load(model.serialize());

    for (const prefix of [
      [{ tool: "open", summary: "open#0" }],
      [
        { tool: "open", summary: "open#0" },
        { tool: "type", summary: "type#1" },
      ],
    ]) {
      expect(reloaded.predict(prefix)).toEqual(model.predict(prefix));
    }
    expect(JSON.stringify(reloaded.serialize())).toBe(JSON.stringify(model.serialize()));
  });

  it("returns undefined for an empty, untrained context", () => {
    const model = new NGramMovementBackend().train({ version: 1, sequences: [] }, { order: 2 });
    expect(model.predict([])).toBeUndefined();
    expect(model.rollout([], 5)).toEqual([]);
  });
});

describe("evaluateMovementModel", () => {
  it("reports perfect reproduction on the training sequence", () => {
    const dataset: MovementDataset = {
      version: 1,
      sequences: [seq("t1", ["open", "type", "select", "save"])],
    };
    const model = new NGramMovementBackend().train(dataset, { order: 2 });
    const report = evaluateMovementModel(model, dataset.sequences);
    expect(report.tokenAccuracy).toBe(1);
    expect(report.exactMatchRate).toBe(1);
    expect(report.sequences).toBe(1);
  });

  it("measures partial generalization on held-out related sequences", () => {
    const train: MovementDataset = {
      version: 1,
      sequences: [seq("t1", ["open", "type", "submit"]), seq("t2", ["focus", "type", "submit"])],
    };
    const model = new NGramMovementBackend().train(train, { order: 2 });
    // Related but unseen: shares the "type -> submit" tail.
    const heldOut = [seq("h1", ["scroll", "type", "submit"])];
    const report = evaluateMovementModel(model, heldOut);
    // "type -> submit" is recovered by back-off, so at least one prediction is correct.
    expect(report.correctPredictions).toBeGreaterThan(0);
    expect(report.tokenAccuracy).toBeGreaterThan(0);
  });
});
