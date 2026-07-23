import { describe, expect, it } from "vitest";
import type { ReplayManifest } from "../capture/replay.js";
import {
  MarkovMovementBackend,
  buildMovementDataset,
  defaultMovementTokenizer,
  evaluateMovementModel,
  trainMovementPolicy,
  type MovementDataset,
  type MovementLearningBackend,
  type MovementSequence,
  type TrainedMovementModel,
} from "./movement-policy.js";

function replayFor(trajectoryId: string, tools: string[], startTs = 1): ReplayManifest {
  return {
    version: 1,
    sessionId: `sess-${trajectoryId}`,
    trajectoryIds: [trajectoryId],
    eventCount: tools.length,
    events: tools.map((tool, index) => ({
      kind: "action" as const,
      ts: startTs + index,
      trajectoryId,
      tool,
      summary: `did ${tool}`,
    })),
  };
}

function seq(trajectoryId: string, tools: string[]): MovementSequence {
  return { trajectoryId, tokens: tools.map((tool) => `action:${tool}`) };
}

describe("buildMovementDataset", () => {
  it("groups action events per trajectory and orders them by timestamp", () => {
    const dataset = buildMovementDataset([
      {
        version: 1,
        sessionId: "sess-1",
        trajectoryIds: ["traj-1"],
        eventCount: 3,
        events: [
          { kind: "action", ts: 3, trajectoryId: "traj-1", tool: "editor", summary: "type" },
          { kind: "observation", ts: 2, trajectoryId: "traj-1", source: "screen", summary: "saw" },
          { kind: "action", ts: 1, trajectoryId: "traj-1", tool: "browser", summary: "click" },
        ],
      },
    ]);

    expect(dataset.sequences).toEqual([
      { trajectoryId: "traj-1", tokens: ["action:browser", "action:editor"] },
    ]);
  });

  it("drops observations by default and honors minLength", () => {
    const dataset = buildMovementDataset([replayFor("traj-1", ["browser"]), replayFor("traj-2", ["a", "b"])], {
      minLength: 2,
    });
    expect(dataset.sequences.map((sequence) => sequence.trajectoryId)).toEqual(["traj-2"]);
  });

  it("supports a custom tokenizer", () => {
    const dataset = buildMovementDataset([replayFor("traj-1", ["browser", "editor"])], {
      tokenizer: (event) => (event.kind === "action" ? event.tool.toUpperCase() : undefined),
    });
    expect(dataset.sequences[0]?.tokens).toEqual(["BROWSER", "EDITOR"]);
  });

  it("has a default tokenizer that keeps only actions", () => {
    expect(defaultMovementTokenizer({ kind: "action", ts: 1, trajectoryId: "t", tool: "browser", summary: "x" })).toBe(
      "action:browser",
    );
    expect(defaultMovementTokenizer({ kind: "observation", ts: 1, trajectoryId: "t", source: "s", summary: "x" })).toBeUndefined();
  });
});

describe("MarkovMovementBackend", () => {
  const dataset: MovementDataset = {
    version: 1,
    sequences: [
      seq("traj-1", ["open", "search", "click", "submit"]),
      seq("traj-2", ["open", "search", "click", "submit"]),
      seq("traj-3", ["open", "search", "scroll", "submit"]),
    ],
  };

  it("repeats a recorded movement exactly for a seen prefix", () => {
    const model = new MarkovMovementBackend().fit(dataset, { order: 3 });
    const prediction = model.predict(["action:open", "action:search"]);
    // Full 2-context "open,search" is seen; majority next is "click" (2 of 3).
    expect(prediction.token).toBe("action:click");
    expect(prediction.backedOff).toBe(false);
    expect(prediction.contextOrderUsed).toBe(2);
    expect(prediction.confidence).toBeCloseTo(2 / 3);
  });

  it("generalizes to a novel-but-related prefix by backing off", () => {
    const model = new MarkovMovementBackend().fit(dataset, { order: 3 });
    // This exact 3-token context was never observed, but its suffix has.
    const prediction = model.predict(["action:launch", "action:open", "action:search"]);
    expect(prediction.token).toBe("action:click");
    expect(prediction.backedOff).toBe(true);
    expect(prediction.contextOrderUsed).toBeLessThan(3);
  });

  it("breaks ties deterministically by count then token", () => {
    const tied: MovementDataset = {
      version: 1,
      sequences: [seq("t1", ["a", "z"]), seq("t2", ["a", "m"])],
    };
    const model = new MarkovMovementBackend().fit(tied, { order: 2 });
    const prediction = model.predict(["action:a"]);
    // Both candidates count 1; lexicographically smaller token wins.
    expect(prediction.token).toBe("action:m");
    expect(prediction.candidates.map((candidate) => candidate.token)).toEqual(["action:m", "action:z"]);
  });

  it("rolls out a full recorded movement and stops at end-of-sequence", () => {
    const model = new MarkovMovementBackend().fit(dataset, { order: 3 });
    const produced = model.rollout(["action:open"], 10);
    expect(produced).toEqual(["action:search", "action:click", "action:submit"]);
  });

  it("returns an empty prediction for an empty model", () => {
    const model = new MarkovMovementBackend().fit({ version: 1, sequences: [] }, { order: 2 });
    const prediction = model.predict(["action:open"]);
    expect(prediction.token).toBeUndefined();
    expect(prediction.candidates).toEqual([]);
  });

  it("exposes a deterministic snapshot with sorted vocabulary", () => {
    const model = new MarkovMovementBackend().fit(dataset, { order: 3 });
    const snapshot = model.snapshot();
    expect(snapshot.backendId).toBe("markov-in-process");
    expect(snapshot.order).toBe(3);
    expect(snapshot.sequenceCount).toBe(3);
    expect(snapshot.vocabulary).toEqual([...snapshot.vocabulary].sort((a, b) => a.localeCompare(b)));
    expect(snapshot.vocabulary).toContain("action:scroll");
  });
});

describe("evaluateMovementModel", () => {
  const dataset: MovementDataset = {
    version: 1,
    sequences: [
      seq("traj-1", ["open", "search", "click", "submit"]),
      seq("traj-2", ["open", "search", "click", "submit"]),
    ],
  };

  it("scores repeat fidelity at 100% on the training sequences", () => {
    const model = trainMovementPolicy(dataset, { order: 3 });
    const evaluation = evaluateMovementModel(model, dataset.sequences);
    expect(evaluation.accuracy).toBe(1);
    expect(evaluation.exact.accuracy).toBe(1);
  });

  it("reports generalization on held-out related sequences", () => {
    const model = trainMovementPolicy(dataset, { order: 3 });
    // Related held-out trajectory with a novel opening step forcing backoff.
    const heldOut = [seq("held-1", ["launch", "open", "search", "click", "submit"])];
    const evaluation = evaluateMovementModel(model, heldOut);
    expect(evaluation.generalized.predictions).toBeGreaterThan(0);
    // The shared suffix (open→search→click→submit) is still predicted correctly.
    expect(evaluation.generalized.correct).toBeGreaterThan(0);
  });
});

describe("pluggable backend seam", () => {
  it("accepts an alternative backend implementing the interface", () => {
    // A trivial "always predict the globally most frequent token" backend,
    // proving fit/predict are swappable without touching call sites.
    class ConstantBackend implements MovementLearningBackend {
      readonly id = "constant-test";
      fit(dataset: MovementDataset): TrainedMovementModel {
        const counts = new Map<string, number>();
        for (const sequence of dataset.sequences) {
          for (const token of sequence.tokens) {
            counts.set(token, (counts.get(token) ?? 0) + 1);
          }
        }
        const best = [...counts.entries()].sort((a, b) => (a[1] !== b[1] ? b[1] - a[1] : a[0].localeCompare(b[0])))[0]?.[0];
        const vocabulary = [...counts.keys()].sort((a, b) => a.localeCompare(b));
        return {
          backendId: this.id,
          order: 0,
          vocabulary,
          predict: () => ({ token: best, confidence: 1, contextOrderUsed: 0, backedOff: false, candidates: [] }),
          rollout: (_context, steps) => (best ? Array.from({ length: steps }, () => best) : []),
          snapshot: () => ({
            version: 1,
            backendId: this.id,
            order: 0,
            vocabulary,
            sequenceCount: dataset.sequences.length,
            tokenCount: [...counts.values()].reduce((sum, value) => sum + value, 0),
          }),
        };
      }
    }

    const model = trainMovementPolicy(
      { version: 1, sequences: [seq("t1", ["a", "a", "b"])] },
      { backend: new ConstantBackend() },
    );
    expect(model.backendId).toBe("constant-test");
    expect(model.predict([]).token).toBe("action:a");
  });
});
