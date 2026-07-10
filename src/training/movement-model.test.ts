import { describe, expect, it } from "vitest";
import type { ReplayManifest } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  MarkovMovementBackend,
  MarkovMovementModel,
  buildMovementDatasetFromReplays,
  buildMovementDatasetFromTrajectories,
  evaluateMovementModel,
  generateSyntheticMovementSequences,
  tokenizeAction,
  tokenizeObservation,
  type MovementDataset,
} from "./movement-model.js";

function span(id: string, actions: TrajectorySpan["actions"], observations: TrajectorySpan["observations"] = []): TrajectorySpan {
  return {
    id,
    sessionId: `session-${id}`,
    createdAt: "2026-07-10T00:00:00.000Z",
    captureTier: "app",
    observations,
    actions,
  };
}

describe("tokenization", () => {
  it("builds structured device tokens from gesture metadata", () => {
    expect(
      tokenizeAction({ tool: "device", summary: "swiped up", metadata: { gesture: "swipe", direction: "up" } }),
    ).toBe("device:swipe:up");
    expect(
      tokenizeAction({ tool: "device", summary: "tapped submit", metadata: { gesture: "tap", target: "submit" } }),
    ).toBe("device:tap");
  });

  it("falls back to a tool token for non-gesture actions", () => {
    expect(tokenizeAction({ tool: "bash", summary: "ran ls", metadata: undefined })).toBe("act:bash");
  });

  it("omits direction when requested", () => {
    expect(
      tokenizeAction(
        { tool: "device", summary: "swiped up", metadata: { gesture: "swipe", direction: "up" } },
        { includeDirection: false },
      ),
    ).toBe("device:swipe");
  });

  it("tokenizes observations by source", () => {
    expect(tokenizeObservation({ source: "device", summary: "app active", metadata: undefined })).toBe("obs:device");
  });
});

describe("dataset builders", () => {
  it("interleaves observations and actions by timestamp", () => {
    const dataset = buildMovementDatasetFromTrajectories([
      span(
        "t1",
        [
          { kind: "action", tool: "device", summary: "type", ts: 20, metadata: { gesture: "type" } },
          { kind: "action", tool: "device", summary: "tap", ts: 10, metadata: { gesture: "tap" } },
        ],
        [{ kind: "observation", source: "device", summary: "active", ts: 5 }],
      ),
    ]);
    expect(dataset.sequences[0]?.tokens).toEqual(["obs:device", "device:tap", "device:type"]);
  });

  it("drops empty trajectories", () => {
    const dataset = buildMovementDatasetFromTrajectories([span("empty", [])]);
    expect(dataset.sequences).toHaveLength(0);
  });

  it("builds from replay manifests", () => {
    const replay: ReplayManifest = {
      version: 1,
      sessionId: "s1",
      trajectoryIds: ["t1"],
      eventCount: 2,
      events: [
        { kind: "observation", ts: 1, trajectoryId: "t1", source: "browser", summary: "page" },
        { kind: "action", ts: 2, trajectoryId: "t1", tool: "navigate", summary: "go" },
        { kind: "transcript", ts: 3, messageId: "m1", role: "user", content: "hi" },
      ],
    };
    const dataset = buildMovementDatasetFromReplays([replay]);
    expect(dataset.sequences[0]?.tokens).toEqual(["obs:browser", "act:navigate"]);
  });
});

describe("MarkovMovementModel", () => {
  const dataset: MovementDataset = {
    version: 1,
    sequences: [
      { id: "a", tokens: ["obs:device", "device:tap", "device:type", "device:shortcut"] },
      { id: "b", tokens: ["obs:device", "device:tap", "device:type", "device:shortcut"] },
    ],
  };

  it("reproduces recorded movements exactly from a seed", () => {
    const model = MarkovMovementModel.train(dataset, { order: 3 });
    const generated = model.generate(["obs:device"], 10);
    expect(generated).toEqual(["device:tap", "device:type", "device:shortcut"]);
  });

  it("returns undefined prediction for an empty model", () => {
    const model = MarkovMovementModel.train({ version: 1, sequences: [] }, { order: 2 });
    expect(model.predictNext(["anything"])).toBeUndefined();
  });

  it("generalizes to unseen prefixes via backoff", () => {
    // Full-order context "x device:tap" is never seen, but "device:tap" alone
    // always precedes "device:type" — backoff to a lower order recovers it.
    const model = MarkovMovementModel.train(dataset, { order: 3 });
    const prediction = model.predictNext(["novel-context", "device:tap"]);
    expect(prediction?.token).toBe("device:type");
    expect(prediction?.order).toBeLessThan(3);
  });

  it("ranks candidates deterministically", () => {
    const branching: MovementDataset = {
      version: 1,
      sequences: [
        { id: "1", tokens: ["start", "a"] },
        { id: "2", tokens: ["start", "a"] },
        { id: "3", tokens: ["start", "b"] },
      ],
    };
    const model = MarkovMovementModel.train(branching, { order: 1 });
    const ranked = model.rankNext(["start"]);
    expect(ranked.map((r) => r.token)).toEqual(["a", "b"]);
    expect(ranked[0]?.probability).toBeCloseTo(2 / 3);
  });

  it("round-trips through serialization", () => {
    const model = MarkovMovementModel.train(dataset, { order: 3 });
    const restored = MarkovMovementModel.fromSerialized(model.serialize());
    expect(restored.order).toBe(3);
    expect(restored.generate(["obs:device"], 10)).toEqual(model.generate(["obs:device"], 10));
    expect(restored.serialize()).toEqual(model.serialize());
  });
});

describe("MarkovMovementBackend", () => {
  it("trains a model through the pluggable backend interface", async () => {
    const backend = new MarkovMovementBackend();
    const model = await backend.train(
      { version: 1, sequences: [{ id: "a", tokens: ["x", "y", "z"] }] },
      { order: 2 },
    );
    expect(model.backend).toBe("markov");
    expect(model.generate(["x"], 5)).toEqual(["y", "z"]);
  });
});

describe("generalization eval harness", () => {
  it("scores exact replay of held-out sequences drawn from the same grammar", async () => {
    const training = generateSyntheticMovementSequences(12, 0);
    const heldOut = generateSyntheticMovementSequences(12, 1);
    const backend = new MarkovMovementBackend();
    const model = await backend.train({ version: 1, sequences: training }, { order: 4 });
    const report = evaluateMovementModel(model, heldOut);
    expect(report.sequences).toBeGreaterThan(0);
    // Held-out sequences share structure with training, so backoff should
    // predict most next movements and reproduce whole flows.
    expect(report.accuracy).toBeGreaterThan(0.5);
    expect(report.exactReplayRate).toBeGreaterThan(0);
  });

  it("reports zero accuracy for an untrained model", () => {
    const model = MarkovMovementModel.train({ version: 1, sequences: [] }, { order: 2 });
    const report = evaluateMovementModel(model, generateSyntheticMovementSequences(4, 0));
    expect(report.accuracy).toBe(0);
    expect(report.exactReplayRate).toBe(0);
  });
});

describe("synthetic generator", () => {
  it("is deterministic for a given seed", () => {
    expect(generateSyntheticMovementSequences(5, 2)).toEqual(generateSyntheticMovementSequences(5, 2));
  });

  it("produces non-empty token sequences", () => {
    const sequences = generateSyntheticMovementSequences(6, 0);
    expect(sequences).toHaveLength(6);
    for (const sequence of sequences) {
      expect(sequence.tokens.length).toBeGreaterThanOrEqual(3);
    }
  });
});
