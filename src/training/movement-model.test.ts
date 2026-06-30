import { describe, expect, it } from "vitest";
import {
  MarkovMovementBackend,
  canonicalMovementToken,
  evaluateMovementGeneralization,
  movementDatasetFromReplays,
  movementDatasetFromTrajectories,
  splitMovementDataset,
  trainMovementModel,
  type MovementDataset,
} from "./movement-model.js";

describe("canonicalMovementToken", () => {
  it("prefers structured device-gesture metadata", () => {
    expect(
      canonicalMovementToken({
        tool: "device",
        summary: "swiped up",
        metadata: { gesture: "swipe", direction: "up", target: "Photos List" },
      }),
    ).toBe("swipe:up>photos-list");
  });

  it("falls back to tool + slugged summary when no gesture metadata", () => {
    expect(canonicalMovementToken({ tool: "Browser", summary: "Open Tab" })).toBe("browser:open-tab");
  });

  it("is stable regardless of summary punctuation/casing", () => {
    const a = canonicalMovementToken({ tool: "device", summary: "Tapped: Submit!" });
    const b = canonicalMovementToken({ tool: "device", summary: "tapped submit" });
    expect(a).toBe(b);
  });
});

describe("dataset builders", () => {
  it("orders trajectory actions by timestamp", () => {
    const dataset = movementDatasetFromTrajectories([
      {
        id: "traj-1",
        actions: [
          { ts: 30, tool: "device", summary: "c" },
          { ts: 10, tool: "device", summary: "a" },
          { ts: 20, tool: "device", summary: "b" },
        ],
      },
    ]);
    expect(dataset.samples[0]).toEqual({
      id: "traj-1",
      tokens: ["device:a", "device:b", "device:c"],
    });
  });

  it("extracts only action events from replays", () => {
    const dataset = movementDatasetFromReplays([
      {
        trajectoryIds: ["t1"],
        events: [
          { kind: "observation", ts: 1, summary: "screen" },
          { kind: "action", ts: 2, tool: "device", summary: "tap" },
          { kind: "transcript", ts: 3, summary: "hi" },
          { kind: "action", ts: 4, tool: "device", summary: "type" },
        ],
      },
    ]);
    expect(dataset.samples).toEqual([{ id: "t1", tokens: ["device:tap", "device:type"] }]);
  });

  it("drops empty samples", () => {
    const dataset = movementDatasetFromTrajectories([{ id: "empty", actions: [] }]);
    expect(dataset.samples).toHaveLength(0);
  });
});

const SEQUENCE: MovementDataset = {
  samples: [
    { tokens: ["open", "search", "type", "submit", "close"] },
    { tokens: ["open", "search", "type", "submit", "close"] },
  ],
};

describe("MarkovMovementBackend — repeat recorded movements (objective c)", () => {
  it("predicts the exact next movement for a seen context", () => {
    const model = trainMovementModel(SEQUENCE, { order: 3 });
    const prediction = model.predictNext(["open", "search", "type"]);
    expect(prediction?.token).toBe("submit");
    expect(prediction?.probability).toBe(1);
    expect(prediction?.contextOrderUsed).toBe(3);
  });

  it("regenerates the full recorded trajectory from its seed", () => {
    const model = trainMovementModel(SEQUENCE, { order: 3 });
    expect(model.generate(["open"], 4)).toEqual(["search", "type", "submit", "close"]);
  });

  it("is deterministic on ties (lexicographic tie-break)", () => {
    const dataset: MovementDataset = {
      samples: [
        { tokens: ["a", "z"] },
        { tokens: ["a", "b"] },
      ],
    };
    const model = trainMovementModel(dataset, { order: 1 });
    const first = model.predictNext(["a"]);
    const second = trainMovementModel(dataset, { order: 1 }).predictNext(["a"]);
    expect(first?.token).toBe("b");
    expect(first?.token).toBe(second?.token);
    expect(first?.candidates).toHaveLength(2);
  });

  it("returns undefined for an empty model", () => {
    const model = trainMovementModel({ samples: [] });
    expect(model.predictNext(["anything"])).toBeUndefined();
  });
});

describe("MarkovMovementBackend — generalize to related movements (objective d)", () => {
  it("backs off to a shorter known context for an unseen prefix", () => {
    // 'search' is always followed by 'type', regardless of what came before.
    const dataset: MovementDataset = {
      samples: [
        { tokens: ["open", "search", "type"] },
        { tokens: ["menu", "search", "type"] },
      ],
    };
    const model = trainMovementModel(dataset, { order: 3 });
    // Novel full context 'login > search' was never seen, but 'search' was.
    const prediction = model.predictNext(["login", "search"]);
    expect(prediction?.token).toBe("type");
    expect(prediction?.contextOrderUsed).toBeLessThan(2);
  });

  it("falls all the way back to the global prior when nothing matches", () => {
    const model = trainMovementModel(SEQUENCE, { order: 3 });
    const prediction = model.predictNext(["totally-unseen-token"]);
    expect(prediction).toBeDefined();
    expect(prediction?.contextOrderUsed).toBe(0);
  });

  it("scoreNext discounts probabilities obtained via back-off", () => {
    const model = trainMovementModel(SEQUENCE, { order: 3 });
    const exact = model.scoreNext(["open", "search", "type"], "submit");
    const backedOff = model.scoreNext(["unseen", "unseen", "search"], "type");
    expect(exact).toBe(1);
    expect(backedOff).toBeGreaterThan(0);
    expect(backedOff).toBeLessThan(1);
  });
});

describe("evaluateMovementGeneralization", () => {
  it("scores perfect accuracy when holdout repeats a learned pattern", () => {
    const dataset: MovementDataset = {
      samples: [
        { tokens: ["open", "search", "type", "submit"] },
        { tokens: ["open", "search", "type", "submit"] },
        { tokens: ["open", "search", "type", "submit"] },
      ],
    };
    const { train, holdout } = splitMovementDataset(dataset, 3);
    expect(holdout.samples).toHaveLength(1);
    const result = evaluateMovementGeneralization({
      backend: new MarkovMovementBackend({ order: 3 }),
      train,
      holdout,
    });
    expect(result.total).toBe(3);
    expect(result.accuracy).toBe(1);
    expect(result.meanTrueProbability).toBeGreaterThan(0);
  });

  it("reports zero accuracy for an empty holdout", () => {
    const result = evaluateMovementGeneralization({
      backend: new MarkovMovementBackend(),
      train: SEQUENCE,
      holdout: { samples: [] },
    });
    expect(result.total).toBe(0);
    expect(result.accuracy).toBe(0);
    expect(result.meanTrueProbability).toBe(0);
  });

  it("captures generalization via back-off on related-but-unseen holdout", () => {
    const train: MovementDataset = {
      samples: [
        { tokens: ["home", "search", "type", "submit"] },
        { tokens: ["menu", "search", "type", "submit"] },
      ],
    };
    // Unseen prefix 'login', but the search->type->submit suffix generalizes.
    const holdout: MovementDataset = {
      samples: [{ tokens: ["login", "search", "type", "submit"] }],
    };
    const result = evaluateMovementGeneralization({
      backend: new MarkovMovementBackend({ order: 3 }),
      train,
      holdout,
      minContext: 1,
    });
    expect(result.correct).toBeGreaterThan(0);
    expect(result.backoffCorrect).toBeGreaterThan(0);
  });
});

describe("snapshot", () => {
  it("serializes transitions and vocabulary for inspection/persistence", () => {
    const model = trainMovementModel({ samples: [{ tokens: ["a", "b"] }] }, { order: 2 });
    const snapshot = model.toJSON();
    expect(snapshot.backend).toBe("markov");
    expect(snapshot.order).toBe(2);
    expect(snapshot.vocabulary).toEqual(["a", "b"]);
    expect(snapshot.transitions[""]).toMatchObject({ a: 1, b: 1 });
  });
});
