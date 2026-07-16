import { describe, expect, it } from "vitest";
import type { ReplayManifest } from "../capture/replay.js";
import {
  NGramMovementBackend,
  buildMovementDataset,
  evaluateMovementModel,
  generateSyntheticMovementSequences,
  loadMovementModel,
  movementTokenId,
  parseMovementToken,
  trainMovementModel,
  type MovementDataset,
  type MovementSequence,
  type MovementToken,
} from "./movement-model.js";

const TAP_SEARCH: MovementToken = { tool: "device", action: "tap", target: "search-field" };
const TYPE_SEARCH: MovementToken = { tool: "device", action: "type", target: "search-field" };
const TAP_RESULT: MovementToken = { tool: "device", action: "tap", target: "result-row" };
const SCROLL_DOWN: MovementToken = { tool: "device", action: "scroll", direction: "down" };

function seq(id: string, tokens: MovementToken[]): MovementSequence {
  return { id, tokens };
}

describe("movementTokenId", () => {
  it("is collision-free across fields", () => {
    expect(movementTokenId({ tool: "device", action: "tap", target: "a" })).not.toBe(
      movementTokenId({ tool: "device", action: "tap", target: "b" }),
    );
    // A target value must not be able to impersonate a tool+action pair.
    expect(movementTokenId({ tool: "device", action: "tap" })).not.toBe(
      movementTokenId({ tool: "device", action: "taptarget" }),
    );
  });
});

describe("parseMovementToken", () => {
  it("inverts the capture adapter's gesture phrasing", () => {
    expect(parseMovementToken("device", "tapped submit button")).toEqual({
      tool: "device",
      action: "tap",
      target: "submit-button",
    });
    expect(parseMovementToken("device", "swiped left")).toEqual({
      tool: "device",
      action: "swipe",
      direction: "left",
    });
    expect(parseMovementToken("device", "typed into search field")).toEqual({
      tool: "device",
      action: "type",
      target: "search-field",
    });
  });

  it("falls back to the first word for unknown phrasings", () => {
    expect(parseMovementToken("browser", "hovered menu")).toEqual({
      tool: "browser",
      action: "hovered",
      target: "menu",
    });
  });
});

describe("NGramMovementBackend training + inference", () => {
  const dataset: MovementDataset = {
    sequences: [
      seq("a", [TAP_SEARCH, TYPE_SEARCH, TAP_RESULT, SCROLL_DOWN]),
      seq("b", [TAP_SEARCH, TYPE_SEARCH, TAP_RESULT, SCROLL_DOWN]),
      seq("c", [TAP_SEARCH, TYPE_SEARCH, TAP_RESULT]),
    ],
  };

  it("repeats a recorded movement grammar (objective #2c)", () => {
    const model = trainMovementModel(dataset);
    const afterTapType = model.predictNext([TAP_SEARCH, TYPE_SEARCH]);
    expect(afterTapType.token).toEqual(TAP_RESULT);
    expect(afterTapType.source).toBe("ngram");
    expect(afterTapType.confidence).toBeGreaterThan(0.9);
  });

  it("backs off to shorter contexts when the full context is unseen", () => {
    const model = trainMovementModel(dataset, { order: 3 });
    // [SCROLL_DOWN, TAP_SEARCH] never occurs, but the bigram from TAP_SEARCH does.
    const prediction = model.predictNext([SCROLL_DOWN, TAP_SEARCH]);
    expect(prediction.token).toEqual(TYPE_SEARCH);
    expect(prediction.source).toBe("backoff");
  });

  it("returns the prior when there is no usable context", () => {
    const model = trainMovementModel(dataset);
    const prediction = model.predictNext([]);
    expect(prediction.source).toBe("prior");
    expect(prediction.token).toBeDefined();
  });

  it("returns a `none` prediction for an empty dataset", () => {
    const model = trainMovementModel({ sequences: [] });
    const prediction = model.predictNext([TAP_SEARCH]);
    expect(prediction.source).toBe("none");
    expect(prediction.token).toBeUndefined();
    expect(model.vocabularySize).toBe(0);
  });

  it("is deterministic across repeated training", () => {
    const a = trainMovementModel(dataset).predictNext([TAP_SEARCH, TYPE_SEARCH]);
    const b = trainMovementModel(dataset).predictNext([TAP_SEARCH, TYPE_SEARCH]);
    expect(a).toEqual(b);
  });
});

describe("generalization to new-but-related movements (objective #2d)", () => {
  it("borrows transitions from a feature-similar seen movement", () => {
    // The model has only ever seen taps on `search-field`. A tap on a brand-new
    // `search-box` target shares tool+action features, so the model should
    // generalize to the learnt "tap → type" continuation.
    const model = trainMovementModel({
      sequences: [seq("a", [TAP_SEARCH, TYPE_SEARCH]), seq("b", [TAP_SEARCH, TYPE_SEARCH])],
    });
    const unseen: MovementToken = { tool: "device", action: "tap", target: "search-box" };
    const prediction = model.predictNext([unseen]);
    expect(prediction.source).toBe("generalized");
    expect(prediction.token).toEqual(TYPE_SEARCH);
    expect(prediction.confidence).toBeGreaterThan(0);
    expect(prediction.confidence).toBeLessThanOrEqual(1);
  });
});

describe("generate", () => {
  it("autoregressively continues a seed", () => {
    const model = trainMovementModel({
      sequences: [
        seq("a", [TAP_SEARCH, TYPE_SEARCH, TAP_RESULT, SCROLL_DOWN]),
        seq("b", [TAP_SEARCH, TYPE_SEARCH, TAP_RESULT, SCROLL_DOWN]),
      ],
    });
    const continued = model.generate([TAP_SEARCH], 3);
    expect(continued).toEqual([TYPE_SEARCH, TAP_RESULT, SCROLL_DOWN]);
  });

  it("stops early when nothing can be predicted", () => {
    const model = trainMovementModel({ sequences: [] });
    expect(model.generate([TAP_SEARCH], 5)).toEqual([]);
  });
});

describe("serialize / loadMovementModel round-trip", () => {
  it("reproduces predictions from a snapshot", () => {
    const dataset: MovementDataset = {
      sequences: [seq("a", [TAP_SEARCH, TYPE_SEARCH, TAP_RESULT])],
    };
    const model = trainMovementModel(dataset);
    const snapshot = model.serialize();
    expect(snapshot.backend).toBe("ngram");
    const restored = loadMovementModel(snapshot);
    expect(restored.vocabularySize).toBe(model.vocabularySize);
    expect(restored.predictNext([TAP_SEARCH, TYPE_SEARCH])).toEqual(
      model.predictNext([TAP_SEARCH, TYPE_SEARCH]),
    );
  });
});

describe("buildMovementDataset from replay manifests", () => {
  it("extracts action events into ordered movement sequences", () => {
    const replay: ReplayManifest = {
      version: 1,
      sessionId: "session-1",
      trajectoryIds: ["t1"],
      eventCount: 3,
      events: [
        { kind: "observation", ts: 1, trajectoryId: "t1", source: "device", summary: "app active" },
        { kind: "action", ts: 2, trajectoryId: "t1", tool: "device", summary: "tapped search field" },
        { kind: "action", ts: 3, trajectoryId: "t1", tool: "device", summary: "typed into search field" },
      ],
    };
    const dataset = buildMovementDataset([replay]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0].tokens).toEqual([
      { tool: "device", action: "tap", target: "search-field" },
      { tool: "device", action: "type", target: "search-field" },
    ]);
  });

  it("drops replays with no action events", () => {
    const replay: ReplayManifest = {
      version: 1,
      sessionId: "empty",
      trajectoryIds: [],
      eventCount: 1,
      events: [{ kind: "observation", ts: 1, trajectoryId: "t", source: "device", summary: "idle" }],
    };
    expect(buildMovementDataset([replay]).sequences).toHaveLength(0);
  });
});

describe("synthetic stream + evaluation harness", () => {
  it("is byte-stable for a fixed seed", () => {
    const a = generateSyntheticMovementSequences({ count: 3, lengthEach: 6, seed: 42 });
    const b = generateSyntheticMovementSequences({ count: 3, lengthEach: 6, seed: 42 });
    expect(a).toEqual(b);
    expect(a).toHaveLength(3);
    expect(a[0].tokens).toHaveLength(6);
  });

  it("learns held-out synthetic structure better than chance", () => {
    const train = generateSyntheticMovementSequences({ count: 40, lengthEach: 12, seed: 7 });
    const heldOut = generateSyntheticMovementSequences({ count: 10, lengthEach: 12, seed: 999 });
    const model = new NGramMovementBackend().train({ sequences: train }, { order: 3 });
    const result = evaluateMovementModel(model, heldOut);
    expect(result.total).toBeGreaterThan(0);
    // The grammar advances to the neighbouring movement 80% of the time, so a
    // learned model should clear a generous chance baseline.
    expect(result.accuracy).toBeGreaterThan(0.5);
  });

  it("reports zero accuracy on empty held-out data", () => {
    const model = trainMovementModel({ sequences: [] });
    expect(evaluateMovementModel(model, [])).toEqual({
      total: 0,
      correct: 0,
      accuracy: 0,
      generalizedCorrect: 0,
    });
  });
});
