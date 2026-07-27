import { describe, expect, it } from "vitest";
import { buildReplayManifest, type ReplayManifest } from "../capture/replay.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import {
  NgramSimilarityBackend,
  buildMovementDataset,
  loadMovementModel,
  measureReplayFidelity,
  tokenizeMovementText,
  trainMovementModel,
  type MovementDataset,
} from "./movement-model.js";

/** Build a replay manifest from a synthetic (observation, action) movement script. */
function syntheticReplay(
  sessionId: string,
  steps: Array<{ observe: string; tool: string; act: string }>,
): ReplayManifest {
  let ts = 0;
  const observations = steps.map((step) => ({
    kind: "observation" as const,
    source: "screen",
    summary: step.observe,
    ts: (ts += 10),
  }));
  const actions = steps.map((step) => ({
    kind: "action" as const,
    tool: step.tool,
    summary: step.act,
    ts: (ts += 10),
  }));
  const trajectory = buildTrajectorySpan({
    id: `${sessionId}-traj`,
    sessionId,
    observations,
    actions,
  });
  return buildReplayManifest({ sessionId, transcript: [], trajectories: [trajectory] });
}

describe("tokenizeMovementText", () => {
  it("lowercases, splits, drops stopwords and short tokens, and dedupes", () => {
    expect(tokenizeMovementText("Open the Deploy page, open Deploy!")).toEqual(["open", "deploy", "page"]);
  });
});

describe("buildMovementDataset", () => {
  it("emits one example per action with rolling context and prior-action tools", () => {
    const replay = syntheticReplay("s1", [
      { observe: "login screen", tool: "keyboard", act: "type username" },
      { observe: "password field", tool: "keyboard", act: "type password" },
      { observe: "submit button", tool: "mouse", act: "click submit" },
    ]);

    const dataset = buildMovementDataset([replay]);

    expect(dataset.examples).toHaveLength(3);
    expect(dataset.examples[0].priorActions).toEqual([]);
    expect(dataset.examples[1].priorActions).toEqual(["keyboard"]);
    expect(dataset.examples[2].priorActions).toEqual(["keyboard", "keyboard"]);
    // The third example's context should include earlier observations/actions.
    expect(dataset.examples[2].context).toContain("submit");
    expect(dataset.examples[2].action).toEqual({ tool: "mouse", summary: "click submit" });
  });
});

describe("NgramSimilarityBackend — repeat recorded movements", () => {
  it("reproduces a recorded action sequence exactly via rollout (fidelity 1.0)", () => {
    const replay = syntheticReplay("s1", [
      { observe: "editor open", tool: "keyboard", act: "type hello" },
      { observe: "text present", tool: "hotkey", act: "save file" },
      { observe: "saved", tool: "mouse", act: "close editor" },
    ]);
    const { model } = trainMovementModel([replay]);

    const rollout = model.rollout({ tokens: [] }, { maxSteps: 3 });
    expect(rollout.map((action) => action.tool)).toEqual(["keyboard", "hotkey", "mouse"]);
    expect(rollout.map((action) => action.summary)).toEqual(["type hello", "save file", "close editor"]);
    expect(measureReplayFidelity(model, replay)).toBe(1);
  });

  it("predicts the next action deterministically from an exact prior-action n-gram", () => {
    const replay = syntheticReplay("s1", [
      { observe: "step one", tool: "a", act: "do a" },
      { observe: "step two", tool: "b", act: "do b" },
      { observe: "step three", tool: "c", act: "do c" },
    ]);
    const { model } = trainMovementModel([replay]);

    const prediction = model.predict({ tokens: [], priorActions: ["a", "b"] });
    expect(prediction?.action.tool).toBe("c");
    expect(prediction?.source).toBe("exact-ngram");
    expect(prediction?.confidence).toBe(1);
  });
});

describe("NgramSimilarityBackend — generalize to related movements", () => {
  it("routes an unseen-but-related context to the right action family via similarity", () => {
    // Two distinct movement families with disjoint vocabulary.
    const deployReplay = syntheticReplay("deploy", [
      { observe: "deploy dashboard production release", tool: "deploy-bot", act: "trigger deploy" },
    ]);
    const searchReplay = syntheticReplay("search", [
      { observe: "search logs error stacktrace query", tool: "search-bot", act: "run search" },
    ]);
    const { model } = trainMovementModel([deployReplay, searchReplay]);

    // Neither exact phrase was seen, but the vocabulary is related to one family.
    const deployLike = model.predict({ tokens: ["release", "the", "production", "build"] });
    expect(deployLike?.action.tool).toBe("deploy-bot");
    expect(deployLike?.source).toBe("similarity");
    expect(deployLike?.confidence).toBeGreaterThan(0);

    const searchLike = model.predict({ tokens: ["query", "the", "error", "logs"] });
    expect(searchLike?.action.tool).toBe("search-bot");
    expect(searchLike?.source).toBe("similarity");
  });

  it("falls back to the globally most frequent action when nothing matches", () => {
    const dataset: MovementDataset = {
      version: 1,
      examples: [
        { context: ["alpha"], priorActions: [], action: { tool: "common", summary: "x" } },
        { context: ["beta"], priorActions: [], action: { tool: "common", summary: "x" } },
        { context: ["gamma"], priorActions: [], action: { tool: "rare", summary: "y" } },
      ],
    };
    const model = new NgramSimilarityBackend().train(dataset);

    const prediction = model.predict({ tokens: ["totally", "unrelated", "vocabulary"] });
    expect(prediction?.action.tool).toBe("common");
    expect(prediction?.source).toBe("fallback");
    expect(prediction?.confidence).toBe(0);
  });

  it("returns undefined for an empty model", () => {
    const model = new NgramSimilarityBackend().train({ version: 1, examples: [] });
    expect(model.predict({ tokens: ["anything"] })).toBeUndefined();
  });
});

describe("movement model serialization", () => {
  it("round-trips through toJSON/loadMovementModel with identical predictions", () => {
    const replay = syntheticReplay("s1", [
      { observe: "one", tool: "a", act: "do a" },
      { observe: "two", tool: "b", act: "do b" },
    ]);
    const { model } = trainMovementModel([replay]);

    const serialized = JSON.parse(JSON.stringify(model.toJSON()));
    const restored = loadMovementModel(serialized);

    const context = { tokens: ["one"], priorActions: ["a"] };
    expect(restored.predict(context)).toEqual(model.predict(context));
    expect(restored.rollout({ tokens: [] })).toEqual(model.rollout({ tokens: [] }));
    expect(restored.backendId).toBe(model.backendId);
  });
});

describe("measureReplayFidelity — generalization eval", () => {
  it("scores held-out trajectories below 1 when the model cannot reproduce them", () => {
    const trained = syntheticReplay("train", [
      { observe: "alpha", tool: "a", act: "do a" },
      { observe: "beta", tool: "b", act: "do b" },
    ]);
    const heldOut = syntheticReplay("holdout", [
      { observe: "alpha", tool: "a", act: "do a" },
      { observe: "gamma", tool: "z", act: "do z" },
    ]);
    const { model } = trainMovementModel([trained]);

    // First step matches (a), second diverges (z never seen at that position).
    expect(measureReplayFidelity(model, heldOut)).toBeCloseTo(0.5, 5);
    // A manifest with no actions is trivially fidelity 1.
    expect(measureReplayFidelity(model, buildReplayManifest({ sessionId: "empty", transcript: [], trajectories: [] }))).toBe(1);
  });
});
