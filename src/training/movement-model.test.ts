import { describe, expect, it } from "vitest";
import {
  MOVEMENT_END_TOKEN,
  MarkovMovementBackend,
  buildMovementDataset,
  deserializeMovementModel,
  generateSyntheticMovementDataset,
  movementTokenForAction,
  slugifyMovement,
  type MovementDataset,
} from "./movement-model.js";
import type { ReplayManifest } from "../capture/replay.js";

function replayFromActions(trajectoryId: string, actions: Array<{ tool: string; summary: string }>): ReplayManifest {
  return {
    version: 1,
    sessionId: `session-${trajectoryId}`,
    trajectoryIds: [trajectoryId],
    eventCount: actions.length,
    events: actions.map((action, index) => ({
      kind: "action" as const,
      ts: index,
      trajectoryId,
      tool: action.tool,
      summary: action.summary,
    })),
  };
}

describe("slugifyMovement / movementTokenForAction", () => {
  it("collapses semantically-equal summaries to a stable token", () => {
    expect(slugifyMovement("Tapped Submit Button!")).toBe("tapped-submit-button");
    expect(movementTokenForAction("device", "Tapped Submit Button")).toBe("device:tapped-submit-button");
    // Same movement described with different casing/punctuation -> same token.
    expect(movementTokenForAction("device", "tapped submit button")).toBe(
      movementTokenForAction("device", "Tapped Submit Button"),
    );
  });

  it("falls back to a non-empty slug", () => {
    expect(slugifyMovement("   ")).toBe("movement");
  });
});

describe("buildMovementDataset", () => {
  it("extracts ordered action tokens from replay manifests and drops empties", () => {
    const replays: ReplayManifest[] = [
      replayFromActions("t1", [
        { tool: "device", summary: "focus search" },
        { tool: "device", summary: "type query" },
      ]),
      // A manifest with mixed event kinds — only actions become tokens.
      {
        version: 1,
        sessionId: "s2",
        trajectoryIds: ["t2"],
        eventCount: 2,
        events: [
          { kind: "observation", ts: 0, trajectoryId: "t2", source: "device", summary: "screen" },
          { kind: "action", ts: 1, trajectoryId: "t2", tool: "device", summary: "focus search" },
        ],
      },
      // Below minTokens -> dropped.
      replayFromActions("t3", []),
    ];

    const dataset = buildMovementDataset(replays, { minTokens: 1 });
    expect(dataset.examples.map((example) => example.trajectoryId)).toEqual(["t1", "t2"]);
    expect(dataset.examples[0]!.tokens).toEqual(["device:focus-search", "device:type-query"]);
    expect(dataset.vocabulary).toContain("device:focus-search");
    // Vocabulary is sorted and de-duplicated across examples.
    expect(dataset.vocabulary).toEqual([...dataset.vocabulary].sort());
  });
});

describe("MarkovMovementBackend — repeat recorded movements", () => {
  it("reproduces a recorded movement sequence exactly from its start", async () => {
    const dataset: MovementDataset = {
      version: 1,
      examples: [{ trajectoryId: "t1", tokens: ["a", "b", "c", "d"] }],
      vocabulary: ["a", "b", "c", "d"],
    };
    const model = await new MarkovMovementBackend().train(dataset, { order: 2 });

    // From an empty prompt the model should regenerate the whole recorded run.
    expect(model.generate([])).toEqual(["a", "b", "c", "d"]);
    // And continue correctly from a mid-sequence prompt.
    expect(model.generate(["a", "b"])).toEqual(["c", "d"]);
  });

  it("predicts END after the last recorded token", async () => {
    const dataset: MovementDataset = {
      version: 1,
      examples: [{ trajectoryId: "t1", tokens: ["a", "b"] }],
      vocabulary: ["a", "b"],
    };
    const model = await new MarkovMovementBackend().train(dataset, { order: 2 });
    const prediction = model.predictNext(["a", "b"]);
    expect(prediction?.token).toBe(MOVEMENT_END_TOKEN);
  });

  it("resolves ambiguous continuations by frequency, deterministically", async () => {
    const dataset: MovementDataset = {
      version: 1,
      examples: [
        { trajectoryId: "t1", tokens: ["open", "save"] },
        { trajectoryId: "t2", tokens: ["open", "save"] },
        { trajectoryId: "t3", tokens: ["open", "close"] },
      ],
      vocabulary: ["open", "save", "close"],
    };
    const model = await new MarkovMovementBackend().train(dataset, { order: 1 });
    const prediction = model.predictNext(["open"]);
    expect(prediction?.token).toBe("save"); // 2/3 vs 1/3
    expect(prediction?.probability).toBeCloseTo(2 / 3, 6);
    expect(prediction?.alternatives.map((alt) => alt.token)).toEqual(["save", "close"]);
  });
});

describe("MarkovMovementBackend — generalize to new but related movements", () => {
  it("backs off to a shorter context for an unseen prefix", async () => {
    // Two workflows share the transition "type-query" -> "submit".
    const dataset: MovementDataset = {
      version: 1,
      examples: [
        { trajectoryId: "t1", tokens: ["open-mail", "type-query", "submit"] },
        { trajectoryId: "t2", tokens: ["open-browser", "type-query", "submit"] },
      ],
      vocabulary: ["open-mail", "open-browser", "type-query", "submit"],
    };
    const model = await new MarkovMovementBackend().train(dataset, { order: 2 });

    // Novel prefix the model never saw at full order: (open-docs, type-query).
    // Full-order context is unknown, so it must back off to the order-1
    // context "type-query" -> "submit" that both workflows share.
    const prediction = model.predictNext(["open-docs", "type-query"]);
    expect(prediction?.token).toBe("submit");
    expect(prediction?.order).toBe(1); // proves backoff happened
  });

  it("completes a novel-but-related trajectory end to end", async () => {
    const dataset = generateSyntheticMovementDataset({
      workflows: [
        { id: "search", tokens: ["focus-search", "type-query", "press-enter", "read-result"] },
        { id: "compose", tokens: ["open-compose", "type-query", "press-enter", "send"] },
      ],
      repetitions: 5,
      seed: 42,
    });
    const model = await new MarkovMovementBackend().train(dataset, { order: 2 });

    // Start from a shared middle sub-sequence; generation should land on a
    // learned continuation rather than stalling.
    const continuation = model.generate(["type-query", "press-enter"]);
    expect(continuation.length).toBeGreaterThan(0);
    expect(["read-result", "send"]).toContain(continuation[0]);
  });
});

describe("serialization", () => {
  it("round-trips a trained model without behavior change", async () => {
    const dataset = generateSyntheticMovementDataset({
      workflows: [{ id: "w", tokens: ["a", "b", "c"] }],
      repetitions: 3,
      seed: 7,
    });
    const backend = new MarkovMovementBackend();
    const model = await backend.train(dataset, { order: 2 });
    const restored = deserializeMovementModel(model.serialize());

    expect(restored.order).toBe(model.order);
    expect(restored.backendId).toBe(model.backendId);
    expect(restored.generate([])).toEqual(model.generate([]));
    expect(restored.predictNext(["a"])?.token).toBe(model.predictNext(["a"])?.token);
    // Serialization is stable/sorted -> deterministic on-disk form.
    expect(model.serialize()).toEqual(restored.serialize());
  });
});

describe("generateSyntheticMovementDataset", () => {
  it("is deterministic for a fixed seed", () => {
    const params = {
      workflows: [{ id: "w", tokens: ["a", "b", "c"] }],
      repetitions: 4,
      seed: 123,
      dropTailProbability: 0.5,
    };
    expect(generateSyntheticMovementDataset(params)).toEqual(generateSyntheticMovementDataset(params));
  });

  it("produces repetitions x workflows examples", () => {
    const dataset = generateSyntheticMovementDataset({
      workflows: [
        { id: "a", tokens: ["x"] },
        { id: "b", tokens: ["y"] },
      ],
      repetitions: 3,
    });
    expect(dataset.examples).toHaveLength(6);
  });
});
