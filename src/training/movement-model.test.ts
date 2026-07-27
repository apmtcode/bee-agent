import { describe, expect, it } from "vitest";
import type { ReplayManifest } from "../capture/replay.js";
import {
  MarkovMovementBackend,
  MOVEMENT_BOS,
  MOVEMENT_EOS,
  buildMovementDataset,
  loadMarkovMovementModel,
  tokenizeReplayEvent,
} from "./movement-model.js";

function actionReplay(trajectoryId: string, summaries: string[]): ReplayManifest {
  const events = summaries.map((summary, index) => ({
    kind: "action" as const,
    ts: 1000 + index,
    trajectoryId,
    tool: "device",
    summary,
  }));
  return {
    version: 1,
    sessionId: `session-${trajectoryId}`,
    trajectoryIds: [trajectoryId],
    eventCount: events.length,
    events,
  };
}

describe("buildMovementDataset", () => {
  it("tokenizes actions per trajectory with boundaries and a sorted vocabulary", () => {
    const dataset = buildMovementDataset([
      actionReplay("t1", ["tapped Menu", "tapped Settings"]),
      actionReplay("t2", ["tapped Menu", "swiped down"]),
    ]);

    expect(dataset.sequences).toHaveLength(2);
    expect(dataset.sequences[0].tokens).toEqual([
      MOVEMENT_BOS,
      "action:device:tapped Menu",
      "action:device:tapped Settings",
      MOVEMENT_EOS,
    ]);
    expect(dataset.tokenization).toEqual({ includeObservations: false, withBoundaries: true });
    // vocabulary is sorted and de-duplicated (shared "tapped Menu" appears once)
    expect(dataset.vocabulary).toEqual([...dataset.vocabulary].sort());
    expect(dataset.vocabulary.filter((token) => token === "action:device:tapped Menu")).toHaveLength(1);
  });

  it("excludes observations by default but includes them when requested", () => {
    const replay: ReplayManifest = {
      version: 1,
      sessionId: "s",
      trajectoryIds: ["t"],
      eventCount: 2,
      events: [
        { kind: "observation", ts: 1, trajectoryId: "t", source: "device", summary: "app active" },
        { kind: "action", ts: 2, trajectoryId: "t", tool: "device", summary: "tapped Go" },
      ],
    };

    const actionsOnly = buildMovementDataset([replay]);
    expect(actionsOnly.sequences[0].tokens).toEqual([MOVEMENT_BOS, "action:device:tapped Go", MOVEMENT_EOS]);

    const withObs = buildMovementDataset([replay], { includeObservations: true });
    expect(withObs.sequences[0].tokens).toContain("observation:device:app active");
  });

  it("normalizes whitespace in summaries and drops transcript events", () => {
    expect(
      tokenizeReplayEvent({ kind: "action", ts: 1, trajectoryId: "t", tool: "device", summary: "  tapped   Save  " }),
    ).toBe("action:device:tapped Save");
    expect(
      tokenizeReplayEvent({ kind: "transcript", ts: 1, messageId: "m", role: "user", content: "hi" }),
    ).toBeUndefined();
  });
});

describe("MarkovMovementBackend", () => {
  it("repeats a memorized movement sequence exactly", async () => {
    const dataset = buildMovementDataset([actionReplay("t1", ["tapped Menu", "tapped Settings", "swiped down"])]);
    const model = await new MarkovMovementBackend().train(dataset, { order: 2 });

    const rollout = model.generate([MOVEMENT_BOS], 10);
    expect(rollout).toEqual([
      "action:device:tapped Menu",
      "action:device:tapped Settings",
      "action:device:swiped down",
    ]);
  });

  it("predicts the highest-count continuation with deterministic tie-breaking", async () => {
    // "tapped Menu" is followed twice by "tapped Settings", once by "swiped down".
    const dataset = buildMovementDataset([
      actionReplay("a", ["tapped Menu", "tapped Settings"]),
      actionReplay("b", ["tapped Menu", "tapped Settings"]),
      actionReplay("c", ["tapped Menu", "swiped down"]),
    ]);
    const model = await new MarkovMovementBackend().train(dataset, { order: 1 });

    const prediction = model.predictNext(["action:device:tapped Menu"]);
    expect(prediction?.token).toBe("action:device:tapped Settings");
    expect(prediction?.order).toBe(1);
    expect(prediction?.fallback).toBe(false);
    expect(prediction?.probability).toBeCloseTo(2 / 3, 5);
  });

  it("generalizes to an unseen context by backing off to a shorter suffix", async () => {
    // Train: A -> B -> C  and  X -> B -> C. The context [X, B] never appears at
    // full order 2 in the exact form [<bos>, X-analog]; assert back-off recovers C.
    const dataset = buildMovementDataset([
      actionReplay("a", ["A", "B", "C"]),
      actionReplay("b", ["X", "B", "C"]),
    ]);
    const model = await new MarkovMovementBackend().train(dataset, { order: 2 });

    // Novel context: [Q, B] — pair (Q,B) was never trained, but (B -> C) was.
    const prediction = model.predictNext(["action:device:Q", "action:device:B"]);
    expect(prediction?.token).toBe("action:device:C");
    expect(prediction?.fallback).toBe(true);
    expect(prediction?.order).toBe(1);
  });

  it("returns undefined when no context (even the unigram prior) has been seen", async () => {
    const dataset = buildMovementDataset([actionReplay("a", ["A"])], { withBoundaries: false });
    const model = await new MarkovMovementBackend().train(dataset, { order: 2 });
    // Empty model context has the unigram prior, so a fresh empty context predicts something...
    expect(model.predictNext([])).toBeDefined();
    // ...but a model trained on nothing predicts nothing.
    const empty = await new MarkovMovementBackend().train(
      { version: 1, tokenization: { includeObservations: false, withBoundaries: false }, sequences: [], vocabulary: [] },
      { order: 2 },
    );
    expect(empty.predictNext(["anything"])).toBeUndefined();
  });

  it("round-trips through serialize / load deterministically", async () => {
    const dataset = buildMovementDataset([actionReplay("t1", ["tapped Menu", "tapped Settings"])]);
    const model = await new MarkovMovementBackend().train(dataset, { order: 2 });
    const serialized = model.serialize();

    // Serialization is stable (sorted keys) — same object twice.
    expect(model.serialize()).toEqual(serialized);
    expect(serialized.backendId).toBe("markov-backoff");
    expect(serialized.order).toBe(2);

    const reloaded = loadMarkovMovementModel(serialized);
    expect(reloaded.generate([MOVEMENT_BOS], 10)).toEqual(model.generate([MOVEMENT_BOS], 10));
  });
});
