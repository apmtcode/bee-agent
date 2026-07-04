import { describe, expect, it } from "vitest";
import type { ReplayManifest } from "../capture/replay.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import {
  MOVEMENT_END_TOKEN,
  NGramMovementBackend,
  buildMovementDataset,
  evaluateNextTokenAccuracy,
  tokenizeReplays,
  tokenizeTrajectories,
  type MovementSequence,
} from "./movement-model.js";

function replay(sessionId: string, events: ReplayManifest["events"]): ReplayManifest {
  return {
    version: 1,
    sessionId,
    trajectoryIds: [],
    eventCount: events.length,
    events,
  };
}

function action(ts: number, tool: string): ReplayManifest["events"][number] {
  return { kind: "action", ts, trajectoryId: "t", tool, summary: tool };
}

function observation(ts: number, source: string): ReplayManifest["events"][number] {
  return { kind: "observation", ts, trajectoryId: "t", source, summary: source };
}

describe("movement tokenization", () => {
  it("tokenizes replay actions and observations, excluding transcript by default", () => {
    const sequences = tokenizeReplays([
      replay("s1", [
        { kind: "transcript", ts: 0, messageId: "m", role: "user", content: "hi" },
        observation(1, "screen"),
        action(2, "mouse.move"),
        action(3, "mouse.click"),
      ]),
    ]);
    expect(sequences).toEqual([
      { id: "s1", tokens: ["obs:screen", "action:mouse.move", "action:mouse.click"] },
    ]);
  });

  it("can opt transcript tokens in and actions out", () => {
    const sequences = tokenizeReplays(
      [replay("s1", [{ kind: "transcript", ts: 0, messageId: "m", role: "assistant", content: "x" }, action(1, "key.press")])],
      { includeTranscript: true, includeActions: false },
    );
    expect(sequences).toEqual([{ id: "s1", tokens: ["msg:assistant"] }]);
  });

  it("orders trajectory tokens by timestamp", () => {
    const span = buildTrajectorySpan({
      id: "traj-1",
      sessionId: "s1",
      observations: [{ kind: "observation", source: "screen", summary: "s", ts: 5 }],
      actions: [
        { kind: "action", tool: "mouse.click", summary: "c", ts: 10 },
        { kind: "action", tool: "mouse.move", summary: "m", ts: 1 },
      ],
    });
    const [sequence] = tokenizeTrajectories([span]);
    expect(sequence?.tokens).toEqual(["action:mouse.move", "obs:screen", "action:mouse.click"]);
  });
});

describe("buildMovementDataset", () => {
  it("collects a sorted, de-duplicated vocabulary", () => {
    const dataset = buildMovementDataset([
      { id: "a", tokens: ["action:b", "action:a"] },
      { id: "b", tokens: ["action:a", "obs:x"] },
    ]);
    expect(dataset.vocabulary).toEqual(["action:a", "action:b", "obs:x"]);
    expect(dataset.sequences).toHaveLength(2);
  });
});

describe("NGramMovementBackend — replay fidelity", () => {
  it("reproduces a recorded movement sequence exactly via rollout", async () => {
    const recorded: MovementSequence = {
      id: "recorded",
      tokens: ["obs:screen", "action:mouse.move", "action:mouse.click", "action:key.type", "action:key.enter"],
    };
    const dataset = buildMovementDataset([recorded]);
    const model = await new NGramMovementBackend(3).train(dataset);

    const generated = model.rollout([recorded.tokens[0]!]);
    expect([recorded.tokens[0], ...generated]).toEqual(recorded.tokens);
  });

  it("terminates rollouts at the end sentinel instead of running to maxSteps", async () => {
    const dataset = buildMovementDataset([{ id: "r", tokens: ["action:a", "action:b"] }]);
    const model = await new NGramMovementBackend(2).train(dataset);
    const generated = model.rollout(["action:a"], 100);
    expect(generated).toEqual(["action:b"]);
    expect(generated).not.toContain(MOVEMENT_END_TOKEN);
  });

  it("scores full next-token accuracy on a sequence it was trained on", async () => {
    const seq: MovementSequence = { id: "r", tokens: ["action:a", "action:b", "action:c"] };
    const model = await new NGramMovementBackend(3).train(buildMovementDataset([seq]));
    const result = evaluateNextTokenAccuracy(model, seq);
    // Every token after the first (which has no predictive context) is nailed.
    expect(result.correct).toBe(result.total - 1);
  });
});

describe("NGramMovementBackend — generalization via backoff", () => {
  it("predicts a next movement for an unseen-but-related context by backing off", async () => {
    // Two related demos: after a click, the recorded habit is to type.
    const dataset = buildMovementDataset([
      { id: "d1", tokens: ["obs:screen", "action:mouse.click", "action:key.type"] },
      { id: "d2", tokens: ["obs:menu", "action:mouse.click", "action:key.type"] },
    ]);
    const model = await new NGramMovementBackend(3).train(dataset);

    // Held-out context: a brand-new observation the model never saw before the
    // click. The full-order trigram context is unseen, so it must back off.
    const prediction = model.predictNext(["obs:dialog", "action:mouse.click"]);
    expect(prediction).toBeDefined();
    expect(prediction?.token).toBe("action:key.type");
    expect(prediction?.generalized).toBe(true);
    expect(prediction?.matchedContextLength).toBe(1);
  });

  it("falls back to the unigram prior when no context matches", async () => {
    const dataset = buildMovementDataset([{ id: "d", tokens: ["action:a", "action:a", "action:b"] }]);
    const model = await new NGramMovementBackend(3).train(dataset);
    const prediction = model.predictNext(["totally:unknown"]);
    expect(prediction?.token).toBe("action:a"); // most frequent token overall
    expect(prediction?.matchedContextLength).toBe(0);
  });

  it("measures generalization on a held-out related trajectory", async () => {
    const train = buildMovementDataset([
      { id: "t1", tokens: ["action:open", "action:scroll", "action:read", "action:close"] },
      { id: "t2", tokens: ["action:open", "action:scroll", "action:read", "action:close"] },
    ]);
    const model = await new NGramMovementBackend(3).train(train);
    // Held-out sequence shares the sub-movements but starts differently.
    const heldOut: MovementSequence = { id: "eval", tokens: ["action:scroll", "action:read", "action:close"] };
    const result = evaluateNextTokenAccuracy(model, heldOut);
    expect(result.correct).toBeGreaterThan(0);
    expect(result.accuracy).toBeGreaterThan(0.5);
  });
});

describe("NGramMovementBackend — determinism & serialization", () => {
  it("produces identical models for identical datasets", async () => {
    const dataset = buildMovementDataset([
      { id: "a", tokens: ["action:x", "action:y", "action:z"] },
      { id: "b", tokens: ["action:x", "action:y", "action:w"] },
    ]);
    const first = await new NGramMovementBackend(3).train(dataset);
    const second = await new NGramMovementBackend(3).train(dataset);
    expect(first.serialize()).toEqual(second.serialize());
  });

  it("round-trips through a snapshot without retraining", async () => {
    const dataset = buildMovementDataset([{ id: "a", tokens: ["action:x", "action:y", "action:z"] }]);
    const model = await new NGramMovementBackend(3).train(dataset);
    const restored = NGramMovementBackend.fromSnapshot(model.serialize());
    expect(restored.predictNext(["action:x", "action:y"])?.token).toBe("action:z");
    expect(restored.serialize()).toEqual(model.serialize());
  });

  it("returns undefined from an empty model", async () => {
    const model = await new NGramMovementBackend(3).train(buildMovementDataset([]));
    expect(model.predictNext(["anything"])).toBeUndefined();
    expect(model.rollout(["anything"])).toEqual([]);
  });
});
