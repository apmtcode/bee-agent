import { describe, expect, it } from "vitest";
import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import {
  NGramMovementBackend,
  buildMovementDataset,
  movementSequenceFromReplay,
  replayFidelity,
  tokenizeReplayEvent,
  type MovementDataset,
} from "./movement-model.js";

function action(tool: string, summary: string, ts: number): ReplayTimelineEvent {
  return { kind: "action", ts, trajectoryId: "traj", tool, summary };
}

function observation(source: string, summary: string, ts: number): ReplayTimelineEvent {
  return { kind: "observation", ts, trajectoryId: "traj", source, summary };
}

function manifest(sessionId: string, events: ReplayTimelineEvent[]): ReplayManifest {
  return {
    version: 1,
    sessionId,
    trajectoryIds: [`${sessionId}-traj`],
    eventCount: events.length,
    events,
  };
}

describe("tokenizeReplayEvent", () => {
  it("encodes actions and observations, ignores transcript dialogue", () => {
    expect(tokenizeReplayEvent(action("device", "tapped submit", 1))).toBe("action:device:tapped submit");
    expect(tokenizeReplayEvent(observation("device", "cart screen", 1))).toBe("obs:device:cart screen");
    expect(
      tokenizeReplayEvent({ kind: "transcript", ts: 1, messageId: "m1", role: "user", content: "hi" }),
    ).toBeUndefined();
  });
});

describe("buildMovementDataset", () => {
  it("derives sequences from replays and drops empty/transcript-only ones", () => {
    const replays = [
      manifest("s1", [observation("device", "cart", 1), action("device", "tap checkout", 2)]),
      manifest("s2", [{ kind: "transcript", ts: 1, messageId: "m", role: "user", content: "x" }]),
    ];
    const dataset = buildMovementDataset(replays);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]!.tokens).toEqual(["obs:device:cart", "action:device:tap checkout"]);
    expect(dataset.sequences[0]!.trajectoryIds).toEqual(["s1-traj"]);
  });

  it("movementSequenceFromReplay preserves timeline order", () => {
    const sequence = movementSequenceFromReplay(
      manifest("s", [action("a", "one", 1), action("a", "two", 2), action("a", "three", 3)]),
    );
    expect(sequence.tokens).toEqual(["action:a:one", "action:a:two", "action:a:three"]);
  });
});

describe("NGramMovementBackend training + replay", () => {
  const recorded = manifest("s1", [
    observation("device", "cart screen", 1),
    action("device", "tap checkout", 2),
    action("device", "type card number", 3),
    action("device", "tap pay", 4),
  ]);

  it("reproduces a recorded movement exactly from its first token", async () => {
    const dataset = buildMovementDataset([recorded]);
    const model = await new NGramMovementBackend().train(dataset);
    const tokens = dataset.sequences[0]!.tokens;

    const generated = model.generate([tokens[0]!], 10);
    expect(generated).toEqual(tokens.slice(1));
    // The exact-order match should be reported as an "exact" prediction.
    const prediction = model.predictNext([tokens[0]!]);
    expect(prediction?.token).toBe(tokens[1]);
    expect(prediction?.source).toBe("exact");
  });

  it("terminates at end-of-sequence instead of looping", async () => {
    const dataset = buildMovementDataset([recorded]);
    const model = await new NGramMovementBackend().train(dataset);
    const generated = model.generate([dataset.sequences[0]!.tokens[0]!], 100);
    // Even with a large step budget it stops when the recording ends.
    expect(generated.length).toBe(dataset.sequences[0]!.tokens.length - 1);
  });

  it("is deterministic across repeated training runs", async () => {
    const dataset = buildMovementDataset([recorded]);
    const a = (await new NGramMovementBackend().train(dataset)).serialize();
    const b = (await new NGramMovementBackend().train(dataset)).serialize();
    expect(a).toEqual(b);
  });
});

describe("NGramMovementBackend generalization", () => {
  it("generalizes to a novel-but-related context via backoff", async () => {
    // Two recordings share the sub-pattern "open editor" -> "save file".
    const dataset = buildMovementDataset([
      manifest("s1", [
        action("app", "focus window"),
        action("app", "open editor"),
        action("app", "save file"),
      ].map((event, index) => ({ ...event, ts: index }))),
      manifest("s2", [
        action("app", "switch tab"),
        action("app", "open editor"),
        action("app", "save file"),
      ].map((event, index) => ({ ...event, ts: index }))),
    ]);
    const model = await new NGramMovementBackend({ order: 3 }).train(dataset);

    // A context never seen at full order — "close panel" then "open editor" — should
    // still predict "save file" by backing off to the shorter learned suffix.
    const prediction = model.predictNext(["action:app:close panel", "action:app:open editor"]);
    expect(prediction?.token).toBe("action:app:save file");
    expect(prediction?.source).toBe("backoff");
    expect(prediction?.matchedOrder).toBeLessThan(2);
  });

  it("falls back to the unigram prior for a fully unseen context", async () => {
    const dataset = buildMovementDataset([
      manifest("s1", [action("app", "click"), action("app", "click"), action("app", "scroll")].map(
        (event, index) => ({ ...event, ts: index }),
      )),
    ]);
    const model = await new NGramMovementBackend().train(dataset);
    const prediction = model.predictNext(["action:app:totally novel"]);
    expect(prediction).toBeDefined();
    // "click" is the most frequent token, so the prior favors it.
    expect(prediction?.token).toBe("action:app:click");
    expect(prediction?.source).toBe("prior");
    expect(prediction?.matchedOrder).toBe(0);
  });
});

describe("serialize / restore", () => {
  it("round-trips a trained model to an identical predictor", async () => {
    const dataset = buildMovementDataset([
      manifest("s1", [action("a", "one"), action("a", "two"), action("a", "three")].map((event, index) => ({
        ...event,
        ts: index,
      }))),
    ]);
    const backend = new NGramMovementBackend();
    const model = await backend.train(dataset);
    const snapshot = model.serialize();

    const restored = backend.restore(snapshot);
    expect(restored.serialize()).toEqual(snapshot);
    expect(restored.generate(["action:a:one"], 5)).toEqual(model.generate(["action:a:one"], 5));
  });

  it("snapshot is stable (sorted) and JSON-serializable", async () => {
    const dataset: MovementDataset = buildMovementDataset([
      manifest("s1", [action("a", "b"), action("a", "c")].map((event, index) => ({ ...event, ts: index }))),
    ]);
    const model = await new NGramMovementBackend().train(dataset);
    const snapshot = model.serialize();
    expect(() => JSON.stringify(snapshot)).not.toThrow();
    expect(snapshot.vocabulary).toEqual([...snapshot.vocabulary].sort());
  });
});

describe("replayFidelity", () => {
  it("scores exact reproduction as 1 and a first-step divergence proportionally", () => {
    expect(replayFidelity(["a", "b", "c"], ["a", "b", "c"])).toBe(1);
    expect(replayFidelity(["a", "b", "c", "d"], ["a", "b", "x", "y"])).toBe(0.5);
    expect(replayFidelity(["a", "b"], ["x"])).toBe(0);
    expect(replayFidelity([], [])).toBe(1);
  });

  it("measures generalization fidelity end-to-end", async () => {
    const train = buildMovementDataset([
      manifest("s1", [action("a", "login"), action("a", "dashboard"), action("a", "logout")].map(
        (event, index) => ({ ...event, ts: index }),
      )),
    ]);
    const model = await new NGramMovementBackend().train(train);
    const generated = model.generate(["action:a:login"], 10);
    const fidelity = replayFidelity(["action:a:dashboard", "action:a:logout"], generated);
    expect(fidelity).toBe(1);
  });
});
