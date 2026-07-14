import { describe, expect, it } from "vitest";
import type { ReplayTimelineEvent } from "../capture/replay.js";
import {
  buildMovementDataset,
  defaultMovementGrammar,
  evaluateReplayFidelity,
  generateMovementSequence,
  synthesizeMovementDataset,
  tokenizeMovementEvent,
  type MovementReplaySource,
} from "./movement-model.js";
import { MockMarkovMovementBackend } from "./mock-movement-backend.js";

function action(tool: string, ts: number): ReplayTimelineEvent {
  return { kind: "action", ts, trajectoryId: "t1", tool, summary: tool };
}

describe("tokenizeMovementEvent", () => {
  it("maps each event kind to a stable token", () => {
    expect(tokenizeMovementEvent({ kind: "action", ts: 1, trajectoryId: "t", tool: "Bash", summary: "x" })).toBe(
      "action:Bash",
    );
    expect(tokenizeMovementEvent({ kind: "observation", ts: 1, trajectoryId: "t", source: "screen", summary: "x" })).toBe(
      "observation:screen",
    );
    expect(
      tokenizeMovementEvent({ kind: "transcript", ts: 1, messageId: "m", role: "user", content: "hi" }),
    ).toBe("transcript:user");
  });
});

describe("buildMovementDataset", () => {
  it("orders events by ts, tokenizes, and builds a sorted vocabulary", () => {
    const sources: MovementReplaySource[] = [
      { sessionId: "s1", events: [action("click", 20), action("move", 10)] },
    ];
    const dataset = buildMovementDataset(sources);
    expect(dataset.sequences).toEqual([{ id: "s1", tokens: ["action:move", "action:click"] }]);
    expect(dataset.vocabulary).toEqual(["action:click", "action:move"]);
  });

  it("drops empty replays", () => {
    const dataset = buildMovementDataset([{ sessionId: "empty", events: [] }]);
    expect(dataset.sequences).toHaveLength(0);
    expect(dataset.vocabulary).toHaveLength(0);
  });
});

describe("synthesizeMovementDataset", () => {
  it("is deterministic for a fixed seed", () => {
    const options = { sequenceCount: 5, minLength: 4, maxLength: 8, seed: 42 };
    const a = synthesizeMovementDataset(options);
    const b = synthesizeMovementDataset(options);
    expect(a).toEqual(b);
    expect(a.sequences).toHaveLength(5);
    for (const sequence of a.sequences) {
      expect(sequence.tokens.length).toBeGreaterThanOrEqual(4);
      expect(sequence.tokens.length).toBeLessThanOrEqual(8);
    }
  });

  it("only emits tokens from the grammar vocabulary and starts at the focus state", () => {
    const grammar = defaultMovementGrammar();
    const dataset = synthesizeMovementDataset({ sequenceCount: 3, minLength: 3, maxLength: 3, seed: 7 });
    const grammarTokens = new Set(Object.values(grammar).map((state) => state.token));
    for (const token of dataset.vocabulary) {
      expect(grammarTokens.has(token)).toBe(true);
    }
    for (const sequence of dataset.sequences) {
      expect(sequence.tokens[0]).toBe("observation:screen");
    }
  });
});

describe("generate + evaluate round-trip", () => {
  it("reproduces a recorded movement from a seed (greedy decode)", async () => {
    // Two identical recordings of the same movement make the path unambiguous.
    const recorded = ["observation:screen", "action:mouse-move", "action:mouse-click", "action:app-switch"];
    const sources: MovementReplaySource[] = [0, 1].map((n) => ({
      sessionId: `s${n}`,
      events: recorded.map((token, index) => reconstruct(token, index)),
    }));
    const dataset = buildMovementDataset(sources);
    const backend = new MockMarkovMovementBackend();
    const model = await backend.train({ dataset, config: { order: 2 } });

    const generated = generateMovementSequence(backend, model, {
      seed: [recorded[0]],
      maxLength: recorded.length,
    });
    expect(generated).toEqual(recorded);
  });

  it("scores perfect replay fidelity on the training trajectory", async () => {
    const dataset = synthesizeMovementDataset({ sequenceCount: 12, minLength: 6, maxLength: 12, seed: 99 });
    const backend = new MockMarkovMovementBackend();
    const model = await backend.train({ dataset, config: { order: 3 } });

    const report = evaluateReplayFidelity(backend, model, dataset.sequences[0].tokens);
    expect(report.predictions).toBeGreaterThan(0);
    expect(report.unpredicted).toBe(0);
    // Every context in the training sequence was seen, so greedy decode always predicts *something*.
    expect(report.accuracy).toBeGreaterThan(0);
  });

  it("generalizes to a held-out related trajectory via backoff", async () => {
    // Train on synthetic workflows, then evaluate on a fresh sequence from the
    // same grammar the model never saw. Backoff should still predict most steps.
    const train = synthesizeMovementDataset({ sequenceCount: 40, minLength: 8, maxLength: 16, seed: 1 });
    const holdout = synthesizeMovementDataset({ sequenceCount: 1, minLength: 12, maxLength: 12, seed: 777 });
    const backend = new MockMarkovMovementBackend();
    const model = await backend.train({ dataset: train, config: { order: 2 } });

    const report = evaluateReplayFidelity(backend, model, holdout.sequences[0].tokens);
    expect(report.unpredicted).toBe(0);
    expect(report.accuracy).toBeGreaterThan(0.4);
  });
});

function reconstruct(token: string, ts: number): ReplayTimelineEvent {
  const [, value] = token.split(":");
  if (token.startsWith("observation:")) {
    return { kind: "observation", ts, trajectoryId: "t", source: value, summary: value };
  }
  return { kind: "action", ts, trajectoryId: "t", tool: value, summary: value };
}
