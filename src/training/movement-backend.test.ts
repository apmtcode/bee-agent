import { describe, expect, it } from "vitest";
import type { ReplayTimelineEvent } from "../capture/replay.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import {
  MOVEMENT_END_TOKEN,
  MarkovMovementBackend,
  MovementBackendRegistry,
  buildMovementDataset,
  evaluateMovementFidelity,
  tokenizeReplayEvents,
  tokenizeTrajectory,
  type LocalMovementBackend,
  type MovementDataset,
  type MovementSequence,
  type TrainedMovementModel,
} from "./movement-backend.js";

function seq(trajectoryId: string, tokens: string[]): MovementSequence {
  return { trajectoryId, tokens };
}

describe("MarkovMovementBackend", () => {
  it("trains a serializable model with vocabulary and transitions", async () => {
    const backend = new MarkovMovementBackend();
    const dataset: MovementDataset = {
      version: 1,
      sequences: [
        seq("t1", ["obs:screen", "action:move", "action:click"]),
        seq("t2", ["obs:screen", "action:move", "action:click"]),
      ],
    };

    const model = await backend.train(dataset, { order: 2 });

    expect(model.backend).toBe("markov-mock");
    expect(model.sequenceCount).toBe(2);
    expect(model.vocabulary).toContain("action:click");
    expect(model.vocabulary).toContain(MOVEMENT_END_TOKEN);
    // Model must round-trip through JSON (persistable next to job artifacts).
    expect(() => JSON.parse(JSON.stringify(model))).not.toThrow();
  });

  it("reproduces a recorded movement sequence exactly from its start", async () => {
    const backend = new MarkovMovementBackend();
    const recorded = ["obs:screen", "action:focus", "action:type", "action:submit"];
    const model = await backend.train({ version: 1, sequences: [seq("t1", recorded)] }, { order: 2 });

    const prediction = await backend.predict(model, { context: ["obs:screen"], maxTokens: 10 });

    expect(prediction.tokens).toEqual(["action:focus", "action:type", "action:submit"]);
    expect(prediction.reachedEnd).toBe(true);
    expect(prediction.confidence).toBeGreaterThan(0);
  });

  it("is deterministic: identical input yields identical output", async () => {
    const backend = new MarkovMovementBackend();
    const model = await backend.train(
      {
        version: 1,
        sequences: [
          seq("a", ["obs:x", "action:a", "action:b"]),
          seq("b", ["obs:x", "action:a", "action:c"]),
        ],
      },
      { order: 1 },
    );

    const first = await backend.predict(model, { context: ["obs:x"], maxTokens: 5 });
    const second = await backend.predict(model, { context: ["obs:x"], maxTokens: 5 });
    expect(first).toEqual(second);
  });

  it("breaks ties by token string deterministically", async () => {
    const backend = new MarkovMovementBackend();
    // "action:z" and "action:a" are equally likely after obs:x — expect the
    // lexicographically-smallest ("action:a") every time.
    const model = await backend.train(
      {
        version: 1,
        sequences: [
          seq("a", ["obs:x", "action:z"]),
          seq("b", ["obs:x", "action:a"]),
        ],
      },
      { order: 1 },
    );
    const prediction = await backend.predict(model, { context: ["obs:x"], maxTokens: 1 });
    expect(prediction.tokens[0]).toBe("action:a");
  });

  it("generalizes to an unseen prefix via context back-off", async () => {
    const backend = new MarkovMovementBackend();
    // Every recorded trajectory: after action:move, click. The model never saw
    // the prefix ["obs:widget", "action:move"] but should still predict click
    // by backing off to the shorter ["action:move"] context.
    const model = await backend.train(
      {
        version: 1,
        sequences: [
          seq("t1", ["obs:screen", "action:move", "action:click"]),
          seq("t2", ["obs:window", "action:move", "action:click"]),
          seq("t3", ["obs:panel", "action:move", "action:click"]),
        ],
      },
      { order: 2 },
    );

    const prediction = await backend.predict(model, {
      context: ["obs:widget", "action:move"],
      maxTokens: 1,
    });
    expect(prediction.tokens[0]).toBe("action:click");
  });

  it("returns empty prediction for an empty model", async () => {
    const backend = new MarkovMovementBackend();
    const model = await backend.train({ version: 1, sequences: [] });
    const prediction = await backend.predict(model, { context: ["obs:x"], maxTokens: 5 });
    expect(prediction.tokens).toEqual([]);
    expect(prediction.confidence).toBe(0);
    expect(prediction.reachedEnd).toBe(false);
  });

  it("respects maxTokens and can keep going past the end sentinel", async () => {
    const backend = new MarkovMovementBackend();
    const model = await backend.train(
      { version: 1, sequences: [seq("t1", ["a", "b", "c", "d"])] },
      { order: 3 },
    );
    const capped = await backend.predict(model, { context: ["a"], maxTokens: 2 });
    expect(capped.tokens).toEqual(["b", "c"]);

    const past = await backend.predict(model, { context: ["a"], maxTokens: 10, stopAtEnd: false });
    // Reaching the end sentinel does not append it as a movement token.
    expect(past.tokens.slice(0, 3)).toEqual(["b", "c", "d"]);
    expect(past.tokens).not.toContain(MOVEMENT_END_TOKEN);
  });
});

describe("MovementBackendRegistry", () => {
  it("resolves the deterministic backend by default", () => {
    const registry = new MovementBackendRegistry();
    expect(registry.resolve().name).toBe("markov-mock");
    expect(registry.list()).toContain("markov-mock");
    expect(registry.has("markov-mock")).toBe(true);
  });

  it("registers a pluggable backend and can make it the default", () => {
    const registry = new MovementBackendRegistry();
    const custom: LocalMovementBackend = {
      name: "mlx-on-device",
      async train(): Promise<TrainedMovementModel> {
        return {
          backend: "mlx-on-device",
          version: 1,
          order: 1,
          vocabulary: [],
          transitions: {},
          sequenceCount: 0,
          tokenCount: 0,
        };
      },
      async predict() {
        return { tokens: ["action:noop"], confidence: 1, reachedEnd: true };
      },
    };

    registry.register(custom, { makeDefault: true });
    expect(registry.resolve().name).toBe("mlx-on-device");
    expect(registry.resolve("markov-mock").name).toBe("markov-mock");
    expect(registry.list()).toEqual(["markov-mock", "mlx-on-device"]);
  });

  it("throws a helpful error for an unknown backend", () => {
    const registry = new MovementBackendRegistry();
    expect(() => registry.resolve("does-not-exist")).toThrow(/Unknown movement backend/);
  });
});

describe("tokenizers", () => {
  it("tokenizes replay events to coarse movement tokens", () => {
    const events: ReplayTimelineEvent[] = [
      { kind: "observation", ts: 1, trajectoryId: "t1", source: "Screen Reader", summary: "s" },
      { kind: "transcript", ts: 2, messageId: "m", role: "user", content: "hi" },
      { kind: "action", ts: 3, trajectoryId: "t1", tool: "Mouse Click", summary: "click ok" },
    ];
    expect(tokenizeReplayEvents(events)).toEqual(["obs:screen-reader", "action:mouse-click"]);
    expect(tokenizeReplayEvents(events, { includeTranscript: true })).toEqual([
      "obs:screen-reader",
      "msg:user",
      "action:mouse-click",
    ]);
    expect(tokenizeReplayEvents(events, { includeObservations: false })).toEqual(["action:mouse-click"]);
  });

  it("tokenizes a trajectory span in timestamp order", () => {
    const span = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      observations: [{ kind: "observation", source: "screen", summary: "o", ts: 5 }],
      actions: [
        { kind: "action", tool: "click", summary: "c", ts: 10 },
        { kind: "action", tool: "move", summary: "m", ts: 2 },
      ],
    });
    const tokenized = tokenizeTrajectory(span);
    expect(tokenized.trajectoryId).toBe("t1");
    // ts order: move(2) < screen(5) < click(10)
    expect(tokenized.tokens).toEqual(["action:move", "obs:screen", "action:click"]);
  });

  it("builds a dataset from replay manifests, skipping empty ones", () => {
    const dataset = buildMovementDataset([
      {
        trajectoryIds: ["t1"],
        events: [{ kind: "action", ts: 1, trajectoryId: "t1", tool: "click", summary: "c" }],
      },
      { trajectoryIds: ["t2"], events: [] },
    ]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]?.tokens).toEqual(["action:click"]);
  });
});

describe("evaluateMovementFidelity", () => {
  it("scores perfect reproduction of a held-out recorded prefix", async () => {
    const backend = new MarkovMovementBackend();
    const sequences = [
      seq("t1", ["obs:screen", "action:move", "action:click"]),
      seq("t2", ["obs:screen", "action:move", "action:click"]),
    ];
    const model = await backend.train({ version: 1, sequences }, { order: 2 });

    const result = await evaluateMovementFidelity(backend, model, sequences, { promptTokens: 1 });
    expect(result.tokenAccuracy).toBe(1);
    expect(result.exactSequenceMatch).toBe(1);
    expect(result.evaluatedSequences).toBe(2);
  });

  it("measures partial fidelity on held-out related sequences", async () => {
    const backend = new MarkovMovementBackend();
    const train = [
      seq("t1", ["obs:a", "action:move", "action:click"]),
      seq("t2", ["obs:b", "action:move", "action:click"]),
    ];
    const model = await backend.train({ version: 1, sequences: train }, { order: 2 });

    // Held-out sequence shares the move->click pattern but ends differently.
    const heldOut = [seq("t3", ["obs:c", "action:move", "action:drag"])];
    const result = await evaluateMovementFidelity(backend, model, heldOut, { promptTokens: 2 });
    // Prompt is [obs:c, action:move]; model predicts action:click (generalized),
    // ground truth is action:drag -> 0 matches of the 1 held-out token.
    expect(result.evaluatedTokens).toBe(1);
    expect(result.tokenAccuracy).toBe(0);
    expect(result.exactSequenceMatch).toBe(0);
  });
});
