import { describe, expect, it } from "vitest";
import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import {
  MarkovMovementBackend,
  MovementBackendRegistry,
  buildMovementDataset,
  createDefaultMovementBackendRegistry,
  evaluateReplayFidelity,
  generateMovements,
  tokenizeEvent,
  toMovementSequence,
  type MovementModelBackend,
  type MovementSequence,
  type MovementTrainingDataset,
  type TrainedMovementModel,
} from "./movement-model.js";

function action(tool: string, summary: string, ts: number): ReplayTimelineEvent {
  return { kind: "action", ts, trajectoryId: "t", tool, summary };
}

function manifest(sessionId: string, events: ReplayTimelineEvent[]): ReplayManifest {
  return {
    version: 1,
    sessionId,
    trajectoryIds: ["t"],
    eventCount: events.length,
    events,
  };
}

// A repeated "open → type → submit" workflow used across most tests.
const workflow = manifest("s1", [
  action("device", "focused app", 1),
  action("device", "typed into search", 2),
  action("device", "tapped submit", 3),
]);

describe("tokenizeEvent", () => {
  it("canonicalizes actions, observations, and transcript messages", () => {
    expect(tokenizeEvent(action("device", "Tapped   Submit", 1))).toBe("action:device:tapped submit");
    expect(
      tokenizeEvent({ kind: "observation", ts: 1, trajectoryId: "t", source: "os", summary: "x" }),
    ).toBe("obs:os");
    expect(
      tokenizeEvent({ kind: "transcript", ts: 1, messageId: "m", role: "user", content: "hi" }),
    ).toBe("msg:user");
  });
});

describe("buildMovementDataset", () => {
  it("keeps only action movements by default", () => {
    const withObs = manifest("s", [
      { kind: "observation", ts: 1, trajectoryId: "t", source: "os", summary: "screen" },
      action("device", "tapped submit", 2),
    ]);
    const dataset = buildMovementDataset([withObs]);
    expect(dataset.sequences[0].tokens).toEqual(["action:device:tapped submit"]);
  });

  it("can include observations when asked", () => {
    const seq = toMovementSequence(
      manifest("s", [{ kind: "observation", ts: 1, trajectoryId: "t", source: "os", summary: "screen" }]),
      { includeObservations: true },
    );
    expect(seq.tokens).toEqual(["obs:os"]);
  });
});

describe("MarkovMovementBackend", () => {
  it("trains transition counts and a sorted vocabulary", async () => {
    const backend = new MarkovMovementBackend();
    const model = await backend.train(buildMovementDataset([workflow]), { order: 2 });
    expect(model.backendId).toBe("markov");
    expect(model.order).toBe(2);
    expect(model.trainedSequenceCount).toBe(1);
    expect(model.trainedTokenCount).toBe(3);
    expect(model.vocabulary).toEqual([
      "action:device:focused app",
      "action:device:tapped submit",
      "action:device:typed into search",
    ]);
  });

  it("predicts the next recorded movement from an exact context", async () => {
    const backend = new MarkovMovementBackend();
    const model = await backend.train(buildMovementDataset([workflow]), { order: 2 });
    const prediction = backend.predict(model, [
      "action:device:focused app",
      "action:device:typed into search",
    ]);
    expect(prediction.token).toBe("action:device:tapped submit");
    expect(prediction.contextOrderUsed).toBe(2);
    expect(prediction.probability).toBe(1);
  });

  it("generalizes via backoff when the full-order context is unseen", async () => {
    const backend = new MarkovMovementBackend();
    const model = await backend.train(buildMovementDataset([workflow]), { order: 2 });
    // Novel 2-token prefix, but its 1-token suffix ("typed into search") was
    // seen; backoff should still predict the related next movement.
    const prediction = backend.predict(model, [
      "action:device:some novel move",
      "action:device:typed into search",
    ]);
    expect(prediction.token).toBe("action:device:tapped submit");
    expect(prediction.contextOrderUsed).toBe(1);
  });

  it("falls back to the unigram for a wholly unseen context", async () => {
    const backend = new MarkovMovementBackend();
    // Two sequences so the unigram has a clear, deterministic mode.
    const dataset = buildMovementDataset([workflow, workflow]);
    const model = await backend.train(dataset, { order: 2 });
    const prediction = backend.predict(model, ["action:device:totally unknown"]);
    expect(prediction.contextOrderUsed).toBe(0);
    expect(prediction.token).toBeDefined();
  });

  it("returns an empty prediction when nothing was trained", async () => {
    const backend = new MarkovMovementBackend();
    const model = await backend.train({ sequences: [] });
    const prediction = backend.predict(model, ["anything"]);
    expect(prediction.token).toBeUndefined();
    expect(prediction.contextOrderUsed).toBe(-1);
    expect(prediction.candidates).toEqual([]);
  });

  it("is deterministic — identical training yields identical models", async () => {
    const backend = new MarkovMovementBackend();
    const a = await backend.train(buildMovementDataset([workflow]), { order: 2 });
    const b = await backend.train(buildMovementDataset([workflow]), { order: 2 });
    expect(JSON.stringify(a)).toBe(JSON.stringify(b));
  });

  it("breaks ties deterministically by count then lexicographically", async () => {
    const backend = new MarkovMovementBackend();
    const seq: MovementSequence = { id: "s", tokens: ["a", "z", "a", "b"] };
    // After "a": once "z", once "b" — equal counts, so "b" wins lexicographically.
    const model = await backend.train({ sequences: [seq] }, { order: 1 });
    const prediction = backend.predict(model, ["a"]);
    expect(prediction.token).toBe("b");
    expect(prediction.candidates.map((c) => c.token)).toEqual(["b", "z"]);
  });
});

describe("generateMovements", () => {
  it("reproduces a recorded workflow from its opening movement", async () => {
    const backend = new MarkovMovementBackend();
    const model = await backend.train(buildMovementDataset([workflow]), { order: 2 });
    const rollout = generateMovements(backend, model, ["action:device:focused app"], {
      maxLength: 5,
      minContextOrder: 1,
    });
    expect(rollout).toEqual([
      "action:device:typed into search",
      "action:device:tapped submit",
    ]);
  });

  it("stops on a repeat-guard instead of looping forever", async () => {
    const backend = new MarkovMovementBackend();
    // Self-transition: "loop" always follows "loop".
    const model = await backend.train({ sequences: [{ id: "s", tokens: ["loop", "loop", "loop", "loop"] }] }, {
      order: 1,
    });
    const rollout = generateMovements(backend, model, ["loop"], { maxLength: 100, maxRepeat: 3 });
    expect(rollout.length).toBeLessThanOrEqual(3);
    expect(new Set(rollout)).toEqual(new Set(["loop"]));
  });
});

describe("evaluateReplayFidelity", () => {
  it("scores next-movement accuracy on a held-out related sequence", async () => {
    const backend = new MarkovMovementBackend();
    const model = await backend.train(buildMovementDataset([workflow]), { order: 2 });
    // Held-out sequence shares the tail of the trained workflow.
    const heldOut: MovementSequence[] = [
      { id: "h", tokens: ["action:device:typed into search", "action:device:tapped submit"] },
    ];
    const result = evaluateReplayFidelity(backend, model, heldOut);
    // 1st token predicts from empty context (unigram, order 0) and misses;
    // 2nd token matches the order-1 context and hits.
    expect(result.total).toBe(2);
    expect(result.correct).toBe(1);
    expect(result.accuracy).toBe(0.5);
    expect(result.byContextOrder[0]).toEqual({ total: 1, correct: 0 });
    expect(result.byContextOrder[1]).toEqual({ total: 1, correct: 1 });
  });
});

describe("MovementBackendRegistry", () => {
  it("registers, resolves default, and swaps in an alternate backend", async () => {
    const registry = createDefaultMovementBackendRegistry();
    expect(registry.list()).toEqual(["markov"]);
    expect(registry.get().id).toBe("markov");

    // A pluggable alternate backend standing in for a real on-device model.
    const stub: MovementModelBackend = {
      id: "on-device-stub",
      async train(dataset: MovementTrainingDataset): Promise<TrainedMovementModel> {
        return {
          version: 1,
          backendId: "on-device-stub",
          order: 0,
          vocabulary: [],
          transitions: {},
          trainedSequenceCount: dataset.sequences.length,
          trainedTokenCount: 0,
        };
      },
      predict() {
        return { token: "stub-move", probability: 1, contextOrderUsed: 0, candidates: [] };
      },
    };
    registry.register(stub);
    expect(registry.has("on-device-stub")).toBe(true);
    expect(registry.get("on-device-stub").predict({} as TrainedMovementModel, []).token).toBe("stub-move");
    // Default is unchanged because makeDefault was not set.
    expect(registry.get().id).toBe("markov");
  });

  it("throws for an unknown backend id", () => {
    const registry = createDefaultMovementBackendRegistry();
    expect(() => registry.get("missing")).toThrow(/unknown movement backend/);
  });
});
