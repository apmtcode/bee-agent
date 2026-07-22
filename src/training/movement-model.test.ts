import { describe, expect, it } from "vitest";
import type { ReplayManifest } from "../capture/replay.js";
import {
  DeterministicMarkovBackend,
  MOVEMENT_END_TOKEN,
  buildMovementDataset,
  createMovementModelBackend,
  movementTokenFromAction,
  normalizeMovementSummary,
  type MovementDataset,
} from "./movement-model.js";

function dataset(sequences: string[][]): MovementDataset {
  return {
    version: 1,
    sequences: sequences.map((tokens, index) => ({ id: `seq-${index}`, tokens })),
  };
}

describe("normalizeMovementSummary", () => {
  it("collapses whitespace and punctuation to a stable slug", () => {
    expect(normalizeMovementSummary("tapped Send button")).toBe("tapped-send-button");
    expect(normalizeMovementSummary("  tapped   Send   Button!! ")).toBe("tapped-send-button");
  });
});

describe("DeterministicMarkovBackend", () => {
  it("trains a serializable artifact with vocabulary and counts", () => {
    const backend = new DeterministicMarkovBackend();
    const model = backend.train(dataset([["a", "b", "c"]]), { order: 2 });

    expect(model.backendId).toBe("deterministic-markov");
    expect(model.order).toBe(2);
    expect(model.vocabulary).toEqual(["a", "b", "c"]);
    expect(model.sequenceCount).toBe(1);
    expect(model.tokenCount).toBe(3);
    // Round-trips through JSON — it is a local model artifact on disk.
    expect(JSON.parse(JSON.stringify(model))).toEqual(model);
  });

  it("repeats a recorded movement deterministically (objective 2c)", () => {
    const backend = new DeterministicMarkovBackend();
    const recorded = ["open-app", "click-compose", "type-subject", "click-send"];
    const model = backend.train(dataset([recorded]), { order: 2 });

    // From a cold start it should reproduce the single memorized trajectory.
    expect(backend.generate(model, [])).toEqual(recorded);
  });

  it("predicts the most frequent continuation with a real probability", () => {
    const backend = new DeterministicMarkovBackend();
    // After "a", "b" appears 3x and "c" 1x.
    const model = backend.train(
      dataset([
        ["a", "b"],
        ["a", "b"],
        ["a", "b"],
        ["a", "c"],
      ]),
      { order: 1 },
    );

    const prediction = backend.predictNext(model, ["a"]);
    expect(prediction.token).toBe("b");
    expect(prediction.probability).toBeCloseTo(0.75);
    expect(prediction.backoffOrder).toBe(1);
    expect(prediction.alternatives[0]).toEqual({ token: "c", probability: 0.25 });
  });

  it("breaks ties deterministically by token order", () => {
    const backend = new DeterministicMarkovBackend();
    const model = backend.train(
      dataset([
        ["x", "zebra"],
        ["x", "apple"],
      ]),
      { order: 1 },
    );
    // Both continuations have count 1 — lexicographic tie-break picks "apple".
    expect(backend.predictNext(model, ["x"]).token).toBe("apple");
  });

  it("generalizes to a new-but-related context via stupid-backoff (objective 2d)", () => {
    const backend = new DeterministicMarkovBackend();
    // "save" is always followed by "confirm", across several prefixes.
    const model = backend.train(
      dataset([
        ["edit", "save", "confirm"],
        ["rename", "save", "confirm"],
        ["resize", "save", "confirm"],
      ]),
      { order: 2 },
    );

    // Unseen full bigram context ("crop","save") — the order-2 key was never
    // observed, so it must back off to the order-1 context "save".
    const prediction = backend.predictNext(model, ["crop", "save"]);
    expect(prediction.token).toBe("confirm");
    expect(prediction.backoffOrder).toBe(1);
    expect(prediction.probability).toBeCloseTo(1);
  });

  it("falls back to the unigram prior for a wholly unseen context", () => {
    const backend = new DeterministicMarkovBackend();
    const model = backend.train(dataset([["a", "b", "b", "b", "c"]]), { order: 2 });
    const prediction = backend.predictNext(model, ["totally", "unknown"]);
    // Backoff all the way to order 0 (unigram) — "b" is the most common token.
    expect(prediction.backoffOrder).toBe(0);
    expect(prediction.token).toBe("b");
  });

  it("terminates generation with maxTokens even on a cyclic model", () => {
    const backend = new DeterministicMarkovBackend();
    // "a" -> "a" forever; there is no END transition after "a".
    const model = backend.train(dataset([["a", "a", "a", "a"]]), { order: 1 });
    const out = backend.generate(model, ["a"], { maxTokens: 5 });
    expect(out.length).toBeLessThanOrEqual(6); // seed + at most maxTokens
    expect(out.every((token) => token === "a")).toBe(true);
  });

  it("returns END on an empty model instead of throwing", () => {
    const backend = new DeterministicMarkovBackend();
    const model = backend.train(dataset([]), { order: 2 });
    const prediction = backend.predictNext(model, ["anything"]);
    expect(prediction.token).toBe(MOVEMENT_END_TOKEN);
    expect(backend.generate(model, [])).toEqual([]);
  });
});

describe("createMovementModelBackend", () => {
  it("resolves the deterministic backend by default", () => {
    expect(createMovementModelBackend().id).toBe("deterministic-markov");
    expect(createMovementModelBackend("deterministic-markov").id).toBe("deterministic-markov");
  });
});

describe("buildMovementDataset", () => {
  it("extracts ordered per-trajectory action sequences from replays", () => {
    const replay: ReplayManifest = {
      version: 1,
      sessionId: "s1",
      trajectoryIds: ["t1"],
      eventCount: 4,
      events: [
        { kind: "transcript", ts: 1, messageId: "m1", role: "user", content: "do the thing" },
        { kind: "observation", ts: 2, trajectoryId: "t1", source: "device", summary: "app active" },
        { kind: "action", ts: 5, trajectoryId: "t1", tool: "device", summary: "tapped Send" },
        { kind: "action", ts: 3, trajectoryId: "t1", tool: "device", summary: "typed Subject" },
      ],
    };

    const built = buildMovementDataset([replay]);
    expect(built.sequences).toHaveLength(1);
    // Actions only, sorted by timestamp (ts 3 before ts 5), observation/transcript dropped.
    expect(built.sequences[0]!.tokens).toEqual([
      "action:device:typed-subject",
      "action:device:tapped-send",
    ]);
  });

  it("keeps trajectories from different sessions/ids separate and skips empty ones", () => {
    const replays: ReplayManifest[] = [
      {
        version: 1,
        sessionId: "s1",
        trajectoryIds: ["t1"],
        eventCount: 1,
        events: [{ kind: "action", ts: 1, trajectoryId: "t1", tool: "device", summary: "a" }],
      },
      {
        version: 1,
        sessionId: "s2",
        trajectoryIds: ["t1", "t2"],
        eventCount: 2,
        events: [
          { kind: "action", ts: 1, trajectoryId: "t1", tool: "device", summary: "b" },
          { kind: "observation", ts: 2, trajectoryId: "t2", source: "device", summary: "noop" },
        ],
      },
    ];

    const built = buildMovementDataset(replays);
    // s1::t1 and s2::t1 are distinct; s2::t2 has no actions and is skipped.
    expect(built.sequences.map((sequence) => sequence.id)).toEqual(["s1::t1", "s2::t1"]);
  });

  it("feeds a full replay -> dataset -> train -> generate round-trip", () => {
    const replay: ReplayManifest = {
      version: 1,
      sessionId: "s1",
      trajectoryIds: ["t1"],
      eventCount: 3,
      events: [
        { kind: "action", ts: 1, trajectoryId: "t1", tool: "device", summary: "open app" },
        { kind: "action", ts: 2, trajectoryId: "t1", tool: "device", summary: "click compose" },
        { kind: "action", ts: 3, trajectoryId: "t1", tool: "device", summary: "click send" },
      ],
    };
    const built = buildMovementDataset([replay]);
    const backend = createMovementModelBackend();
    const model = backend.train(built);
    const expected = replay.events
      .filter((event): event is Extract<typeof event, { kind: "action" }> => event.kind === "action")
      .map(movementTokenFromAction);
    expect(backend.generate(model, [])).toEqual(expected);
  });
});
