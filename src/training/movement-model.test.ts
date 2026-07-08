import { describe, expect, it } from "vitest";
import {
  MarkovMovementModelBackend,
  buildMovementDatasetFromReplays,
  encodeMovementToken,
  tokenizeReplayEvents,
  trainMovementModel,
  type MovementToken,
} from "./movement-model.js";
import {
  generateSyntheticMovementFamily,
  generateSyntheticMovementStream,
} from "./synthetic-movement.js";

function keys(tokens: MovementToken[]): string[] {
  return tokens.map(encodeMovementToken);
}

describe("movement-model tokenization", () => {
  it("tokenizes replay events into ordered movement tokens", () => {
    const replay = generateSyntheticMovementStream({
      trajectoryId: "t1",
      app: "mail",
      steps: [
        { verb: "click", target: "compose" },
        { verb: "type", target: "body" },
        { verb: "click", target: "send" },
      ],
    });
    const tokens = tokenizeReplayEvents(replay.events);
    expect(keys(tokens)).toEqual([
      "observation os focused",
      "action device click",
      "action device type",
      "action device click",
    ]);
  });
});

describe("MarkovMovementModelBackend — repeat recorded movements (objective 2c)", () => {
  it("greedily reproduces a recorded movement sequence from scratch", async () => {
    const replay = generateSyntheticMovementStream({
      trajectoryId: "t1",
      app: "mail",
      steps: [
        { verb: "focus", target: "inbox" },
        { verb: "click", target: "compose" },
        { verb: "type", target: "body" },
        { verb: "click", target: "send" },
      ],
    });
    const dataset = buildMovementDatasetFromReplays([replay]);
    const model = await trainMovementModel(dataset, { order: 2 });

    // Rolling out from an empty seed should regenerate the recorded trajectory.
    const generated = model.rollout([], 20);
    expect(keys(generated)).toEqual(keys(dataset.sequences[0]!.tokens));
  });

  it("predicts the recorded continuation with a confident probability", async () => {
    const replay = generateSyntheticMovementStream({
      trajectoryId: "t1",
      app: "editor",
      steps: [
        { verb: "open", target: "file" },
        { verb: "type", target: "code" },
        { verb: "save" },
      ],
    });
    const model = await trainMovementModel(buildMovementDatasetFromReplays([replay]), { order: 2 });
    const prediction = model.predictNext([
      { kind: "observation", channel: "os", verb: "focused" },
      { kind: "action", channel: "device", verb: "open" },
    ]);
    expect(prediction?.token).toEqual({ kind: "action", channel: "device", verb: "type" });
    expect(prediction?.probability).toBe(1);
  });
});

describe("MarkovMovementModelBackend — generalize to related movements (objective 2d)", () => {
  it("uses backoff to continue an unseen lead-in that shares a known action suffix", async () => {
    // Recorded grammar: (login) → compose → body → send.
    const training = generateSyntheticMovementFamily({
      apps: ["mail", "notes"],
      steps: [{ verb: "login" }, { verb: "compose" }, { verb: "body" }, { verb: "send" }],
    });
    const model = await trainMovementModel(buildMovementDatasetFromReplays(training), { order: 2 });

    // Seed with a NEW but related movement: the user "reply"s instead of
    // "login"s, then composes. The 2-gram (reply→compose) was never recorded.
    const prediction = model.predictNext([
      { kind: "action", channel: "device", verb: "reply" },
      { kind: "action", channel: "device", verb: "compose" },
    ]);

    // Order-2 context is unseen, but backing off to the known `compose` suffix
    // still yields the correct related continuation (`body`).
    expect(prediction?.token).toEqual({ kind: "action", channel: "device", verb: "body" });
    expect(prediction?.backoffOrder).toBe(1);
  });

  it("falls back to a unigram prediction for a completely unseen context", async () => {
    const replay = generateSyntheticMovementStream({
      trajectoryId: "t1",
      app: "mail",
      steps: [{ verb: "click", target: "a" }, { verb: "click", target: "b" }, { verb: "type", target: "c" }],
    });
    const model = await trainMovementModel(buildMovementDatasetFromReplays([replay]), { order: 2 });
    const prediction = model.predictNext([{ kind: "action", channel: "unknown-tool", verb: "never-seen" }]);
    expect(prediction).toBeDefined();
    expect(prediction?.backoffOrder).toBe(0);
    // The most frequent token overall is the repeated `click`.
    expect(prediction?.token.verb).toBe("click");
  });
});

describe("MovementModelBackend pluggability + serialization", () => {
  it("exposes a stable backend name and a JSON-serializable snapshot", async () => {
    const backend = new MarkovMovementModelBackend();
    expect(backend.name).toBe("markov-backoff");
    const model = await backend.train(
      buildMovementDatasetFromReplays([
        generateSyntheticMovementStream({ trajectoryId: "t1", app: "mail", steps: [{ verb: "click" }] }),
      ]),
      { order: 1 },
    );
    const snapshot = model.snapshot();
    expect(snapshot.backend).toBe("markov-backoff");
    expect(snapshot.order).toBe(1);
    expect(() => JSON.stringify(snapshot)).not.toThrow();
    expect(snapshot.vocabulary.length).toBeGreaterThan(0);
  });

  it("is deterministic across repeated training runs on identical data", async () => {
    const family = generateSyntheticMovementFamily({
      apps: ["a", "b"],
      steps: [{ verb: "click" }, { verb: "type" }],
    });
    const dataset = buildMovementDatasetFromReplays(family);
    const first = await trainMovementModel(dataset, { order: 2 });
    const second = await trainMovementModel(dataset, { order: 2 });
    expect(first.rollout([], 10)).toEqual(second.rollout([], 10));
  });
});
