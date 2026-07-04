import { describe, expect, it } from "vitest";
import type { ReplayTimelineEvent } from "../capture/replay.js";
import {
  DEFAULT_MOVEMENT_MODEL_ORDER,
  NGramMovementBackend,
  buildMovementDataset,
  encodeMovementEvent,
  evaluateMovementModel,
  loadMovementModel,
  tokenizeReplayEvents,
  type MovementSequence,
} from "./movement-model.js";

/**
 * Deterministic synthetic movement stream: a "form fill" workflow. Each step is
 * an observation followed by the action taken on it, so tokenization yields a
 * predictable observation/action alternation with no real OS input required.
 */
function syntheticReplay(
  id: string,
  steps: Array<{ field: string; value: string }>,
  screen = id,
): {
  id: string;
  events: ReplayTimelineEvent[];
} {
  const events: ReplayTimelineEvent[] = [];
  let ts = 0;
  // A distinct leading "screen" observation makes each workflow start from a
  // unique token, so a deterministic argmax rollout can reproduce each exactly.
  events.push({ kind: "observation", ts: ts++, trajectoryId: id, source: "screen", summary: `open ${screen}` });
  for (const step of steps) {
    events.push({ kind: "observation", ts: ts++, trajectoryId: id, source: "ui", summary: `field ${step.field}` });
    events.push({ kind: "action", ts: ts++, trajectoryId: id, tool: "device", summary: `type ${step.value}` });
  }
  return { id, events };
}

describe("movement tokenization", () => {
  it("encodes each replay event kind into a stable namespaced token", () => {
    expect(encodeMovementEvent({ kind: "action", ts: 1, trajectoryId: "t", tool: "device", summary: "Tap OK" })).toBe(
      "act:device:tap ok",
    );
    expect(
      encodeMovementEvent({ kind: "observation", ts: 1, trajectoryId: "t", source: "ui", summary: "Field  Name" }),
    ).toBe("obs:ui:field name");
    expect(
      encodeMovementEvent({ kind: "transcript", ts: 1, messageId: "m", role: "assistant", content: "hi" }),
    ).toBe("msg:assistant");
  });

  it("orders events by timestamp and includes observations + actions by default", () => {
    const sequence = tokenizeReplayEvents("t", [
      { kind: "action", ts: 2, trajectoryId: "t", tool: "device", summary: "submit" },
      { kind: "observation", ts: 1, trajectoryId: "t", source: "ui", summary: "form" },
    ]);
    expect(sequence.tokens).toEqual(["obs:ui:form", "act:device:submit"]);
  });

  it("skips transcript events unless explicitly requested", () => {
    const events: ReplayTimelineEvent[] = [
      { kind: "transcript", ts: 0, messageId: "m", role: "user", content: "go" },
      { kind: "action", ts: 1, trajectoryId: "t", tool: "device", summary: "go" },
    ];
    expect(tokenizeReplayEvents("t", events).tokens).toEqual(["act:device:go"]);
    expect(tokenizeReplayEvents("t", events, { includeTranscript: true }).tokens).toEqual([
      "msg:user",
      "act:device:go",
    ]);
  });

  it("drops empty sequences when building a dataset", () => {
    const dataset = buildMovementDataset([
      syntheticReplay("a", [{ field: "name", value: "ada" }]),
      { id: "empty", events: [] },
    ]);
    expect(dataset.sequences.map((s) => s.id)).toEqual(["a"]);
  });
});

describe("NGramMovementBackend — repeat", () => {
  it("reproduces a recorded movement sequence exactly from its first token", async () => {
    const replay = syntheticReplay("login", [
      { field: "user", value: "ada" },
      { field: "pass", value: "secret" },
      { field: "submit", value: "go" },
    ]);
    const dataset = buildMovementDataset([replay]);
    const model = await new NGramMovementBackend().train(dataset);
    const target = dataset.sequences[0]!.tokens;

    const rollout = model.generate([target[0]!]);

    expect(rollout).toEqual(target);
    expect(model.order).toBe(DEFAULT_MOVEMENT_MODEL_ORDER);
    expect(model.vocabularySize).toBe(new Set(target).size);
  });

  it("stops at the learned end-of-sequence sentinel rather than looping", async () => {
    const replay = syntheticReplay("loop", [
      { field: "a", value: "x" },
      { field: "a", value: "x" },
    ]);
    const model = await new NGramMovementBackend().train(buildMovementDataset([replay]), { order: 2 });
    const rollout = model.generate(["obs:ui:field a"], { maxSteps: 50 });
    // With learnStopToken the rollout terminates instead of cycling forever.
    expect(rollout.length).toBeLessThan(50);
    expect(rollout[0]).toBe("obs:ui:field a");
  });
});

describe("NGramMovementBackend — generalize", () => {
  it("generalizes an unseen context to a learned transition via backoff", async () => {
    // Both recorded runs share the transition "review -> submit", but reached
    // "review" from different prior movements. A novel prefix ending in "review"
    // (never seen at full order) should still predict "submit" by backing off to
    // the learned lower-order regularity.
    const dataset: { version: 1; sequences: MovementSequence[] } = {
      version: 1,
      sequences: [
        { id: "a", tokens: ["open:cart", "act:review", "act:submit"] },
        { id: "b", tokens: ["open:wishlist", "act:review", "act:submit"] },
      ],
    };
    const model = await new NGramMovementBackend().train(dataset);

    const prediction = model.predict(["open:brand-new-screen", "act:review"]);
    // Full-order context is unseen; the bigram "review -> submit" carries it.
    expect(prediction.token).toBe("act:submit");
    expect(prediction.backoffOrder).toBe(1);
    expect(prediction.confidence).toBe(1);
  });

  it("is deterministic: identical datasets yield byte-identical snapshots", async () => {
    const replays = [
      syntheticReplay("a", [{ field: "x", value: "1" }]),
      syntheticReplay("b", [{ field: "y", value: "2" }]),
    ];
    const backend = new NGramMovementBackend();
    const first = (await backend.train(buildMovementDataset(replays))).toSnapshot();
    const second = (await backend.train(buildMovementDataset(replays))).toSnapshot();
    expect(JSON.stringify(first)).toBe(JSON.stringify(second));
  });
});

describe("movement model serialization", () => {
  it("round-trips a snapshot into an identical runnable model", async () => {
    const dataset = buildMovementDataset([
      syntheticReplay("s", [
        { field: "user", value: "ada" },
        { field: "pass", value: "secret" },
      ]),
    ]);
    const model = await new NGramMovementBackend().train(dataset);
    const restored = loadMovementModel(model.toSnapshot());

    const seed = [dataset.sequences[0]!.tokens[0]!];
    expect(restored.generate(seed)).toEqual(model.generate(seed));
    expect(restored.toSnapshot()).toEqual(model.toSnapshot());
    expect(restored.vocabularySize).toBe(model.vocabularySize);
  });
});

describe("evaluateMovementModel", () => {
  it("reports perfect recall on the training set and measures held-out generalization", async () => {
    const training = [
      syntheticReplay("t1", [
        { field: "user", value: "ada" },
        { field: "pass", value: "secret" },
        { field: "submit", value: "go" },
      ]),
      syntheticReplay("t2", [
        { field: "user", value: "lin" },
        { field: "pass", value: "hunter2" },
        { field: "submit", value: "go" },
      ]),
    ];
    const dataset = buildMovementDataset(training);
    const model = await new NGramMovementBackend().train(dataset);

    const trainEval = evaluateMovementModel(model, dataset.sequences);
    expect(trainEval.sequences).toBe(2);
    expect(trainEval.exactMatchRate).toBe(1);
    expect(trainEval.rolloutFidelity).toBe(1);
    expect(trainEval.nextTokenAccuracy).toBe(1);

    // Held-out related sequence: same structure, novel values. The model should
    // still get the observation->action regularity right some of the time.
    const heldOut: MovementSequence[] = buildMovementDataset([
      syntheticReplay("h1", [
        { field: "user", value: "zed" },
        { field: "submit", value: "go" },
      ]),
    ]).sequences;
    const heldEval = evaluateMovementModel(model, heldOut);
    expect(heldEval.sequences).toBe(1);
    expect(heldEval.nextTokenAccuracy).toBeGreaterThan(0);
    expect(heldEval.rolloutFidelity).toBeGreaterThanOrEqual(0);
  });

  it("ignores sequences too short to score", () => {
    const model = new NGramMovementBackend();
    return model.train(buildMovementDataset([syntheticReplay("a", [{ field: "x", value: "1" }])])).then((trained) => {
      const evaluation = evaluateMovementModel(trained, [{ id: "tiny", tokens: ["only-one"] }]);
      expect(evaluation.sequences).toBe(0);
      expect(evaluation.nextTokenAccuracy).toBe(0);
    });
  });
});
