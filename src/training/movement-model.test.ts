import { describe, expect, it } from "vitest";
import type { ReplayTimelineEvent } from "../capture/replay.js";
import type { ReviewedExportManifest } from "./export-manifest.js";
import {
  NGramMovementBackend,
  actionToken,
  buildSyntheticMovementDataset,
  datasetFromReviewedExport,
  evaluateMovementModel,
  observationToken,
  parseMovementTokenKey,
  replayEventsToMovementTokens,
  serializeMovementToken,
  type MovementProgram,
} from "./movement-model.js";

const OPEN_FLOW: MovementProgram = {
  name: "open-and-submit",
  tokens: [
    actionToken("window.open"),
    actionToken("pointer.move"),
    actionToken("pointer.click"),
    actionToken("key.type"),
    actionToken("key.submit"),
  ],
};

const FOCUS_FLOW: MovementProgram = {
  name: "focus-and-submit",
  tokens: [
    actionToken("window.focus"),
    actionToken("pointer.move"),
    actionToken("pointer.click"),
    actionToken("key.type"),
    actionToken("key.submit"),
  ],
};

describe("movement token codec", () => {
  it("round-trips action and observation tokens through the key form", () => {
    const action = actionToken("pointer.move");
    const observation = observationToken("os.window");
    expect(serializeMovementToken(action)).toBe("action:pointer.move");
    expect(serializeMovementToken(observation)).toBe("observation:os.window");
    expect(parseMovementTokenKey("action:pointer.move")).toEqual(action);
    expect(parseMovementTokenKey("observation:os.window")).toEqual(observation);
  });

  it("preserves labels that themselves contain a colon", () => {
    const token = actionToken("http:post");
    expect(parseMovementTokenKey(serializeMovementToken(token))).toEqual(token);
  });

  it("maps the end sentinel to null", () => {
    expect(parseMovementTokenKey("<end>")).toBeNull();
  });
});

describe("NGramMovementBackend reproduction", () => {
  it("regenerates a recorded movement sequence from its first token", () => {
    const backend = new NGramMovementBackend();
    const dataset = buildSyntheticMovementDataset([OPEN_FLOW], { repeats: 3 });
    const model = backend.train(dataset, { order: 3 });

    const [first, ...rest] = OPEN_FLOW.tokens;
    const continuation = backend.generate(model, [first]);
    expect(continuation).toEqual(rest);
  });

  it("scores perfect replay fidelity on an unambiguous training flow", () => {
    const backend = new NGramMovementBackend();
    const dataset = buildSyntheticMovementDataset([OPEN_FLOW], { repeats: 2 });
    const model = backend.train(dataset);
    const evaluation = evaluateMovementModel(backend, model, dataset.sequences);
    expect(evaluation.accuracy).toBe(1);
    expect(evaluation.total).toBeGreaterThan(0);
  });

  it("cannot disambiguate a shared-tail start token, but nails the rest", () => {
    // OPEN_FLOW and FOCUS_FLOW differ only in their first token, so the
    // unconditional start prediction must miss for one of them — every other
    // (context-carrying) prediction is exact.
    const backend = new NGramMovementBackend();
    const dataset = buildSyntheticMovementDataset([OPEN_FLOW, FOCUS_FLOW], { repeats: 2 });
    const model = backend.train(dataset);
    const evaluation = evaluateMovementModel(backend, model, dataset.sequences);
    const perSequence = OPEN_FLOW.tokens.length + 1; // + end token
    // Exactly one start-token miss across the two distinct flows.
    expect(evaluation.correct).toBe(evaluation.total - 2);
    expect(evaluation.total).toBe(perSequence * dataset.sequences.length);
  });

  it("terminates generation at end-of-sequence without a prompt", () => {
    const backend = new NGramMovementBackend();
    const dataset = buildSyntheticMovementDataset([OPEN_FLOW], { repeats: 2 });
    const model = backend.train(dataset);
    const generated = backend.generate(model, [], { maxSteps: 32 });
    expect(generated).toEqual(OPEN_FLOW.tokens);
  });
});

describe("NGramMovementBackend generalization", () => {
  it("predicts a plausible next movement for a novel-but-related prefix via backoff", () => {
    const backend = new NGramMovementBackend();
    // Both training flows share the tail move -> click -> type -> submit.
    const dataset = buildSyntheticMovementDataset([OPEN_FLOW, FOCUS_FLOW], { repeats: 2 });
    const model = backend.train(dataset, { order: 3 });

    // A prefix never seen verbatim (hover start) but ending in a familiar
    // 2-gram context: the model should generalize to the shared continuation.
    const novelPrefix = [
      actionToken("pointer.hover"),
      actionToken("pointer.move"),
      actionToken("pointer.click"),
    ];
    const prediction = backend.predictNext(model, novelPrefix);
    expect(prediction.token).toEqual(actionToken("key.type"));
    expect(prediction.backoffOrder).toBeGreaterThan(0);
    expect(prediction.probability).toBeGreaterThan(0);
  });

  it("backs off to the unconditional prior for an entirely unknown context", () => {
    const backend = new NGramMovementBackend();
    const dataset = buildSyntheticMovementDataset([OPEN_FLOW], { repeats: 1 });
    const model = backend.train(dataset);
    const prediction = backend.predictNext(model, [actionToken("totally.unknown")]);
    // Falls back to the most common sequence-start token.
    expect(prediction.backoffOrder).toBe(0);
    expect(prediction.token).toEqual(actionToken("window.open"));
  });
});

describe("NGramMovementBackend persistence", () => {
  it("serializes and deserializes to an identical model", () => {
    const backend = new NGramMovementBackend();
    const dataset = buildSyntheticMovementDataset([OPEN_FLOW, FOCUS_FLOW], { repeats: 2 });
    const model = backend.train(dataset);
    const restored = backend.deserialize(backend.serialize(model));
    expect(restored).toEqual(model);

    const prefix = OPEN_FLOW.tokens.slice(0, 2);
    expect(backend.predictNext(restored, prefix)).toEqual(backend.predictNext(model, prefix));
  });

  it("rejects a serialized model from a different backend", () => {
    const backend = new NGramMovementBackend();
    expect(() => backend.deserialize(JSON.stringify({ backend: "mlx" }))).toThrow();
  });
});

describe("replay + export dataset adapters", () => {
  it("turns replay timeline events into movement tokens, dropping transcript", () => {
    const events: ReplayTimelineEvent[] = [
      { kind: "transcript", ts: 1, messageId: "m1", role: "user", content: "go" },
      { kind: "observation", ts: 2, trajectoryId: "t1", source: "os.window", summary: "app" },
      { kind: "action", ts: 3, trajectoryId: "t1", tool: "pointer.click", summary: "click" },
    ];
    expect(replayEventsToMovementTokens(events)).toEqual([
      observationToken("os.window"),
      actionToken("pointer.click"),
    ]);
    expect(replayEventsToMovementTokens(events, { includeObservations: false })).toEqual([
      actionToken("pointer.click"),
    ]);
  });

  it("builds a trainable dataset from a reviewed export manifest", () => {
    const manifest = {
      version: 1,
      replays: [
        {
          sessionId: "s1",
          trajectoryIds: ["t1"],
          eventCount: 2,
          events: [
            { kind: "action", ts: 1, trajectoryId: "t1", tool: "pointer.move", summary: "" },
            { kind: "action", ts: 2, trajectoryId: "t1", tool: "pointer.click", summary: "" },
          ],
        },
        { sessionId: "s2", trajectoryIds: ["t2"], eventCount: 0, events: [] },
      ],
    } as unknown as ReviewedExportManifest;

    const dataset = datasetFromReviewedExport(manifest);
    // Empty replays are dropped.
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]).toEqual({
      id: "t1",
      tokens: [actionToken("pointer.move"), actionToken("pointer.click")],
    });

    const backend = new NGramMovementBackend();
    const model = backend.train(dataset);
    expect(backend.generate(model, [actionToken("pointer.move")])).toEqual([
      actionToken("pointer.click"),
    ]);
  });
});
