import { describe, expect, it } from "vitest";
import type { ReplayTimelineEvent } from "../capture/replay.js";
import {
  MOVEMENT_BOUNDARY_TOKEN,
  MovementModelRegistry,
  NgramMovementBackend,
  buildMovementSequence,
  createDefaultMovementModelRegistry,
  evaluateMovementReplayFidelity,
  rolloutMovements,
  synthesizeMovementSequences,
  tokenizeMovementEvent,
  type MovementSequence,
} from "./movement-model.js";

function actionEvent(tool: string, summary: string, ts: number): ReplayTimelineEvent {
  return { kind: "action", ts, trajectoryId: "t", tool, summary };
}

describe("tokenizeMovementEvent", () => {
  it("produces stable, discrete tokens per event kind", () => {
    expect(tokenizeMovementEvent(actionEvent("device", "swiped down", 1))).toBe("act:device:swiped-down");
    expect(
      tokenizeMovementEvent({ kind: "observation", ts: 1, trajectoryId: "t", source: "device", summary: "Mail active" }),
    ).toBe("obs:device:mail-active");
    expect(
      tokenizeMovementEvent({ kind: "transcript", ts: 1, messageId: "m", role: "user", content: "hi" }),
    ).toBe("msg:user");
  });

  it("normalizes punctuation/case so equivalent movements share a token", () => {
    expect(tokenizeMovementEvent(actionEvent("device", "Tapped  Compose!", 1))).toBe(
      tokenizeMovementEvent(actionEvent("device", "tapped compose", 2)),
    );
  });
});

describe("buildMovementSequence", () => {
  it("maps a replay timeline into an ordered token sequence", () => {
    const sequence = buildMovementSequence("traj-1", [
      actionEvent("device", "tapped compose", 1),
      actionEvent("device", "typed subject", 2),
      actionEvent("device", "tapped send", 3),
    ]);
    expect(sequence).toEqual({
      id: "traj-1",
      tokens: ["act:device:tapped-compose", "act:device:typed-subject", "act:device:tapped-send"],
    });
  });
});

describe("NgramMovementBackend", () => {
  const backend = new NgramMovementBackend();

  it("repeats a recorded movement given its prefix (objective 2c)", async () => {
    const recorded: MovementSequence = {
      id: "compose",
      tokens: ["open-mail", "tap-compose", "type-subject", "type-body", "tap-send"],
    };
    const model = await backend.train([recorded], { order: 3 });

    const continuation = rolloutMovements(backend, model, {
      seed: ["open-mail", "tap-compose"],
      maxLength: 10,
    });
    expect(continuation).toEqual(["type-subject", "type-body", "tap-send"]);
  });

  it("stops at the learned boundary token instead of looping forever", async () => {
    const model = await backend.train([{ id: "s", tokens: ["a", "b", "c"] }], { order: 2 });
    const rollout = rolloutMovements(backend, model, { maxLength: 50 });
    expect(rollout).toEqual(["a", "b", "c"]);
    expect(model.boundaryToken).toBe(MOVEMENT_BOUNDARY_TOKEN);
  });

  it("generalizes to a new-but-related movement via backoff (objective 2d)", async () => {
    // Two related flows that share the "tap-menu -> tap-share" motif but diverge
    // on the surrounding app. The model never saw "open-notes -> tap-menu", yet
    // backoff lets it complete the shared motif.
    const model = await backend.train(
      [
        { id: "mail", tokens: ["open-mail", "tap-menu", "tap-share", "tap-confirm"] },
        { id: "photos", tokens: ["open-photos", "tap-menu", "tap-share", "tap-confirm"] },
      ],
      { order: 3 },
    );

    const prediction = backend.predictNext(model, ["open-notes", "tap-menu"]);
    expect(prediction.token).toBe("tap-share");
    expect(prediction.backedOff).toBe(true);

    const continuation = rolloutMovements(backend, model, {
      seed: ["open-notes", "tap-menu"],
      maxLength: 6,
    });
    expect(continuation).toEqual(["tap-share", "tap-confirm"]);
  });

  it("prefers the higher-frequency continuation and breaks ties deterministically", async () => {
    const model = await backend.train(
      [
        { id: "a", tokens: ["ctx", "go-left"] },
        { id: "b", tokens: ["ctx", "go-left"] },
        { id: "c", tokens: ["ctx", "go-right"] },
      ],
      { order: 1, appendBoundaryToken: false },
    );
    const prediction = backend.predictNext(model, ["ctx"]);
    expect(prediction.token).toBe("go-left");
    expect(prediction.probability).toBeCloseTo(2 / 3, 5);
    expect(prediction.backedOff).toBe(false);
  });

  it("returns a null prediction for an empty model", async () => {
    const model = await backend.train([], { order: 2 });
    expect(backend.predictNext(model, ["anything"]).token).toBeNull();
    expect(rolloutMovements(backend, model, { maxLength: 5 })).toEqual([]);
  });

  it("produces a JSON-serializable model", async () => {
    const model = await backend.train([{ id: "s", tokens: ["a", "b"] }], { order: 2 });
    const roundTripped = JSON.parse(JSON.stringify(model));
    expect(roundTripped).toEqual(model);
    expect(roundTripped.vocabulary).toContain("a");
    expect(roundTripped.eventCount).toBe(3); // a, b, boundary
  });
});

describe("evaluateMovementReplayFidelity", () => {
  const backend = new NgramMovementBackend();

  it("reproduces the training set with perfect next-token fidelity", async () => {
    // Unambiguous flows (distinct entry points) so recorded replay is exact.
    const sequences = synthesizeMovementSequences({
      templates: [
        { id: "flow-a", tokens: ["open-a", "select", "confirm"] },
        { id: "flow-b", tokens: ["open-b", "cancel"] },
      ],
      repeats: 2,
    });
    const model = await backend.train(sequences, { order: 3 });
    const evaluation = evaluateMovementReplayFidelity(backend, model, sequences);
    expect(evaluation.accuracy).toBe(1);
    expect(evaluation.sequenceCount).toBe(4);
  });

  it("still predicts most tokens on held-out related sequences (generalization)", async () => {
    const train = synthesizeMovementSequences({
      templates: [
        { id: "mail", tokens: ["launch", "tap-menu", "tap-share", "done"] },
        { id: "photos", tokens: ["launch", "tap-menu", "tap-share", "done"] },
      ],
      repeats: 2,
    });
    const model = await backend.train(train, { order: 2 });

    const heldOut: MovementSequence[] = [{ id: "notes", tokens: ["launch", "tap-menu", "tap-share", "done"] }];
    const evaluation = evaluateMovementReplayFidelity(backend, model, heldOut);
    expect(evaluation.accuracy).toBeGreaterThan(0.5);
    expect(evaluation.predictions).toBeGreaterThan(0);
  });
});

describe("synthesizeMovementSequences", () => {
  it("expands templates deterministically with suffixed ids when repeated", () => {
    const sequences = synthesizeMovementSequences({
      templates: [{ id: "flow", tokens: ["a", "b"] }],
      repeats: 3,
    });
    expect(sequences.map((s) => s.id)).toEqual(["flow#0", "flow#1", "flow#2"]);
    expect(sequences.every((s) => s.tokens.length === 2)).toBe(true);
  });

  it("keeps the bare id when not repeated", () => {
    const sequences = synthesizeMovementSequences({ templates: [{ id: "flow", tokens: ["a"] }] });
    expect(sequences).toEqual([{ id: "flow", tokens: ["a"] }]);
  });
});

describe("MovementModelRegistry", () => {
  it("registers and resolves pluggable backends by name", () => {
    const registry = new MovementModelRegistry().register(new NgramMovementBackend());
    expect(registry.has("ngram")).toBe(true);
    expect(registry.get("ngram")).toBeInstanceOf(NgramMovementBackend);
    expect(registry.list()).toEqual(["ngram"]);
  });

  it("throws for an unknown backend", () => {
    const registry = new MovementModelRegistry();
    expect(() => registry.get("mlx")).toThrow(/Unknown movement-model backend/);
  });

  it("supports a custom drop-in backend implementing the same seam", async () => {
    const registry = createDefaultMovementModelRegistry();
    const custom = {
      name: "always-tap",
      async train() {
        return {
          version: 1 as const,
          backend: "always-tap",
          order: 1,
          boundaryToken: null,
          vocabulary: ["tap"],
          contexts: {},
          sequenceCount: 0,
          eventCount: 0,
        };
      },
      predictNext() {
        return { token: "tap", contextLength: 0, probability: 1, backedOff: false };
      },
    };
    registry.register(custom);
    const backend = registry.get("always-tap");
    const model = await backend.train([]);
    expect(rolloutMovements(backend, model, { maxLength: 3, stopAtBoundary: false })).toEqual(["tap", "tap", "tap"]);
  });
});
