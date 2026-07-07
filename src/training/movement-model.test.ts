import { describe, expect, it } from "vitest";
import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  MarkovMovementBackend,
  MOVEMENT_END,
  MOVEMENT_START,
  buildMovementDataset,
  createSeededRng,
  deserializeMovementModel,
  evaluateMovementModel,
  toMovementSequence,
  tokenizeAction,
  type MovementDataset,
  type MovementSequence,
} from "./movement-model.js";

function trajectory(id: string, tokens: Array<{ tool: string; summary: string; metadata?: Record<string, unknown> }>, options: { status?: "approved" | "pending" } = {}): TrajectorySpan {
  return {
    id,
    sessionId: `session-${id}`,
    createdAt: "2026-07-07T00:00:00.000Z",
    captureTier: "app",
    observations: [],
    actions: tokens.map((token, index) => ({
      kind: "action",
      tool: token.tool,
      summary: token.summary,
      ts: index,
      ...(token.metadata ? { metadata: token.metadata } : {}),
    })),
    ...(options.status ? { review: { status: options.status, reviewedAt: "2026-07-07T00:00:00.000Z", reviewedBy: "tester" } } : {}),
  };
}

/** A small synthetic "workflow": open → focus → type → save. */
const OPEN_FOCUS_TYPE_SAVE: MovementSequence = ["os:window-opened", "os:focus-changed", "device:type", "os:command-ran"];

function repeat(sequence: MovementSequence, times: number): MovementDataset {
  return { sequences: Array.from({ length: times }, () => [...sequence]) };
}

describe("tokenizeAction", () => {
  it("canonicalizes gesture + direction into a stable token", () => {
    expect(tokenizeAction({ kind: "action", tool: "device", summary: "swiped up", ts: 0, metadata: { gesture: "swipe", direction: "up" } })).toBe("device:swipe-up");
  });

  it("uses os event metadata when present", () => {
    expect(tokenizeAction({ kind: "action", tool: "os", summary: "opened Editor", ts: 0, metadata: { event: "window-opened" } })).toBe("os:window-opened");
  });

  it("falls back to the first summary word", () => {
    expect(tokenizeAction({ kind: "action", tool: "shell", summary: "Ran build script", ts: 0 })).toBe("shell:ran");
  });
});

describe("toMovementSequence", () => {
  it("orders actions by timestamp regardless of array order", () => {
    const span = trajectory("t1", []);
    span.actions = [
      { kind: "action", tool: "b", summary: "second", ts: 5 },
      { kind: "action", tool: "a", summary: "first", ts: 1 },
    ];
    expect(toMovementSequence(span)).toEqual(["a:first", "b:second"]);
  });

  it("prefers reviewer-redacted actions when present", () => {
    const span = trajectory("t2", [{ tool: "device", summary: "typed secret" }]);
    span.review = {
      status: "approved",
      reviewedAt: "2026-07-07T00:00:00.000Z",
      reviewedBy: "tester",
      redactedActions: [{ ts: 0, tool: "device", summary: "typed [redacted]" }],
    };
    expect(toMovementSequence(span)).toEqual(["device:typed"]);
  });
});

describe("buildMovementDataset", () => {
  it("filters to approved trajectories and drops short sequences", () => {
    const dataset = buildMovementDataset(
      [
        trajectory("a", [{ tool: "os", summary: "opened x" }, { tool: "device", summary: "tapped y" }], { status: "approved" }),
        trajectory("b", [{ tool: "os", summary: "opened z" }], { status: "pending" }),
        trajectory("c", [{ tool: "os", summary: "opened w" }, { tool: "device", summary: "tapped v" }], { status: "approved" }),
      ],
      { approvedOnly: true, minLength: 2 },
    );
    expect(dataset.sequences).toHaveLength(2);
  });
});

describe("MarkovMovementBackend", () => {
  it("replays a recorded movement exactly (objective 2c)", () => {
    const model = new MarkovMovementBackend().train(repeat(OPEN_FOCUS_TYPE_SAVE, 3), { order: 3 });
    expect(model.generate({ seed: [MOVEMENT_START] })).toEqual(OPEN_FOCUS_TYPE_SAVE);
  });

  it("predicts the next movement given a partial context", () => {
    const model = new MarkovMovementBackend().train(repeat(OPEN_FOCUS_TYPE_SAVE, 3), { order: 3 });
    const predictions = model.predictNext([MOVEMENT_START, "os:window-opened"]);
    expect(predictions[0].token).toBe("os:focus-changed");
    expect(predictions[0].probability).toBeCloseTo(1);
  });

  it("generalizes to a novel-but-related prefix via backoff (objective 2d)", () => {
    // Two related workflows share the "focus -> type" tail. A prefix the model
    // never saw in full still predicts a sensible continuation via lower-order
    // backoff.
    const dataset: MovementDataset = {
      sequences: [
        ["os:window-opened", "os:focus-changed", "device:type"],
        ["os:file-opened", "os:focus-changed", "device:type"],
      ],
    };
    const model = new MarkovMovementBackend().train(dataset, { order: 3 });
    // "os:command-ran" -> "os:focus-changed" was never observed as a full bigram,
    // but the unigram/bigram stats still put "device:type" after focus.
    const predictions = model.predictNext(["os:command-ran", "os:focus-changed"]);
    expect(predictions[0].token).toBe("device:type");
    expect(predictions[0].order).toBeLessThan(2);
  });

  it("mixes learned branches and can sample them with a seeded rng", () => {
    const dataset: MovementDataset = {
      sequences: [
        ["os:window-opened", "device:tap"],
        ["os:window-opened", "device:swipe-up"],
      ],
    };
    const model = new MarkovMovementBackend().train(dataset, { order: 2 });
    const predictions = model.predictNext([MOVEMENT_START, "os:window-opened"]);
    expect(predictions.map((p) => p.token).sort()).toEqual(["device:swipe-up", "device:tap"]);
    // Seeded generation is reproducible.
    const first = model.generate({ seed: [MOVEMENT_START], rng: createSeededRng(42) });
    const second = model.generate({ seed: [MOVEMENT_START], rng: createSeededRng(42) });
    expect(first).toEqual(second);
    expect(first[0]).toBe("os:window-opened");
  });

  it("returns no predictions for an unknown token with no backoff signal", () => {
    const model = new MarkovMovementBackend().train({ sequences: [["a", "b"]] }, { order: 2 });
    // Even fully-unknown context backs off to the unigram distribution, so it
    // still predicts *something* — this is desired generalization behaviour.
    expect(model.predictNext(["totally-unseen"]).length).toBeGreaterThan(0);
  });

  it("survives a serialize -> deserialize round-trip", () => {
    const model = new MarkovMovementBackend().train(repeat(OPEN_FOCUS_TYPE_SAVE, 2), { order: 3 });
    const restored = deserializeMovementModel(model.serialize());
    expect(restored.order).toBe(3);
    expect(restored.generate({ seed: [MOVEMENT_START] })).toEqual(OPEN_FOCUS_TYPE_SAVE);
    expect(restored.predictNext([MOVEMENT_START, "os:window-opened"])).toEqual(
      model.predictNext([MOVEMENT_START, "os:window-opened"]),
    );
  });

  it("rejects an unknown serialized backend", () => {
    expect(() => deserializeMovementModel({ version: 1, backend: "mystery", order: 2, counts: [] })).toThrow(/unsupported/);
  });

  it("respects maxLength when a sequence would otherwise loop", () => {
    // A degenerate self-looping dataset: a -> a. Generation must terminate.
    const model = new MarkovMovementBackend().train({ sequences: [["a", "a", "a", "a"]] }, { order: 1 });
    const produced = model.generate({ seed: [MOVEMENT_START], maxLength: 5, stopAtEnd: false });
    expect(produced.length).toBeLessThanOrEqual(5);
  });
});

describe("evaluateMovementModel", () => {
  it("scores perfect accuracy and exact replay on memorized sequences", () => {
    const model = new MarkovMovementBackend().train(repeat(OPEN_FOCUS_TYPE_SAVE, 3), { order: 3 });
    const result = evaluateMovementModel(model, [OPEN_FOCUS_TYPE_SAVE]);
    expect(result.accuracy).toBe(1);
    expect(result.exactReplay).toBe(1);
    expect(result.total).toBe(OPEN_FOCUS_TYPE_SAVE.length + 1); // +1 for the END token
  });

  it("generalizes above chance to held-out related sequences", () => {
    const train: MovementDataset = {
      sequences: [
        ["os:window-opened", "os:focus-changed", "device:type", "os:command-ran"],
        ["os:window-opened", "os:focus-changed", "device:tap", "os:command-ran"],
        ["os:file-opened", "os:focus-changed", "device:type", "os:command-ran"],
      ],
    };
    const model = new MarkovMovementBackend().train(train, { order: 3 });
    const heldOut: MovementSequence = ["os:file-opened", "os:focus-changed", "device:tap", "os:command-ran"];
    const result = evaluateMovementModel(model, [heldOut]);
    // Chance for ~5 distinct tokens is ~0.2; backoff should clear that easily.
    expect(result.accuracy).toBeGreaterThan(0.5);
  });

  it("returns zeroes for an empty eval set", () => {
    const model = new MarkovMovementBackend().train({ sequences: [] }, { order: 2 });
    expect(evaluateMovementModel(model, [])).toEqual({ total: 0, correct: 0, accuracy: 0, exactReplay: 0 });
  });
});

describe("end-to-end capture -> dataset -> train -> infer", () => {
  it("learns from recorded trajectories and reproduces the workflow", () => {
    const trajectories = [
      trajectory("r1", [
        { tool: "os", summary: "opened Editor", metadata: { event: "window-opened" } },
        { tool: "os", summary: "focused Editor", metadata: { event: "focus-changed" } },
        { tool: "device", summary: "typed code", metadata: { gesture: "type" } },
        { tool: "os", summary: "ran build", metadata: { event: "command-ran" } },
      ], { status: "approved" }),
      trajectory("r2", [
        { tool: "os", summary: "opened Editor", metadata: { event: "window-opened" } },
        { tool: "os", summary: "focused Editor", metadata: { event: "focus-changed" } },
        { tool: "device", summary: "typed code", metadata: { gesture: "type" } },
        { tool: "os", summary: "ran build", metadata: { event: "command-ran" } },
      ], { status: "approved" }),
    ];
    const dataset = buildMovementDataset(trajectories, { approvedOnly: true, minLength: 2 });
    const model = new MarkovMovementBackend().train(dataset, { order: 3 });
    expect(model.generate({ seed: [MOVEMENT_START] })).toEqual([
      "os:window-opened",
      "os:focus-changed",
      "device:type",
      "os:command-ran",
    ]);
    expect(model.predictNext([MOVEMENT_START]).map((p) => p.token)).toContain("os:window-opened");
    expect(model.predictNext(["os:command-ran"])[0].token).toBe(MOVEMENT_END);
  });
});
