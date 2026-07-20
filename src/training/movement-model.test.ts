import { describe, expect, it } from "vitest";
import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  buildMovementDataset,
  movementActionToken,
  MOVEMENT_BOS,
  MOVEMENT_EOS,
  MovementBackendRegistry,
  slugifyMovement,
  tokenizeTrajectory,
} from "./movement-model.js";
import { MarkovMovementBackend, createDefaultMovementBackend } from "./markov-movement-backend.js";

function span(id: string, actions: Array<[string, string]>, reward?: number): TrajectorySpan {
  return {
    id,
    sessionId: `sess-${id}`,
    createdAt: "2026-01-01T00:00:00.000Z",
    captureTier: "operator",
    observations: [],
    actions: actions.map(([tool, summary], index) => ({
      kind: "action" as const,
      tool,
      summary,
      ts: index,
    })),
    ...(reward !== undefined
      ? { outcome: { status: "success" as const, summary: "ok", reward } }
      : {}),
  };
}

describe("movement tokenization", () => {
  it("slugifies and bounds free text deterministically", () => {
    expect(slugifyMovement("Open   File!!")).toBe("open-file");
    expect(slugifyMovement("  ---weird__Case--  ")).toBe("weird-case");
    expect(slugifyMovement("x".repeat(80)).length).toBeLessThanOrEqual(48);
  });

  it("builds stable action tokens from tool + summary", () => {
    expect(movementActionToken("mouse", "Click Save")).toBe("mouse:click-save");
    expect(movementActionToken("Keyboard", "")).toBe("keyboard");
  });

  it("prefers reviewed/redacted actions and orders by timestamp", () => {
    const s: TrajectorySpan = {
      ...span("t", [["a", "raw"]]),
      review: {
        status: "approved",
        reviewedAt: "2026-01-01T00:00:00.000Z",
        reviewedBy: "op",
        redactedActions: [
          { ts: 2, tool: "mouse", summary: "second" },
          { ts: 1, tool: "mouse", summary: "first" },
        ],
      },
    };
    expect(tokenizeTrajectory(s)).toEqual(["mouse:first", "mouse:second"]);
  });
});

describe("buildMovementDataset", () => {
  it("collects vocabulary, skips too-short trajectories, and weights by reward", () => {
    const dataset = buildMovementDataset(
      [
        span("t1", [["mouse", "move a"], ["mouse", "click b"]], 3),
        span("t2", []), // skipped: no actions
      ],
      { minTokens: 1 },
    );
    expect(dataset.samples).toHaveLength(1);
    expect(dataset.samples[0]!.weight).toBe(3);
    expect(dataset.vocabulary).toContain(MOVEMENT_BOS);
    expect(dataset.vocabulary).toContain(MOVEMENT_EOS);
    expect(dataset.vocabulary).toContain("mouse:move-a");
    // vocabulary is sorted + de-duplicated
    expect([...dataset.vocabulary]).toEqual([...dataset.vocabulary].sort());
  });
});

describe("MarkovMovementBackend", () => {
  it("replays a recorded movement sequence from a cold start (objective #2c)", async () => {
    const sequence: Array<[string, string]> = [
      ["mouse", "focus window"],
      ["keyboard", "type query"],
      ["mouse", "click search"],
      ["mouse", "select result"],
    ];
    const dataset = buildMovementDataset([span("t1", sequence)]);
    const model = await new MarkovMovementBackend().train(dataset, { order: 3 });

    const expected = sequence.map(([tool, summary]) => movementActionToken(tool, summary));
    expect(model.generate([], 10)).toEqual(expected);
  });

  it("is deterministic: same dataset -> identical policy", async () => {
    const dataset = buildMovementDataset([
      span("t1", [["mouse", "a"], ["mouse", "b"], ["mouse", "c"]]),
    ]);
    const a = await new MarkovMovementBackend().train(dataset, { order: 2 });
    const b = await new MarkovMovementBackend().train(dataset, { order: 2 });
    expect(a.serialize()).toEqual(b.serialize());
  });

  it("predicts the next action from a partial context", async () => {
    const dataset = buildMovementDataset([
      span("t1", [["mouse", "open menu"], ["mouse", "choose export"], ["keyboard", "confirm"]]),
    ]);
    const model = await new MarkovMovementBackend().train(dataset, { order: 2 });
    const prediction = model.predictNext([
      movementActionToken("mouse", "open menu"),
      movementActionToken("mouse", "choose export"),
    ]);
    expect(prediction.token).toBe(movementActionToken("keyboard", "confirm"));
    expect(prediction.probability).toBeGreaterThan(0);
    expect(prediction.backoffOrder).toBe(2);
  });

  it("generalizes to a new-but-related sequence via backoff (objective #2d)", async () => {
    // Two demonstrations sharing a common suffix behaviour: after "save" the
    // operator always "close"s.
    const save = movementActionToken("keyboard", "save");
    const close = movementActionToken("mouse", "close");

    const dataset = buildMovementDataset([
      span("t1", [["mouse", "open document"], ["keyboard", "save"], ["mouse", "close"]]),
      span("t2", [["mouse", "open document"], ["keyboard", "edit body"], ["keyboard", "save"], ["mouse", "close"]]),
    ]);
    const model = await new MarkovMovementBackend().train(dataset, { order: 2 });

    // The exact bigram (unseen-first-action, save) never appeared in training,
    // so the full-order context matches nothing; the model backs off to the
    // 1-gram context "save" and still predicts the learned "close".
    const prediction = model.predictNext(["mouse:some-novel-action", save]);
    expect(prediction.token).toBe(close);
    expect(prediction.backoffOrder).toBe(1);

    // A full rollout from a partial seed terminates cleanly with a close.
    const rollout = model.generate([save], 10);
    expect(rollout[rollout.length - 1]).toBe(close);
  });

  it("round-trips through serialize/load with identical behaviour", async () => {
    const dataset = buildMovementDataset([
      span("t1", [["mouse", "a"], ["mouse", "b"], ["mouse", "c"]]),
      span("t2", [["mouse", "a"], ["mouse", "b"], ["keyboard", "d"]]),
    ]);
    const backend = new MarkovMovementBackend();
    const model = await backend.train(dataset, { order: 2 });
    const restored = backend.load(JSON.parse(JSON.stringify(model.serialize())));

    expect(restored.serialize()).toEqual(model.serialize());
    expect(restored.generate([], 10)).toEqual(model.generate([], 10));
  });

  it("returns EOS with zero probability when the model is empty", async () => {
    const model = await new MarkovMovementBackend().train(buildMovementDataset([]));
    const prediction = model.predictNext(["anything"]);
    expect(prediction.token).toBe(MOVEMENT_EOS);
    expect(prediction.probability).toBe(0);
    expect(model.generate([], 5)).toEqual([]);
  });
});

describe("MovementBackendRegistry", () => {
  it("registers, resolves, and reports backends", () => {
    const registry = new MovementBackendRegistry().register(createDefaultMovementBackend());
    expect(registry.has("markov-backoff")).toBe(true);
    expect(registry.list()).toEqual(["markov-backoff"]);
    expect(registry.get("markov-backoff").id).toBe("markov-backoff");
    expect(() => registry.get("nope")).toThrow(/unknown movement backend/);
  });
});
