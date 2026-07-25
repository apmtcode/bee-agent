import { describe, expect, it } from "vitest";
import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  DEFAULT_MOVEMENT_BACKEND,
  MOVEMENT_END,
  MarkovMovementBackend,
  buildMovementDataset,
  createMovementModelBackendRegistry,
  movementContextForTrajectory,
  tokenizeMovementAction,
  trainMovementModel,
  type MovementDataset,
} from "./movement-model.js";

function trajectory(overrides: Partial<TrajectorySpan> & Pick<TrajectorySpan, "id">): TrajectorySpan {
  return {
    sessionId: "session-1",
    createdAt: "2026-07-25T00:00:00.000Z",
    captureTier: "full",
    observations: [],
    actions: [],
    ...overrides,
  };
}

describe("tokenizeMovementAction", () => {
  it("prefers structured gesture metadata over free text", () => {
    const token = tokenizeMovementAction({
      kind: "action",
      tool: "device",
      summary: "swiped up on the home screen",
      ts: 1,
      metadata: { gesture: "swipe", direction: "up", target: "home screen" },
    });
    expect(token).toBe("device:swipe:up:home-screen");
  });

  it("falls back to a slug of the summary when no gesture metadata is present", () => {
    const token = tokenizeMovementAction({
      kind: "action",
      tool: "browser",
      summary: "Clicked Submit!",
      ts: 1,
    });
    expect(token).toBe("browser:clicked-submit");
  });
});

describe("buildMovementDataset", () => {
  it("orders actions by timestamp, derives context, and skips action-less trajectories", () => {
    const dataset = buildMovementDataset([
      trajectory({
        id: "t-1",
        observations: [
          { kind: "observation", source: "device", summary: "Notes active", ts: 0, metadata: { appName: "Notes" } },
        ],
        actions: [
          { kind: "action", tool: "device", summary: "b", ts: 20, metadata: { gesture: "type", target: "body" } },
          { kind: "action", tool: "device", summary: "a", ts: 10, metadata: { gesture: "tap", target: "note" } },
        ],
      }),
      trajectory({ id: "t-empty", actions: [] }),
    ]);

    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]!.context).toBe("notes");
    expect(dataset.sequences[0]!.tokens).toEqual(["device:tap:note", "device:type:body"]);
  });
});

describe("movementContextForTrajectory", () => {
  it("uses the observation source when no app name is available", () => {
    const context = movementContextForTrajectory(
      trajectory({
        id: "t-src",
        observations: [{ kind: "observation", source: "OS Window", summary: "focused", ts: 0 }],
      }),
    );
    expect(context).toBe("os-window");
  });
});

describe("MarkovMovementBackend", () => {
  it("exposes the default backend name via the registry", () => {
    const registry = createMovementModelBackendRegistry();
    expect(registry.has(DEFAULT_MOVEMENT_BACKEND)).toBe(true);
    expect(registry.get(DEFAULT_MOVEMENT_BACKEND)).toBeInstanceOf(MarkovMovementBackend);
  });

  it("replays a recorded single-context sequence verbatim (objective 2c)", () => {
    const dataset: MovementDataset = {
      version: 1,
      sequences: [{ id: "s1", context: "editor", tokens: ["open", "type", "select", "copy", "save"] }],
    };
    const model = trainMovementModel(dataset, { order: 3 });
    expect(model.generate({ context: "editor" })).toEqual(["open", "type", "select", "copy", "save"]);
  });

  it("predicts the next movement deterministically with confidence", () => {
    const dataset: MovementDataset = {
      version: 1,
      sequences: [{ id: "s1", context: "editor", tokens: ["open", "type", "save"] }],
    };
    const model = new MarkovMovementBackend().train(dataset, { order: 2 });
    const prediction = model.predictNext(["open"], { context: "editor" });
    expect(prediction.token).toBe("type");
    expect(prediction.confidence).toBeCloseTo(1);
    expect(prediction.contextMatched).toBe(true);
  });

  it("terminates generation with an END prediction rather than looping", () => {
    const dataset: MovementDataset = {
      version: 1,
      sequences: [{ id: "s1", context: "editor", tokens: ["open", "close"] }],
    };
    const model = trainMovementModel(dataset);
    const prediction = model.predictNext(["open", "close"], { context: "editor" });
    expect(prediction.token).toBe(MOVEMENT_END);
    // maxSteps guards against runaway rollouts even on a degenerate model.
    const shortRollout = model.generate({ context: "editor", maxSteps: 3 });
    expect(shortRollout.length).toBeLessThanOrEqual(3);
  });

  it("generalizes to a novel prefix via backoff to shared sub-movements (objective 2d)", () => {
    // Two recorded flows both lead into "select" but from different preceding
    // movements ("search" vs. "browse"). The pair (open, browse) never occurred
    // together in training.
    const dataset: MovementDataset = {
      version: 1,
      sequences: [
        { id: "a", context: "files", tokens: ["open", "search", "select", "copy"] },
        { id: "b", context: "files", tokens: ["focus", "browse", "select", "paste"] },
      ],
    };
    const model = trainMovementModel(dataset, { order: 2 });

    // The bigram (open, browse) was never seen, so the order-2 model misses and
    // the model backs off to the order-1 fact "browse -> select" learned from the
    // *other* flow — a movement it never recorded in this combination.
    const next = model.predictNext(["open", "browse"], { context: "files" });
    expect(next.token).toBe("select");
    expect(next.order).toBe(1);

    const rollout = model.generate({ context: "files", seed: ["open", "browse"] });
    expect(rollout.slice(0, 3)).toEqual(["open", "browse", "select"]);
    // It then continues along the learned continuation of the shared movement.
    expect(rollout[3]).toBe("paste");
  });

  it("backs off to the shared/global model when the context has no data", () => {
    const dataset: MovementDataset = {
      version: 1,
      sequences: [{ id: "a", context: "mail", tokens: ["compose", "send"] }],
    };
    const model = trainMovementModel(dataset, { order: 2 });
    // A brand-new context was never trained, but the global bigram carries over.
    const prediction = model.predictNext(["compose"], { context: "unseen-app" });
    expect(prediction.token).toBe("send");
    expect(prediction.contextMatched).toBe(false);
  });

  it("returns a null prediction for an empty model", () => {
    const model = trainMovementModel({ version: 1, sequences: [] });
    expect(model.predictNext([]).token).toBeNull();
    expect(model.generate()).toEqual([]);
  });

  it("throws for an unknown backend name", () => {
    expect(() => trainMovementModel({ version: 1, sequences: [] }, { backend: "nope" })).toThrow(/unknown movement model backend/);
  });

  it("is deterministic across repeated training runs", () => {
    const dataset: MovementDataset = {
      version: 1,
      sequences: [
        { id: "a", context: "app", tokens: ["x", "y", "z"] },
        { id: "b", context: "app", tokens: ["x", "y", "w"] },
      ],
    };
    const first = trainMovementModel(dataset).generate({ context: "app", seed: ["x", "y"] });
    const second = trainMovementModel(dataset).generate({ context: "app", seed: ["x", "y"] });
    expect(first).toEqual(second);
  });
});
