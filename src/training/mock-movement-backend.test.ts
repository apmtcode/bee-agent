import { describe, expect, it } from "vitest";
import {
  MovementModelRegistry,
  movementEventToken,
  type MovementContext,
  type MovementEvent,
} from "./movement-model.js";
import { MOCK_MOVEMENT_BACKEND_ID, NgramMovementBackend } from "./mock-movement-backend.js";
import {
  movementsMatch,
  synthesizeDataset,
  synthesizeTrajectory,
  type SyntheticTaskSpec,
} from "./movement-synthetic.js";

const editorContext: MovementContext = { goal: "save-file", appId: "editor", platform: "macos" };

function saveFileSpec(id = "t1"): SyntheticTaskSpec {
  return {
    id,
    context: editorContext,
    targets: ["file-menu", "save-item"],
    typeValue: "notes.md",
  };
}

describe("NgramMovementBackend", () => {
  it("replays a learned trajectory faithfully (piece c: train + repeat)", async () => {
    const backend = new NgramMovementBackend();
    const trajectory = synthesizeTrajectory(saveFileSpec());
    const model = await backend.train({ trajectories: [trajectory] });

    const replay = model.rollout(editorContext);

    expect(replay.length).toBe(trajectory.events.length);
    expect(movementsMatch(replay, trajectory.events)).toBe(true);
    // Reconstructed timestamps are strictly increasing.
    for (let i = 1; i < replay.length; i += 1) {
      expect(replay[i]!.ts).toBeGreaterThan(replay[i - 1]!.ts);
    }
  });

  it("is deterministic: identical datasets train to identical rollouts", async () => {
    const backend = new NgramMovementBackend();
    const dataset = synthesizeDataset([saveFileSpec("a"), saveFileSpec("b")]);

    const first = await backend.train(dataset);
    const second = await backend.train(dataset);

    expect(second.rollout(editorContext)).toEqual(first.rollout(editorContext));
  });

  it("generalizes to a new-but-related goal in the same app (piece d)", async () => {
    // Train on two editor tasks that share the file-menu target.
    const backend = new NgramMovementBackend();
    const dataset = synthesizeDataset([
      { id: "save", context: editorContext, targets: ["file-menu", "save-item"], typeValue: "a.md" },
      {
        id: "open",
        context: { ...editorContext, goal: "open-file" },
        targets: ["file-menu", "open-item"],
        typeValue: "b.md",
      },
    ]);
    const model = await backend.train(dataset);

    // Unseen goal in the SAME app: no exact-context match exists, so the model
    // must back off to app-level knowledge and still produce coherent movement.
    const unseen: MovementContext = { goal: "print-file", appId: "editor", platform: "macos" };
    const prediction = model.predictNext(unseen, []);

    expect(prediction.source).toBe("app");
    expect(prediction.event).toBeDefined();
    // The generalized first move reuses a real, learned event token.
    const rollout = model.rollout(unseen);
    expect(rollout.length).toBeGreaterThan(0);
    const knownTokens = new Set([
      ...synthesizeTrajectory({ id: "x", context: editorContext, targets: ["file-menu", "save-item"], typeValue: "a.md" }).events.map(movementEventToken),
      ...synthesizeTrajectory({ id: "y", context: { ...editorContext, goal: "open-file" }, targets: ["file-menu", "open-item"], typeValue: "b.md" }).events.map(movementEventToken),
    ]);
    for (const event of rollout) {
      expect(knownTokens.has(movementEventToken(event))).toBe(true);
    }
  });

  it("predicts the end of a sequence rather than looping forever", async () => {
    const backend = new NgramMovementBackend();
    const trajectory = synthesizeTrajectory(saveFileSpec());
    const model = await backend.train({ trajectories: [trajectory] });

    const full = model.rollout(editorContext, { maxSteps: 500 });
    const atEnd = model.predictNext(editorContext, full);

    expect(atEnd.end).toBe(true);
    expect(full.length).toBe(trajectory.events.length);
  });

  it("round-trips through serialization with identical behavior", async () => {
    const backend = new NgramMovementBackend();
    const dataset = synthesizeDataset([saveFileSpec("a"), saveFileSpec("b")]);
    const model = await backend.train(dataset);

    const serialized = model.serialize();
    const restored = backend.load(serialized);

    expect(serialized.backendId).toBe(MOCK_MOVEMENT_BACKEND_ID);
    expect(restored.rollout(editorContext)).toEqual(model.rollout(editorContext));
    expect(JSON.stringify(restored.serialize())).toBe(JSON.stringify(serialized));
  });

  it("returns an empty rollout for an unknown context with no backoff data", async () => {
    const backend = new NgramMovementBackend();
    const model = await backend.train({ trajectories: [] });

    expect(model.rollout({ goal: "anything", appId: "nowhere" })).toEqual([]);
    expect(model.predictNext({ goal: "anything" }, []).end).toBe(true);
  });
});

describe("MovementModelRegistry", () => {
  it("registers, resolves, and rehydrates a pluggable backend", async () => {
    const registry = new MovementModelRegistry();
    registry.register(new NgramMovementBackend());

    expect(registry.has(MOCK_MOVEMENT_BACKEND_ID)).toBe(true);
    expect(registry.list()).toContain(MOCK_MOVEMENT_BACKEND_ID);

    const model = await registry
      .get(MOCK_MOVEMENT_BACKEND_ID)
      .train({ trajectories: [synthesizeTrajectory(saveFileSpec())] });
    const restored = registry.load(model.serialize());

    expect(restored.backendId).toBe(MOCK_MOVEMENT_BACKEND_ID);
    expect(restored.rollout(editorContext)).toEqual(model.rollout(editorContext));
  });

  it("rejects duplicate backend registration and unknown lookups", () => {
    const registry = new MovementModelRegistry();
    registry.register(new NgramMovementBackend());
    expect(() => registry.register(new NgramMovementBackend())).toThrow(/already registered/);
    expect(() => registry.get("nope")).toThrow(/unknown movement model backend/);
  });

  it("tokenizes events stably and buckets nearby coordinates together", () => {
    const near1: MovementEvent = { ts: 0, actor: "mouse", action: "click", target: "ok", x: 100, y: 100 };
    const near2: MovementEvent = { ts: 5, actor: "mouse", action: "click", target: "ok", x: 110, y: 108 };
    const far: MovementEvent = { ts: 9, actor: "mouse", action: "click", target: "ok", x: 400, y: 400 };
    expect(movementEventToken(near1)).toBe(movementEventToken(near2));
    expect(movementEventToken(near1)).not.toBe(movementEventToken(far));
  });
});
