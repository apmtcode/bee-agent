import { describe, expect, it } from "vitest";
import {
  MarkovMovementBackend,
  SEQUENCE_END,
  buildMovementDataset,
  movementTokenFromAction,
  trajectoryToSequence,
  type MovementDataset,
} from "./model-backend.js";
import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

function span(id: string, tools: string[]): TrajectorySpan {
  const actions: TrajectoryAction[] = tools.map((tool, index) => ({
    kind: "action",
    tool,
    summary: `${tool} step`,
    ts: index,
  }));
  return {
    id,
    sessionId: "s1",
    createdAt: "2026-07-03T00:00:00.000Z",
    captureTier: "operator",
    observations: [],
    actions,
  };
}

describe("MarkovMovementBackend training + inference", () => {
  const dataset: MovementDataset = {
    sequences: [
      { id: "a", tokens: ["focus-window", "move-right", "click", "open", "type"] },
      { id: "b", tokens: ["focus-window", "move-right", "click", "open", "type"] },
      { id: "c", tokens: ["scroll-down", "move-left", "click", "drag", "drop"] },
    ],
  };

  it("reproduces a recorded movement it was trained on", async () => {
    const model = await new MarkovMovementBackend({ order: 3 }).train({ dataset });
    // From a cold start it rolls out the majority-branch trajectory verbatim.
    const generated = model.generate({ maxTokens: 10 });
    expect(generated).toEqual(["focus-window", "move-right", "click", "open", "type"]);
    // Given a prompt, generation continues from it (the prompt is not echoed).
    const continuation = model.generate({ prompt: ["focus-window"], maxTokens: 10 });
    expect(continuation).toEqual(["move-right", "click", "open", "type"]);
  });

  it("exposes deterministic metadata and vocabulary", async () => {
    const model = await new MarkovMovementBackend({ order: 3 }).train({ dataset });
    expect(model.metadata.backend).toBe("markov");
    expect(model.metadata.order).toBe(3);
    expect(model.metadata.sequenceCount).toBe(3);
    // Vocabulary excludes the START sentinel but includes END.
    expect(model.vocabulary()).toContain(SEQUENCE_END);
    expect(model.vocabulary()).not.toContain("<s>");
    // Sorted for determinism.
    expect(model.vocabulary()).toEqual([...model.vocabulary()].sort());
  });

  it("is fully deterministic across repeated training runs", async () => {
    const backend = new MarkovMovementBackend({ order: 3 });
    const a = await backend.train({ dataset });
    const b = await backend.train({ dataset });
    const ctx = ["focus-window", "move-right"];
    expect(a.predictNext(ctx)).toEqual(b.predictNext(ctx));
    expect(a.generate({ maxTokens: 8 })).toEqual(b.generate({ maxTokens: 8 }));
  });

  it("predicts the dominant continuation from a learned context", async () => {
    const model = await new MarkovMovementBackend({ order: 2 }).train({ dataset });
    const prediction = model.predictNext(["click"]);
    expect(prediction?.token).toBe("open"); // 2 of 3 clicks -> open
    expect(prediction?.probability).toBeGreaterThan(0);
    expect(prediction?.contextOrder).toBe(1);
  });

  it("backs off to a shorter context for an unseen prefix (generalization)", async () => {
    const model = await new MarkovMovementBackend({ order: 3 }).train({ dataset });
    // "drag" was only ever preceded by "click" in the "scroll-down" branch, but
    // asking after a novel high-order context still yields a sensible drag->drop.
    const prediction = model.predictNext(["never-seen-token", "drag"]);
    expect(prediction?.token).toBe("drop");
    expect(prediction?.contextOrder).toBeLessThanOrEqual(1);
  });

  it("returns finite log-probabilities even for unseen tokens", async () => {
    const model = await new MarkovMovementBackend({ order: 3 }).train({ dataset });
    const logProb = model.sequenceLogProb(["focus-window", "totally-novel-move"]);
    expect(Number.isFinite(logProb)).toBe(true);
    expect(logProb).toBeLessThan(0);
  });

  it("assigns higher probability to a trained sequence than a scrambled one", async () => {
    const model = await new MarkovMovementBackend({ order: 3 }).train({ dataset });
    const trained = model.sequenceLogProb(["focus-window", "move-right", "click", "open", "type"]);
    const scrambled = model.sequenceLogProb(["type", "open", "focus-window", "click", "move-right"]);
    expect(trained).toBeGreaterThan(scrambled);
  });
});

describe("tokenization helpers", () => {
  it("derives tokens from tool plus a direction hint", () => {
    expect(
      movementTokenFromAction({ kind: "action", tool: "move", summary: "", ts: 0, metadata: { direction: "left" } }),
    ).toBe("move:left");
    expect(movementTokenFromAction({ kind: "action", tool: "click", summary: "", ts: 0 })).toBe("click");
  });

  it("builds a dataset from trajectory spans and drops empty ones", () => {
    const dataset = buildMovementDataset([
      span("t1", ["click", "drag", "drop"]),
      span("t2", []),
    ]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]).toEqual(trajectoryToSequence(span("t1", ["click", "drag", "drop"])));
  });
});
