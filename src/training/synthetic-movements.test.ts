import { describe, expect, it } from "vitest";
import { tokenizeTrajectory } from "./movement-model.js";
import {
  DEFAULT_MOVEMENT_MOTIFS,
  generateSyntheticSequences,
  generateSyntheticTrajectories,
} from "./synthetic-movements.js";

describe("generateSyntheticTrajectories", () => {
  // `createdAt` is intentional wall-clock; compare the movement content only.
  const movementShape = (spans: ReturnType<typeof generateSyntheticTrajectories>) =>
    spans.map((span) => ({ id: span.id, actions: span.actions, outcome: span.outcome }));

  it("is deterministic for a given seed", () => {
    const a = generateSyntheticTrajectories({ seed: 42, count: 8 });
    const b = generateSyntheticTrajectories({ seed: 42, count: 8 });
    expect(movementShape(a)).toEqual(movementShape(b));
  });

  it("varies with the seed", () => {
    const a = generateSyntheticTrajectories({ seed: 1, count: 8 });
    const b = generateSyntheticTrajectories({ seed: 2, count: 8 });
    expect(movementShape(a)).not.toEqual(movementShape(b));
  });

  it("emits well-formed, time-ordered gesture actions", () => {
    const [span] = generateSyntheticTrajectories({ seed: 3, count: 1 });
    expect(span).toBeDefined();
    expect(span!.actions.length).toBeGreaterThan(0);
    const timestamps = span!.actions.map((a) => a.ts);
    expect([...timestamps].sort((x, y) => x - y)).toEqual(timestamps);
    for (const action of span!.actions) {
      expect(action.metadata?.gesture).toBeTypeOf("string");
    }
  });

  it("only draws from the provided motif library", () => {
    const spans = generateSyntheticTrajectories({
      seed: 5,
      count: 20,
      motifs: [DEFAULT_MOVEMENT_MOTIFS[0]!],
    });
    for (const span of spans) {
      expect(span.outcome?.summary).toContain(DEFAULT_MOVEMENT_MOTIFS[0]!.name);
    }
  });

  it("produces an empty stream when asked for zero trajectories", () => {
    expect(generateSyntheticTrajectories({ count: 0 })).toEqual([]);
  });
});

describe("generateSyntheticSequences", () => {
  it("matches tokenizeTrajectory over the same trajectories", () => {
    const options = { seed: 11, count: 6 } as const;
    const viaTrajectories = generateSyntheticTrajectories(options).map((span) => tokenizeTrajectory(span));
    const direct = generateSyntheticSequences(options);
    expect(direct).toEqual(viaTrajectories);
  });
});
