import { describe, expect, it } from "vitest";
import {
  createSeededRandom,
  generateSyntheticDataset,
  movementSequenceFromTrajectory,
  tokenizeEvent,
  tokenizeSequence,
  type MovementEvent,
} from "./event.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

describe("movement/event", () => {
  it("tokenizes events into discrete, feature-bearing tokens", () => {
    expect(tokenizeEvent({ t: 0, type: "click", button: "left", target: "submit-button" })).toBe(
      "click:left:@submit-button",
    );
    expect(tokenizeEvent({ t: 0, type: "scroll", dy: 120 })).toBe("scroll:down");
    expect(tokenizeEvent({ t: 0, type: "key", key: "Enter" })).toBe("key:Enter");
  });

  it("quantizes untargeted pointer moves to a coarse grid for generalization", () => {
    const a = tokenizeEvent({ t: 0, type: "move", x: 401, y: 199 }, { gridSize: 64 });
    const b = tokenizeEvent({ t: 0, type: "move", x: 410, y: 210 }, { gridSize: 64 });
    // Nearby moves collapse to the same grid cell token.
    expect(a).toBe(b);
    expect(a).toBe("move:#6,3");
  });

  it("derives direction from deltas by dominant axis", () => {
    expect(tokenizeEvent({ t: 0, type: "move", dx: 10, dy: 2 })).toBe("move:right");
    expect(tokenizeEvent({ t: 0, type: "move", dx: -1, dy: -8 })).toBe("move:up");
  });

  it("is deterministic for a fixed seed", () => {
    const a = generateSyntheticDataset({ seed: 42, count: 5 });
    const b = generateSyntheticDataset({ seed: 42, count: 5 });
    expect(a).toEqual(b);
    const c = generateSyntheticDataset({ seed: 43, count: 5 });
    expect(c).not.toEqual(a);
  });

  it("generates the requested number of sequences with non-empty events", () => {
    const dataset = generateSyntheticDataset({ seed: 7, count: 9 });
    expect(dataset.sequences).toHaveLength(9);
    for (const sequence of dataset.sequences) {
      expect(sequence.events.length).toBeGreaterThan(0);
      expect(sequence.label).toBeTruthy();
    }
  });

  it("seeded PRNG stays within [0,1) and is reproducible", () => {
    const rng = createSeededRandom(123);
    const values = [rng(), rng(), rng()];
    for (const value of values) {
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThan(1);
    }
    const rng2 = createSeededRandom(123);
    expect([rng2(), rng2(), rng2()]).toEqual(values);
  });

  it("converts a trajectory's device actions into a relative-time sequence", () => {
    const trajectory: TrajectorySpan = {
      id: "traj-1",
      sessionId: "s-1",
      createdAt: "2026-01-01T00:00:00.000Z",
      captureTier: "app",
      observations: [],
      actions: [
        {
          kind: "action",
          tool: "device",
          summary: "tapped submit",
          ts: 1000,
          metadata: { gesture: "tap", target: "submit-button" },
        },
        {
          kind: "action",
          tool: "device",
          summary: "scrolled down",
          ts: 1600,
          metadata: { gesture: "scroll", direction: "down" },
        },
      ],
    };

    const sequence = movementSequenceFromTrajectory(trajectory);
    expect(sequence.id).toBe("traj-1");
    expect(sequence.events).toHaveLength(2);
    expect(sequence.events[0]).toMatchObject({ t: 0, type: "click", target: "submit-button" });
    expect(sequence.events[1]).toMatchObject({ t: 600, type: "scroll", dy: 1 });
    expect(tokenizeSequence(sequence)).toEqual(["click:@submit-button", "scroll:down"]);
  });

  it("prefers reviewed/redacted actions when a review is present", () => {
    const trajectory: TrajectorySpan = {
      id: "traj-2",
      sessionId: "s-2",
      createdAt: "2026-01-01T00:00:00.000Z",
      captureTier: "full",
      observations: [],
      actions: [
        { kind: "action", tool: "device", summary: "typed secret", ts: 500, metadata: { gesture: "type" } },
      ],
      review: {
        status: "approved",
        reviewedAt: "2026-01-02T00:00:00.000Z",
        reviewedBy: "reviewer",
        redactedActions: [{ ts: 500, tool: "device", summary: "typed into field" }],
      },
    };

    const sequence = movementSequenceFromTrajectory(trajectory);
    expect(sequence.events).toHaveLength(1);
    // Inferred from the redacted summary (no raw metadata leaked).
    expect(sequence.events[0]!.type).toBe("type");
  });

  it("keeps events ordered by timestamp", () => {
    const events: MovementEvent[] = [];
    const trajectory: TrajectorySpan = {
      id: "traj-3",
      sessionId: "s-3",
      createdAt: "2026-01-01T00:00:00.000Z",
      captureTier: "app",
      observations: [],
      actions: [
        { kind: "action", tool: "device", summary: "b", ts: 200, metadata: { gesture: "tap", target: "b" } },
        { kind: "action", tool: "device", summary: "a", ts: 100, metadata: { gesture: "tap", target: "a" } },
      ],
    };
    const sequence = movementSequenceFromTrajectory(trajectory);
    expect(sequence.events.map((e) => e.target)).toEqual(["a", "b"]);
    expect(events).toHaveLength(0);
  });
});
