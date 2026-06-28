import { describe, expect, it } from "vitest";
import { tokenizeTrajectory } from "./movement-model.js";
import {
  DEFAULT_SKILL_TEMPLATES,
  generateSyntheticTrajectories,
} from "./synthetic-movements.js";

describe("generateSyntheticTrajectories", () => {
  it("is deterministic for a given seed", () => {
    const a = generateSyntheticTrajectories({ seed: 42, count: 6 });
    const b = generateSyntheticTrajectories({ seed: 42, count: 6 });
    expect(a).toEqual(b);
  });

  it("produces different data for different seeds", () => {
    const a = generateSyntheticTrajectories({ seed: 1, count: 6 });
    const b = generateSyntheticTrajectories({ seed: 2, count: 6 });
    expect(a).not.toEqual(b);
  });

  it("generates the requested count with non-empty action sequences", () => {
    const trajectories = generateSyntheticTrajectories({ seed: 7, count: 10 });
    expect(trajectories).toHaveLength(10);
    for (const trajectory of trajectories) {
      expect(trajectory.actions.length).toBeGreaterThan(0);
      expect(tokenizeTrajectory(trajectory).length).toBe(trajectory.actions.length);
    }
  });

  it("emits monotonically increasing timestamps within a trajectory", () => {
    const [trajectory] = generateSyntheticTrajectories({ seed: 3, count: 1 });
    const timestamps = trajectory!.actions.map((a) => a.ts);
    const sorted = [...timestamps].sort((x, y) => x - y);
    expect(timestamps).toEqual(sorted);
    expect(new Set(timestamps).size).toBe(timestamps.length);
  });

  it("tags actions with the originating skill template", () => {
    const trajectories = generateSyntheticTrajectories({ seed: 9, count: 12 });
    const skills = new Set(
      trajectories.flatMap((t) => t.actions.map((a) => a.metadata?.["skill"])),
    );
    for (const skill of skills) {
      expect(DEFAULT_SKILL_TEMPLATES.some((template) => template.name === skill)).toBe(true);
    }
  });

  it("returns an empty set when no templates are provided", () => {
    expect(generateSyntheticTrajectories({ templates: [] })).toEqual([]);
  });
});
