import { describe, expect, it } from "vitest";
import {
  defaultWorkflowTemplates,
  generateSyntheticTrajectories,
} from "./synthetic.js";

describe("generateSyntheticTrajectories", () => {
  it("is deterministic for a given seed", () => {
    const a = generateSyntheticTrajectories({ count: 8, seed: 42 });
    const b = generateSyntheticTrajectories({ count: 8, seed: 42 });
    expect(JSON.stringify(stripCreatedAt(a))).toBe(JSON.stringify(stripCreatedAt(b)));
  });

  it("produces different streams for different seeds", () => {
    const a = generateSyntheticTrajectories({ count: 8, seed: 1 });
    const b = generateSyntheticTrajectories({ count: 8, seed: 2 });
    expect(JSON.stringify(toolsOf(a))).not.toBe(JSON.stringify(toolsOf(b)));
  });

  it("emits actions in strictly increasing timestamp order", () => {
    const trajectories = generateSyntheticTrajectories({ count: 5, seed: 3, noise: 0.5 });
    for (const trajectory of trajectories) {
      for (let i = 1; i < trajectory.actions.length; i += 1) {
        expect(trajectory.actions[i]!.ts).toBeGreaterThan(trajectory.actions[i - 1]!.ts);
      }
    }
  });

  it("each trajectory matches one template's tool sequence when noise-free", () => {
    const templates = defaultWorkflowTemplates();
    const trajectories = generateSyntheticTrajectories({ count: 20, seed: 11 });
    const known = new Set(templates.map((t) => t.tools.join(">")));
    for (const trajectory of trajectories) {
      const seq = trajectory.actions.map((a) => a.tool).join(">");
      expect(known.has(seq)).toBe(true);
    }
  });

  it("injects extra movements when noise is enabled", () => {
    const templates = [defaultWorkflowTemplates()[0]!];
    const noisy = generateSyntheticTrajectories({ count: 30, seed: 7, noise: 0.9, templates });
    const base = templates[0]!.tools.length;
    const anyLonger = noisy.some((trajectory) => trajectory.actions.length > base);
    expect(anyLonger).toBe(true);
  });

  it("throws when no templates are available", () => {
    expect(() => generateSyntheticTrajectories({ count: 1, templates: [] })).toThrow(/at least one template/);
  });
});

function stripCreatedAt(trajectories: ReturnType<typeof generateSyntheticTrajectories>) {
  return trajectories.map(({ createdAt: _createdAt, ...rest }) => rest);
}

function toolsOf(trajectories: ReturnType<typeof generateSyntheticTrajectories>) {
  return trajectories.map((t) => t.actions.map((a) => a.tool));
}
