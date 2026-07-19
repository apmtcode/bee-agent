import { describe, expect, it } from "vitest";
import {
  DEFAULT_SYNTHETIC_WORKFLOWS,
  generateSyntheticTrajectories,
  variantWorkflow,
} from "./synthetic.js";

const workflow = DEFAULT_SYNTHETIC_WORKFLOWS[0]!;

describe("generateSyntheticTrajectories", () => {
  it("is deterministic and monotonic in time", () => {
    const a = generateSyntheticTrajectories({ sessionId: "s", workflow, repeats: 2 });
    const b = generateSyntheticTrajectories({ sessionId: "s", workflow, repeats: 2 });
    // Movement content is byte-stable; only the wall-clock createdAt stamp varies.
    const content = (list: ReturnType<typeof generateSyntheticTrajectories>) =>
      list.map(({ createdAt: _createdAt, ...rest }) => rest);
    expect(content(a)).toEqual(content(b));
    expect(a).toHaveLength(2);
    for (const trajectory of a) {
      const timestamps = trajectory.actions.map((action) => action.ts);
      const sorted = [...timestamps].sort((x, y) => x - y);
      expect(timestamps).toEqual(sorted);
      expect(trajectory.actions).toHaveLength(workflow.steps.length);
    }
  });

  it("gives each repeat a stable, unique id and non-overlapping timestamps", () => {
    const trajectories = generateSyntheticTrajectories({ sessionId: "s", workflow, repeats: 3 });
    expect(trajectories.map((t) => t.id)).toEqual(["compose-email-0", "compose-email-1", "compose-email-2"]);
    const lastOfFirst = trajectories[0]!.actions.at(-1)!.ts;
    const firstOfSecond = trajectories[1]!.actions[0]!.ts;
    expect(firstOfSecond).toBeGreaterThan(lastOfFirst);
  });
});

describe("variantWorkflow", () => {
  it("drops a step", () => {
    const variant = variantWorkflow(workflow, { kind: "drop", at: 1 });
    expect(variant.steps).toHaveLength(workflow.steps.length - 1);
    expect(variant.steps[1]).toEqual(workflow.steps[2]);
  });

  it("inserts a step", () => {
    const step = { tool: "device", summary: "tapped attach" };
    const variant = variantWorkflow(workflow, { kind: "insert", at: 2, step });
    expect(variant.steps).toHaveLength(workflow.steps.length + 1);
    expect(variant.steps[2]).toEqual(step);
  });

  it("replaces a step", () => {
    const step = { tool: "device", summary: "tapped draft" };
    const variant = variantWorkflow(workflow, { kind: "replace", at: 0, step });
    expect(variant.steps[0]).toEqual(step);
    expect(variant.steps).toHaveLength(workflow.steps.length);
  });
});
