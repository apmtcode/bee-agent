import { describe, expect, it } from "vitest";
import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  MovementInferenceEngine,
  NgramMovementBackend,
  buildMovementExamplesFromTrajectories,
  evaluateReplayFidelity,
  type MovementExample,
} from "./movement-model.js";

function trajectory(
  id: string,
  goal: string,
  tools: string[],
  overrides: Partial<TrajectorySpan> = {},
): TrajectorySpan {
  return {
    id,
    sessionId: `session-${id}`,
    createdAt: "2026-01-01T00:00:00.000Z",
    captureTier: "operator",
    observations: [],
    actions: tools.map((tool, index) => ({
      kind: "action",
      tool,
      summary: `${tool} step`,
      ts: index + 1,
    })),
    outcome: { status: "success", summary: goal },
    review: { status: "approved", reviewedAt: "2026-01-02T00:00:00.000Z", reviewedBy: "operator" },
    ...overrides,
  };
}

describe("buildMovementExamplesFromTrajectories", () => {
  it("only includes approved trajectories with actions, ordered by timestamp", () => {
    const approved = trajectory("t1", "open editor", ["focus", "click", "type"]);
    const unapproved = trajectory("t2", "open editor", ["focus"], {
      review: { status: "pending", reviewedAt: "", reviewedBy: "" },
    });
    const empty = trajectory("t3", "noop", []);

    const examples = buildMovementExamplesFromTrajectories([approved, unapproved, empty]);

    expect(examples).toHaveLength(1);
    expect(examples[0]!.goal).toBe("open editor");
    expect(examples[0]!.steps.map((step) => step.tool)).toEqual(["focus", "click", "type"]);
  });

  it("prefers reviewer-redacted actions when present", () => {
    const span = trajectory("t1", "g", ["raw"], {
      review: {
        status: "approved",
        reviewedAt: "",
        reviewedBy: "operator",
        redactedActions: [
          { ts: 2, tool: "safe-b", summary: "b" },
          { ts: 1, tool: "safe-a", summary: "a" },
        ],
      },
    });

    const examples = buildMovementExamplesFromTrajectories([span]);

    expect(examples[0]!.steps.map((step) => step.tool)).toEqual(["safe-a", "safe-b"]);
  });
});

describe("NgramMovementBackend + MovementInferenceEngine — repeat (objective 2c)", () => {
  it("reproduces a recorded movement sequence exactly", () => {
    const engine = new MovementInferenceEngine(new NgramMovementBackend());
    engine.train([trajectory("t1", "open the settings panel", ["focus-window", "click-menu", "click-settings"])]);

    const plan = engine.repeat("open the settings panel");

    expect(plan.steps.map((step) => step.tool)).toEqual(["focus-window", "click-menu", "click-settings"]);
    expect(plan.terminated).toBe(true);
    expect(plan.stopReason).toBe("model-stop");
    expect(plan.backend).toBe("ngram-mock");
  });

  it("is deterministic across runs and instances", () => {
    const examples: MovementExample[] = [
      { goal: "save file", steps: [{ tool: "cmd", summary: "cmd" }, { tool: "s", summary: "s" }] },
      { goal: "save document", steps: [{ tool: "cmd", summary: "cmd" }, { tool: "s", summary: "s" }] },
    ];
    const a = new MovementInferenceEngine(new NgramMovementBackend());
    const b = new MovementInferenceEngine(new NgramMovementBackend());
    a.fitExamples(examples);
    b.fitExamples(examples);

    expect(a.repeat("save file").steps).toEqual(b.repeat("save file").steps);
  });

  it("honors the loop guard against degenerate repetition", () => {
    const engine = new MovementInferenceEngine(new NgramMovementBackend({ maxOrder: 0 }));
    // Order-0 with a self-loop demo forces the same tool repeatedly.
    engine.fitExamples([{ goal: "spin", steps: [{ tool: "x", summary: "x" }, { tool: "x", summary: "x" }, { tool: "x", summary: "x" }] }]);

    const plan = engine.repeat("spin", { maxConsecutiveRepeats: 3, maxSteps: 50 });

    expect(plan.stopReason).toBe("loop-guard");
    expect(plan.steps.length).toBeLessThan(50);
  });
});

describe("generalization to related-but-unseen goals (objective 2d)", () => {
  it("borrows movements from the most similar demonstrated goal", () => {
    const engine = new MovementInferenceEngine(new NgramMovementBackend());
    engine.fitExamples([
      { goal: "open the file menu", steps: [{ tool: "focus", summary: "focus" }, { tool: "open-file-menu", summary: "menu" }] },
      { goal: "open the edit menu", steps: [{ tool: "focus", summary: "focus" }, { tool: "open-edit-menu", summary: "menu" }] },
    ]);

    // Unseen goal, but shares "open ... menu file" tokens with the first demo.
    const plan = engine.repeat("please open file menu now");

    expect(plan.steps[0]!.tool).toBe("focus");
    expect(plan.steps.map((step) => step.tool)).toContain("open-file-menu");
    expect(plan.steps.map((step) => step.tool)).not.toContain("open-edit-menu");
  });

  it("still produces a plan when the goal overlaps nothing (base-weight fallback)", () => {
    const engine = new MovementInferenceEngine(new NgramMovementBackend());
    engine.fitExamples([{ goal: "alpha", steps: [{ tool: "step-1", summary: "one" }] }]);

    const plan = engine.repeat("completely unrelated objective");

    expect(plan.steps.map((step) => step.tool)).toEqual(["step-1"]);
  });
});

describe("evaluateReplayFidelity", () => {
  it("scores an exact reproduction as 1", () => {
    const seq = [{ tool: "a", summary: "" }, { tool: "b", summary: "" }];
    expect(evaluateReplayFidelity(seq, seq).score).toBe(1);
  });

  it("uses longest common subsequence for partial matches", () => {
    const predicted = [{ tool: "a", summary: "" }, { tool: "x", summary: "" }, { tool: "c", summary: "" }];
    const expected = [{ tool: "a", summary: "" }, { tool: "b", summary: "" }, { tool: "c", summary: "" }];
    const fidelity = evaluateReplayFidelity(predicted, expected);
    expect(fidelity.matched).toBe(2);
    expect(fidelity.score).toBeCloseTo(2 / 3);
  });

  it("treats two empty sequences as perfect fidelity", () => {
    expect(evaluateReplayFidelity([], []).score).toBe(1);
  });

  it("measures generalization fidelity end-to-end on a held-out goal", () => {
    const engine = new MovementInferenceEngine(new NgramMovementBackend());
    engine.fitExamples([
      { goal: "open file menu", steps: [{ tool: "focus", summary: "" }, { tool: "menu-file", summary: "" }, { tool: "select", summary: "" }] },
      { goal: "open view menu", steps: [{ tool: "focus", summary: "" }, { tool: "menu-view", summary: "" }, { tool: "select", summary: "" }] },
    ]);

    const plan = engine.repeat("open file menu quickly");
    const expected = [{ tool: "focus", summary: "" }, { tool: "menu-file", summary: "" }, { tool: "select", summary: "" }];
    const fidelity = evaluateReplayFidelity(plan.steps, expected);

    expect(fidelity.score).toBeGreaterThanOrEqual(0.66);
  });
});
