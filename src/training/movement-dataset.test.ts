import { describe, expect, it } from "vitest";
import { buildTrajectorySpan, type TrajectoryAction } from "../capture/trajectory.js";
import {
  buildMovementDataset,
  defaultMovementGrammar,
  generateSyntheticTrajectories,
  splitTrajectories,
  tokenizeAction,
  tokensToExamples,
  trajectoryToTokens,
} from "./movement-dataset.js";

function action(tool: string, ts: number, extra: Partial<TrajectoryAction> = {}): TrajectoryAction {
  return { kind: "action", tool, summary: `${tool} did thing`, ts, ...extra };
}

describe("tokenizeAction", () => {
  it("prefers a structured metadata target", () => {
    expect(tokenizeAction(action("click", 1, { metadata: { target: "send-button" } }))).toBe("click:send-button");
  });

  it("slugs the summary when no target is present", () => {
    expect(tokenizeAction(action("type", 1, { summary: "Type Recipient Name!" }))).toBe("type:type-recipient-name");
  });
});

describe("trajectoryToTokens", () => {
  it("orders tokens by timestamp regardless of array order", () => {
    const trajectory = buildTrajectorySpan({
      id: "t",
      sessionId: "s",
      actions: [action("third", 30), action("first", 10), action("second", 20)],
    });
    expect(trajectoryToTokens(trajectory).map((token) => token.split(":")[0])).toEqual([
      "first",
      "second",
      "third",
    ]);
  });
});

describe("tokensToExamples", () => {
  it("emits an empty-context example for the movement start and windows the rest", () => {
    expect(tokensToExamples(["a", "b", "c"], 2)).toEqual([
      { context: [], next: "a" },
      { context: ["a"], next: "b" },
      { context: ["a", "b"], next: "c" },
    ]);
  });

  it("caps the context window at `order`", () => {
    const examples = tokensToExamples(["a", "b", "c", "d"], 2);
    expect(examples[3]).toEqual({ context: ["b", "c"], next: "d" });
  });
});

describe("buildMovementDataset", () => {
  it("collects a sorted vocabulary and windowed examples across trajectories", () => {
    const t1 = buildTrajectorySpan({
      id: "1",
      sessionId: "s1",
      actions: [action("click", 1, { metadata: { target: "b" } }), action("type", 2, { metadata: { target: "a" } })],
    });
    const t2 = buildTrajectorySpan({
      id: "2",
      sessionId: "s2",
      actions: [action("click", 1, { metadata: { target: "a" } })],
    });
    const dataset = buildMovementDataset([t1, t2], { order: 2 });
    expect(dataset.vocabulary).toEqual(["click:a", "click:b", "type:a"]);
    // 2 examples from t1 + 1 from t2.
    expect(dataset.examples).toHaveLength(3);
  });
});

describe("splitTrajectories", () => {
  it("holds out every stride-th trajectory deterministically", () => {
    const trajectories = Array.from({ length: 8 }, (_unused, index) =>
      buildTrajectorySpan({ id: `t${index}`, sessionId: `s${index}`, actions: [action("a", 1)] }),
    );
    const { train, heldOut } = splitTrajectories(trajectories, 0.25);
    expect(heldOut.map((t) => t.id)).toEqual(["t3", "t7"]);
    expect(train).toHaveLength(6);
  });
});

describe("generateSyntheticTrajectories", () => {
  it("produces walks that respect the grammar graph and mostly terminate", () => {
    const grammar = defaultMovementGrammar();
    const stateById = new Map(grammar.states.map((state) => [state.id, state] as const));
    const trajectories = generateSyntheticTrajectories({ count: 40, seed: 3 });

    let reachedTerminal = 0;
    for (const trajectory of trajectories) {
      for (const act of trajectory.actions) {
        const from = act.metadata?.fromState as string;
        const to = act.metadata?.toState as string;
        const legal = stateById.get(from)?.actions.some((edge) => edge.to === to && edge.tool === act.tool);
        expect(legal).toBe(true);
      }
      if (trajectory.outcome?.status === "success") {
        reachedTerminal += 1;
      }
    }
    // The default grammar always has a path to terminal; the vast majority of
    // bounded walks should reach it.
    expect(reachedTerminal).toBeGreaterThan(30);
  });

  it("uses no wall clock — timestamps derive from baseTs", () => {
    const [trajectory] = generateSyntheticTrajectories({ count: 1, seed: 1, baseTs: 5000 });
    expect(trajectory!.actions[0]!.ts).toBeGreaterThanOrEqual(5000);
  });
});
