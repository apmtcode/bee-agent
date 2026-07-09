import { describe, expect, it } from "vitest";
import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  buildMovementDataset,
  contextTokens,
  evaluateMovementPolicy,
  generateSyntheticMovementTrajectories,
  NearestNeighborMovementBackend,
  type MovementContext,
} from "./movement-policy.js";

function span(id: string, overrides: Partial<TrajectorySpan> = {}): TrajectorySpan {
  return {
    id,
    sessionId: "s1",
    createdAt: "2026-01-01T00:00:00.000Z",
    captureTier: "app",
    observations: [
      {
        kind: "observation",
        source: "device",
        summary: "browser on Inbox",
        ts: 100,
        metadata: { appName: "browser", screenTitle: "Inbox" },
      },
    ],
    actions: [
      { kind: "action", tool: "device", summary: "tap send button", ts: 110, metadata: { gesture: "tap", target: "send button" } },
    ],
    ...overrides,
  };
}

describe("buildMovementDataset", () => {
  it("emits one example per action with pre-action context", () => {
    const dataset = buildMovementDataset([
      span("t1", {
        actions: [
          { kind: "action", tool: "device", summary: "tap search box", ts: 110, metadata: { gesture: "tap", target: "search box" } },
          { kind: "action", tool: "device", summary: "type hello", ts: 120, metadata: { gesture: "type", target: "search box" } },
        ],
      }),
    ]);
    expect(dataset.version).toBe(1);
    expect(dataset.examples).toHaveLength(2);
    // Second action's context should know the first action's tool.
    expect(dataset.examples[1]!.context.priorActionTools).toEqual(["device"]);
    expect(dataset.examples[0]!.context.appId).toBe("browser");
    expect(dataset.examples[0]!.context.screenTitle).toBe("Inbox");
    expect(dataset.examples[1]!.action).toMatchObject({ gesture: "type", target: "search box" });
  });

  it("only includes observations that precede the action", () => {
    const dataset = buildMovementDataset([
      span("t1", {
        observations: [
          { kind: "observation", source: "device", summary: "early obs", ts: 100 },
          { kind: "observation", source: "device", summary: "late obs", ts: 200 },
        ],
        actions: [{ kind: "action", tool: "device", summary: "tap x", ts: 150 }],
      }),
    ]);
    expect(dataset.examples[0]!.context.observationSummaries).toEqual(["early obs"]);
  });
});

describe("NearestNeighborMovementBackend", () => {
  it("reproduces a recorded movement exactly for an identical context", async () => {
    const dataset = buildMovementDataset([span("t1")]);
    const policy = await new NearestNeighborMovementBackend().train(dataset);
    expect(policy.exampleCount).toBe(1);
    const prediction = policy.predict(dataset.examples[0]!.context);
    expect(prediction).toBeDefined();
    expect(prediction!.exact).toBe(true);
    expect(prediction!.confidence).toBe(1);
    expect(prediction!.action.summary).toBe("tap send button");
  });

  it("generalizes to a related-but-unseen context with confidence < 1", async () => {
    const dataset = buildMovementDataset([
      span("t1", {
        observations: [{ kind: "observation", source: "device", summary: "browser on Inbox", ts: 100, metadata: { appName: "browser", screenTitle: "Inbox" } }],
        actions: [{ kind: "action", tool: "device", summary: "tap send button", ts: 110 }],
      }),
      span("t2", {
        observations: [{ kind: "observation", source: "device", summary: "editor on main.ts", ts: 100, metadata: { appName: "editor", screenTitle: "main" } }],
        actions: [{ kind: "action", tool: "device", summary: "tap save icon", ts: 110 }],
      }),
    ]);
    const policy = await new NearestNeighborMovementBackend().train(dataset);
    // A context that shares tokens with the browser case but is not identical.
    const query: MovementContext = {
      appId: "browser",
      screenTitle: "Inbox",
      observationSummaries: ["browser on Inbox drafts"],
      priorActionTools: [],
      priorActionSummaries: [],
    };
    const prediction = policy.predict(query);
    expect(prediction).toBeDefined();
    expect(prediction!.action.summary).toBe("tap send button");
    expect(prediction!.exact).toBe(false);
    expect(prediction!.confidence).toBeGreaterThan(0);
    expect(prediction!.confidence).toBeLessThan(1);
  });

  it("returns undefined when nothing was learned", async () => {
    const policy = await new NearestNeighborMovementBackend().train({ version: 1, examples: [] });
    expect(policy.predict({ observationSummaries: [], priorActionTools: [], priorActionSummaries: [] })).toBeUndefined();
  });

  it("is deterministic on ties (earliest example wins)", async () => {
    const dataset = buildMovementDataset([
      span("a", { actions: [{ kind: "action", tool: "device", summary: "first", ts: 110 }] }),
      span("b", { actions: [{ kind: "action", tool: "device", summary: "second", ts: 110 }] }),
    ]);
    const policy = await new NearestNeighborMovementBackend().train(dataset);
    // Both training contexts are identical → tie → earliest wins, repeatably.
    const first = policy.predict(dataset.examples[0]!.context);
    const again = policy.predict(dataset.examples[0]!.context);
    expect(first!.matchedExampleIndex).toBe(0);
    expect(again!.matchedExampleIndex).toBe(0);
    expect(first!.action.summary).toBe("first");
  });
});

describe("contextTokens", () => {
  it("namespaces tokens by field so app/obs/screen do not collide", () => {
    const tokens = contextTokens({ appId: "browser", screenTitle: "Inbox", observationSummaries: ["send now"], priorActionTools: ["Device"], priorActionSummaries: [] });
    expect(tokens.has("app:browser")).toBe(true);
    expect(tokens.has("screen:inbox")).toBe(true);
    expect(tokens.has("obs:send")).toBe(true);
    expect(tokens.has("prior:device")).toBe(true);
  });
});

describe("generateSyntheticMovementTrajectories + evaluateMovementPolicy", () => {
  it("is byte-identical for the same seed and differs across seeds", () => {
    const a = generateSyntheticMovementTrajectories({ seed: 7, spanCount: 5 });
    const b = generateSyntheticMovementTrajectories({ seed: 7, spanCount: 5 });
    const c = generateSyntheticMovementTrajectories({ seed: 8, spanCount: 5 });
    expect(JSON.stringify(a)).toBe(JSON.stringify(b));
    expect(JSON.stringify(a)).not.toBe(JSON.stringify(c));
    expect(a).toHaveLength(5);
    expect(a[0]!.actions.length).toBeGreaterThan(0);
  });

  it("reproduces recorded movements on the training set (self-match invariants)", async () => {
    const spans = generateSyntheticMovementTrajectories({ seed: 42, spanCount: 30, actionsPerSpan: 4 });
    const dataset = buildMovementDataset(spans);
    const policy = await new NearestNeighborMovementBackend().train(dataset);
    const result = evaluateMovementPolicy(policy, dataset.examples);
    expect(result.total).toBe(dataset.examples.length);
    expect(result.predicted).toBe(dataset.examples.length);
    // Every training context has an identical case in the index (itself), so
    // confidence and exact-context are guaranteed 1 / total.
    expect(result.meanConfidence).toBe(1);
    expect(result.exactContextMatches).toBe(dataset.examples.length);
    expect(result.toolMatchRate).toBe(1);
    // Sequence-aware context reproduces the majority of movements verbatim;
    // the remainder are genuinely ambiguous prefixes that collide across spans
    // in this deliberately tiny synthetic vocabulary.
    expect(result.summaryMatchRate).toBeGreaterThan(0.5);
  });

  it("achieves exact replay when every context is unambiguous", async () => {
    // Distinct screen per span + one action each → no two contexts collide.
    const spans = ["Inbox", "Compose", "Settings", "Drafts", "Sent"].map((screen, index) => ({
      id: `t${index}`,
      sessionId: "s1",
      createdAt: "2026-01-01T00:00:00.000Z",
      captureTier: "app" as const,
      observations: [
        { kind: "observation" as const, source: "device", summary: `browser on ${screen}`, ts: 100, metadata: { appName: "browser", screenTitle: screen } },
      ],
      actions: [{ kind: "action" as const, tool: "device", summary: `tap ${screen} button`, ts: 110 }],
    }));
    const dataset = buildMovementDataset(spans);
    const policy = await new NearestNeighborMovementBackend().train(dataset);
    const result = evaluateMovementPolicy(policy, dataset.examples);
    expect(result.summaryMatchRate).toBe(1);
    expect(result.meanConfidence).toBe(1);
  });

  it("generalizes above chance to held-out related trajectories", async () => {
    const train = generateSyntheticMovementTrajectories({ seed: 1, spanCount: 40, actionsPerSpan: 4 });
    const heldOut = generateSyntheticMovementTrajectories({ seed: 999, spanCount: 20, actionsPerSpan: 4, baseTs: 1_800_000_000_000 });
    const policy = await new NearestNeighborMovementBackend().train(buildMovementDataset(train));
    const result = evaluateMovementPolicy(policy, buildMovementDataset(heldOut).examples);
    expect(result.predicted).toBe(result.total);
    // Same synthetic domain (tool == "device" everywhere) → tool always matches.
    expect(result.toolMatchRate).toBe(1);
    // Held-out summaries are drawn from the same small vocabulary, so many but
    // not all are reproduced — generalization sits strictly between 0 and 1.
    expect(result.summaryMatchRate).toBeGreaterThan(0);
    expect(result.meanConfidence).toBeGreaterThan(0);
  });
});
