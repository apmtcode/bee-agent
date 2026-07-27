import { describe, expect, it } from "vitest";
import type { ExportedReplayManifest } from "./export-manifest.js";
import {
  DeterministicMockMovementBackend,
  buildMovementDataset,
  datasetFromReviewedExport,
  goalSimilarity,
  tokenizeGoal,
  type MovementExample,
} from "./movement-backend.js";

function replay(trajectoryId: string, observations: string[], actions: Array<{ tool: string; summary: string }>): ExportedReplayManifest {
  let ts = 0;
  const events: ExportedReplayManifest["events"] = [];
  for (const summary of observations) {
    events.push({ kind: "observation", ts: ts++, trajectoryId, source: "device", summary });
  }
  for (const action of actions) {
    events.push({ kind: "action", ts: ts++, trajectoryId, tool: action.tool, summary: action.summary });
  }
  return { sessionId: `session-${trajectoryId}`, trajectoryIds: [trajectoryId], eventCount: events.length, events };
}

describe("tokenizeGoal / goalSimilarity", () => {
  it("drops stop tokens and lowercases", () => {
    expect(tokenizeGoal("Open the Safari browser on device")).toEqual(["open", "safari", "browser"]);
  });

  it("scores identical goals as 1 and disjoint goals as 0", () => {
    expect(goalSimilarity("open safari", "open safari")).toBe(1);
    expect(goalSimilarity("open safari", "compose email")).toBe(0);
  });

  it("scores partial overlap between 0 and 1", () => {
    const score = goalSimilarity("open safari browser", "open chrome browser");
    expect(score).toBeGreaterThan(0);
    expect(score).toBeLessThan(1);
  });
});

describe("buildMovementDataset", () => {
  it("derives goal from observations and ordered steps from actions", () => {
    const dataset = buildMovementDataset([
      replay("t1", ["Safari active on device"], [
        { tool: "device", summary: "tapped address bar" },
        { tool: "device", summary: "typed anthropic.com" },
      ]),
    ]);
    expect(dataset).toHaveLength(1);
    expect(dataset[0]!.trajectoryId).toBe("t1");
    expect(dataset[0]!.goal).toBe("Safari active on device");
    expect(dataset[0]!.steps.map((s) => s.summary)).toEqual(["tapped address bar", "typed anthropic.com"]);
    expect(dataset[0]!.steps.every((s) => s.confidence === 1)).toBe(true);
  });

  it("orders steps by timestamp regardless of event order", () => {
    const out: ExportedReplayManifest = {
      sessionId: "s",
      trajectoryIds: ["t"],
      eventCount: 3,
      events: [
        { kind: "action", ts: 30, trajectoryId: "t", tool: "device", summary: "third" },
        { kind: "action", ts: 10, trajectoryId: "t", tool: "device", summary: "first" },
        { kind: "action", ts: 20, trajectoryId: "t", tool: "device", summary: "second" },
      ],
    };
    const dataset = buildMovementDataset([out]);
    expect(dataset[0]!.steps.map((s) => s.summary)).toEqual(["first", "second", "third"]);
  });

  it("skips trajectories that recorded no actions", () => {
    const dataset = buildMovementDataset([replay("t1", ["idle observation only"], [])]);
    expect(dataset).toEqual([]);
  });

  it("reads replays straight from a reviewed export manifest", () => {
    const dataset = datasetFromReviewedExport({
      version: 1,
      createdAt: "2026-07-27T00:00:00.000Z",
      reviewedBy: "reviewer",
      purpose: "test",
      targetPlatform: "apple-silicon",
      modes: ["sft"],
      rawCaptureIncluded: false,
      promotedSkills: [],
      executableSkills: [],
      executableSkillRuns: [],
      memories: [],
      trajectories: [],
      replays: [replay("t1", ["Notes app"], [{ tool: "device", summary: "typed a memo" }])],
    });
    expect(dataset).toHaveLength(1);
    expect(dataset[0]!.goal).toBe("Notes app");
  });
});

describe("DeterministicMockMovementBackend", () => {
  const dataset: MovementExample[] = [
    {
      trajectoryId: "open-safari",
      goal: "open safari browser",
      steps: [
        { tool: "device", summary: "tapped safari icon", confidence: 1, target: "safari" },
        { tool: "device", summary: "waited for safari to load", confidence: 1 },
      ],
    },
    {
      trajectoryId: "compose-mail",
      goal: "compose new mail message",
      steps: [{ tool: "device", summary: "tapped compose button", confidence: 1 }],
    },
  ];

  it("repeats a recorded movement verbatim on a close match", async () => {
    const policy = await new DeterministicMockMovementBackend().loadPolicy({ policyId: "p1", examples: dataset });
    const result = policy.predict({ goal: "open safari browser" });
    expect(result.strategy).toBe("repeat");
    expect(result.generalized).toBe(false);
    expect(result.matchScore).toBe(1);
    expect(result.matchedTrajectoryId).toBe("open-safari");
    expect(result.steps.map((s) => s.summary)).toEqual(["tapped safari icon", "waited for safari to load"]);
  });

  it("generalizes to a related target by entity substitution", async () => {
    const policy = await new DeterministicMockMovementBackend().loadPolicy({ policyId: "p1", examples: dataset });
    const result = policy.predict({ goal: "open chrome browser" });
    expect(result.strategy).toBe("generalize");
    expect(result.generalized).toBe(true);
    expect(result.matchedTrajectoryId).toBe("open-safari");
    // "safari" (unique to the example) -> "chrome" (unique to the request).
    expect(result.steps[0]!.summary).toBe("tapped chrome icon");
    expect(result.steps[0]!.target).toBe("chrome");
    expect(result.steps[1]!.summary).toBe("waited for chrome to load");
    // Generalized steps carry reduced confidence.
    expect(result.steps[0]!.confidence).toBeLessThan(1);
  });

  it("honours maxSteps", async () => {
    const policy = await new DeterministicMockMovementBackend().loadPolicy({ policyId: "p1", examples: dataset });
    const result = policy.predict({ goal: "open safari browser", maxSteps: 1 });
    expect(result.steps).toHaveLength(1);
  });

  it("is deterministic across repeated inference", async () => {
    const policy = await new DeterministicMockMovementBackend().loadPolicy({ policyId: "p1", examples: dataset });
    const first = policy.predict({ goal: "open firefox browser" });
    const second = policy.predict({ goal: "open firefox browser" });
    expect(second).toEqual(first);
  });

  it("returns an empty prediction when the dataset is empty", async () => {
    const policy = await new DeterministicMockMovementBackend().loadPolicy({ policyId: "p1", examples: [] });
    const result = policy.predict({ goal: "anything" });
    expect(result.strategy).toBe("empty");
    expect(result.steps).toEqual([]);
    expect(result.matchScore).toBe(0);
    expect(policy.exampleCount).toBe(0);
  });

  it("respects a custom repeatThreshold to force generalization", async () => {
    const policy = await new DeterministicMockMovementBackend().loadPolicy({
      policyId: "p1",
      examples: dataset,
      repeatThreshold: 1.01,
    });
    const result = policy.predict({ goal: "open safari browser" });
    // Even an exact match now falls below threshold -> generalize path.
    expect(result.strategy).toBe("generalize");
  });
});
