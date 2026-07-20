import { describe, expect, it } from "vitest";
import { buildTrajectorySpan, type TrajectorySpan } from "../capture/trajectory.js";
import type { ReviewedExportManifest } from "./export-manifest.js";
import {
  MovementPolicyEngine,
  RetrievalMovementBackend,
  buildMovementExamplesFromManifest,
  buildMovementExamplesFromTrajectories,
  evaluateMovementPolicy,
  tokenizeMovementText,
  type MovementTrainingExample,
} from "./policy.js";

function syntheticTrajectory(params: {
  id: string;
  goal: string;
  actions: Array<{ tool: string; summary: string; target?: string }>;
  status?: "success" | "failure" | "aborted";
  reward?: number;
}): TrajectorySpan {
  return buildTrajectorySpan({
    id: params.id,
    sessionId: `session-${params.id}`,
    observations: [{ kind: "observation", source: "os", summary: params.goal, ts: 1 }],
    actions: params.actions.map((action, index) => ({
      kind: "action",
      tool: action.tool,
      summary: action.summary,
      ts: 2 + index,
      ...(action.target ? { metadata: { target: action.target } } : {}),
    })),
    outcome: params.status
      ? { status: params.status, summary: params.goal, ...(params.reward !== undefined ? { reward: params.reward } : {}) }
      : undefined,
  });
}

describe("tokenizeMovementText", () => {
  it("lowercases, splits and drops stop-words", () => {
    expect(tokenizeMovementText("Open the Settings Panel")).toEqual(["open", "settings", "panel"]);
  });
});

describe("RetrievalMovementBackend", () => {
  it("reproduces recorded movements verbatim on an exact goal match (objective 2c)", () => {
    const backend = new RetrievalMovementBackend();
    backend.fit([
      {
        trajectoryId: "t1",
        goal: "open the settings panel",
        contextTokens: tokenizeMovementText("open the settings panel"),
        steps: [
          { tool: "ui", summary: "click settings", target: "settings" },
          { tool: "ui", summary: "scroll settings" },
        ],
      },
    ]);

    const prediction = backend.predict({ goal: "open the settings panel" });

    expect(prediction.similarity).toBe(1);
    expect(prediction.generalized).toBe(false);
    expect(prediction.matchedTrajectoryId).toBe("t1");
    expect(prediction.steps).toEqual([
      { tool: "ui", summary: "click settings", target: "settings" },
      { tool: "ui", summary: "scroll settings" },
    ]);
  });

  it("generalizes to a related goal by substituting the changed entity (objective 2d)", () => {
    const backend = new RetrievalMovementBackend();
    backend.fit([
      {
        trajectoryId: "t1",
        goal: "open the settings panel",
        contextTokens: tokenizeMovementText("open the settings panel"),
        steps: [
          { tool: "ui", summary: "click settings", target: "settings" },
          { tool: "ui", summary: "scroll settings" },
        ],
      },
    ]);

    const prediction = backend.predict({ goal: "open the profile panel" });

    expect(prediction.generalized).toBe(true);
    expect(prediction.similarity).toBeGreaterThan(0);
    expect(prediction.similarity).toBeLessThan(1);
    expect(prediction.substitutions).toEqual([{ from: "settings", to: "profile" }]);
    expect(prediction.steps).toEqual([
      { tool: "ui", summary: "click profile", target: "profile" },
      { tool: "ui", summary: "scroll profile" },
    ]);
  });

  it("prefers the most similar example, breaking ties by reward", () => {
    const backend = new RetrievalMovementBackend();
    const shared = tokenizeMovementText("compose a message");
    backend.fit([
      {
        trajectoryId: "low",
        goal: "compose a message",
        contextTokens: shared,
        steps: [{ tool: "device", summary: "tap send" }],
        reward: 0.1,
      },
      {
        trajectoryId: "high",
        goal: "compose a message",
        contextTokens: shared,
        steps: [{ tool: "device", summary: "tap send now" }],
        reward: 0.9,
      },
    ]);

    const prediction = backend.predict({ goal: "compose a message" });
    expect(prediction.matchedTrajectoryId).toBe("high");
  });

  it("returns an empty prediction when nothing has been fitted", () => {
    const backend = new RetrievalMovementBackend();
    const prediction = backend.predict({ goal: "anything" });
    expect(prediction.steps).toEqual([]);
    expect(prediction.similarity).toBe(0);
  });

  it("caps predicted steps at maxSteps", () => {
    const backend = new RetrievalMovementBackend();
    backend.fit([
      {
        trajectoryId: "t1",
        goal: "run the flow",
        contextTokens: tokenizeMovementText("run the flow"),
        steps: [
          { tool: "a", summary: "one" },
          { tool: "b", summary: "two" },
          { tool: "c", summary: "three" },
        ],
      },
    ]);

    const prediction = backend.predict({ goal: "run the flow", maxSteps: 2 });
    expect(prediction.steps).toHaveLength(2);
  });
});

describe("dataset adapters", () => {
  it("builds examples from raw trajectory spans, pulling targets from metadata", () => {
    const trajectories = [
      syntheticTrajectory({
        id: "t1",
        goal: "open the inbox",
        actions: [{ tool: "device", summary: "tapped inbox", target: "inbox" }],
        status: "success",
        reward: 1,
      }),
    ];

    const examples = buildMovementExamplesFromTrajectories(trajectories);
    expect(examples).toHaveLength(1);
    expect(examples[0]).toMatchObject({
      trajectoryId: "t1",
      goal: "open the inbox",
      outcomeStatus: "success",
      reward: 1,
      steps: [{ tool: "device", summary: "tapped inbox", target: "inbox" }],
    });
  });

  it("skips trajectories with no recorded movements", () => {
    const trajectories = [syntheticTrajectory({ id: "empty", goal: "idle", actions: [] })];
    expect(buildMovementExamplesFromTrajectories(trajectories)).toHaveLength(0);
  });

  it("builds examples from an exported reviewed manifest", () => {
    const manifest: ReviewedExportManifest = {
      version: 1,
      createdAt: "2026-01-01T00:00:00.000Z",
      reviewedBy: "operator",
      purpose: "movement policy",
      targetPlatform: "apple-silicon",
      modes: ["sft"],
      rawCaptureIncluded: false,
      promotedSkills: [],
      executableSkills: [],
      executableSkillRuns: [],
      memories: [],
      trajectories: [
        {
          id: "t1",
          sessionId: "s1",
          createdAt: "2026-01-01T00:00:00.000Z",
          captureTier: "app",
          observationCount: 1,
          actionCount: 1,
          outcomeStatus: "success",
          reward: 0.5,
        },
      ],
      replays: [
        {
          sessionId: "s1",
          trajectoryIds: ["t1"],
          eventCount: 2,
          events: [
            { kind: "observation", ts: 1, trajectoryId: "t1", source: "os", summary: "open the mail app" },
            { kind: "action", ts: 2, trajectoryId: "t1", tool: "device", summary: "tapped compose" },
          ],
        },
      ],
    };

    const examples = buildMovementExamplesFromManifest(manifest);
    expect(examples).toHaveLength(1);
    expect(examples[0]).toMatchObject({
      trajectoryId: "t1",
      goal: "open the mail app",
      outcomeStatus: "success",
      reward: 0.5,
      steps: [{ tool: "device", summary: "tapped compose" }],
    });
  });
});

describe("MovementPolicyEngine + evaluation", () => {
  it("fits from trajectories and rolls out a generalized movement", async () => {
    const engine = new MovementPolicyEngine();
    await engine.fitFromTrajectories([
      syntheticTrajectory({
        id: "t1",
        goal: "search for invoices",
        actions: [
          { tool: "ui", summary: "focus search box" },
          { tool: "keyboard", summary: "type invoices", target: "invoices" },
        ],
        status: "success",
        reward: 1,
      }),
    ]);

    const prediction = await engine.rollout("search for receipts");
    expect(prediction.generalized).toBe(true);
    expect(prediction.steps).toEqual([
      { tool: "ui", summary: "focus search box" },
      { tool: "keyboard", summary: "type receipts", target: "receipts" },
    ]);
  });

  it("scores reproduction fidelity at 100% on the training set", async () => {
    const examples: MovementTrainingExample[] = buildMovementExamplesFromTrajectories([
      syntheticTrajectory({ id: "a", goal: "open settings", actions: [{ tool: "ui", summary: "click gear" }] }),
      syntheticTrajectory({ id: "b", goal: "close settings", actions: [{ tool: "ui", summary: "click x" }] }),
    ]);
    const backend = new RetrievalMovementBackend();
    backend.fit(examples);

    const evaluation = await evaluateMovementPolicy(backend, examples);
    expect(evaluation.count).toBe(2);
    expect(evaluation.exactSequenceMatch).toBe(1);
    expect(evaluation.toolAccuracy).toBe(1);
    expect(evaluation.generalizedRate).toBe(0);
  });

  it("still recovers the right tool sequence on held-out related goals", async () => {
    const backend = new RetrievalMovementBackend();
    backend.fit(
      buildMovementExamplesFromTrajectories([
        syntheticTrajectory({
          id: "train",
          goal: "delete the draft",
          actions: [
            { tool: "ui", summary: "select draft" },
            { tool: "keyboard", summary: "press delete" },
          ],
        }),
      ]),
    );

    const heldOut = buildMovementExamplesFromTrajectories([
      syntheticTrajectory({
        id: "held",
        goal: "delete the reminder",
        actions: [
          { tool: "ui", summary: "select reminder" },
          { tool: "keyboard", summary: "press delete" },
        ],
      }),
    ]);

    const evaluation = await evaluateMovementPolicy(backend, heldOut);
    expect(evaluation.toolAccuracy).toBe(1);
    expect(evaluation.generalizedRate).toBe(1);
  });
});
