import { describe, expect, it } from "vitest";
import type { ReviewedExportManifest, ExportedReplayManifest } from "./export-manifest.js";
import {
  MovementInferenceService,
  MovementPolicyRegistry,
  RetrievalMovementPolicyBackend,
  buildMovementDataset,
  defaultMovementPolicyRegistry,
  tokenizeMovementText,
  type MovementDemonstration,
} from "./movement-policy.js";

function replay(params: {
  sessionId: string;
  trajectoryId: string;
  goal: string;
  actions: Array<{ tool: string; summary: string; ts: number }>;
}): ExportedReplayManifest {
  return {
    sessionId: params.sessionId,
    trajectoryIds: [params.trajectoryId],
    eventCount: params.actions.length + 1,
    events: [
      { kind: "transcript", ts: 0, messageId: "m0", role: "user", content: params.goal },
      ...params.actions.map((action) => ({
        kind: "action" as const,
        ts: action.ts,
        trajectoryId: params.trajectoryId,
        tool: action.tool,
        summary: action.summary,
      })),
    ],
  };
}

function manifestWith(replays: ExportedReplayManifest[]): ReviewedExportManifest {
  return {
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
    trajectories: replays.map((r) => ({
      id: r.trajectoryIds[0]!,
      sessionId: r.sessionId,
      createdAt: "2026-01-01T00:00:00.000Z",
      captureTier: "app",
      observationCount: 0,
      actionCount: r.events.filter((e) => e.kind === "action").length,
      outcomeStatus: "success" as const,
      reward: 1,
    })),
    replays,
  };
}

describe("tokenizeMovementText", () => {
  it("lowercases, splits, and drops stopwords/short tokens", () => {
    expect(tokenizeMovementText("Open the Reports folder")).toEqual(["reports", "folder"]);
  });
});

describe("buildMovementDataset", () => {
  it("extracts time-normalized actions and goal text per replay", () => {
    const manifest = manifestWith([
      replay({
        sessionId: "s1",
        trajectoryId: "t1",
        goal: "open the reports folder",
        actions: [
          { tool: "mouse.click", summary: "click Finder", ts: 1000 },
          { tool: "mouse.click", summary: "double-click reports folder", ts: 1600 },
        ],
      }),
    ]);
    const dataset = buildMovementDataset(manifest);
    expect(dataset).toHaveLength(1);
    expect(dataset[0]!.trajectoryId).toBe("t1");
    expect(dataset[0]!.actions.map((a) => a.relativeTs)).toEqual([0, 600]);
    expect(dataset[0]!.goalText).toContain("reports");
    expect(dataset[0]!.outcomeStatus).toBe("success");
  });

  it("skips replays with no actions", () => {
    const manifest = manifestWith([
      { sessionId: "s0", trajectoryIds: ["t0"], eventCount: 0, events: [] },
    ]);
    expect(buildMovementDataset(manifest)).toHaveLength(0);
  });
});

describe("RetrievalMovementPolicyBackend", () => {
  const dataset: MovementDemonstration[] = [
    {
      trajectoryId: "t1",
      sessionId: "s1",
      goalText: "open the reports folder and export csv",
      actions: [
        { tool: "mouse.click", summary: "click Finder", relativeTs: 0 },
        { tool: "mouse.click", summary: "double-click reports folder", relativeTs: 500 },
      ],
      outcomeStatus: "success",
      reward: 1,
    },
    {
      trajectoryId: "t2",
      sessionId: "s2",
      goalText: "compose an email to the finance team",
      actions: [{ tool: "keyboard.type", summary: "type recipient finance", relativeTs: 0 }],
      outcomeStatus: "success",
      reward: 1,
    },
  ];

  it("retrieves the most similar demonstration verbatim", () => {
    const backend = new RetrievalMovementPolicyBackend(dataset);
    const prediction = backend.predict({ goal: "open the reports folder" });
    expect(prediction.sourceTrajectoryId).toBe("t1");
    expect(prediction.generalized).toBe(false);
    expect(prediction.confidence).toBeGreaterThan(0);
    expect(prediction.actions.map((a) => a.summary)).toEqual([
      "click Finder",
      "double-click reports folder",
    ]);
  });

  it("generalizes to a new but related goal by substituting the differing noun", () => {
    const backend = new RetrievalMovementPolicyBackend(dataset);
    const prediction = backend.predict({ goal: "open the invoices folder and export csv" });
    expect(prediction.sourceTrajectoryId).toBe("t1");
    expect(prediction.generalized).toBe(true);
    expect(prediction.actions.map((a) => a.summary)).toEqual([
      "click Finder",
      "double-click invoices folder",
    ]);
  });

  it("applies explicit overrides on top of retrieval", () => {
    const backend = new RetrievalMovementPolicyBackend(dataset, { autoGeneralize: false });
    const prediction = backend.predict({
      goal: "open the reports folder",
      overrides: { reports: "archive" },
    });
    expect(prediction.generalized).toBe(true);
    expect(prediction.actions[1]!.summary).toBe("double-click archive folder");
  });

  it("returns no actions below the confidence threshold", () => {
    const backend = new RetrievalMovementPolicyBackend(dataset);
    const prediction = backend.predict({ goal: "restart the database cluster", minConfidence: 0.5 });
    expect(prediction.actions).toEqual([]);
  });

  it("returns an empty prediction for an empty dataset", () => {
    const backend = new RetrievalMovementPolicyBackend([]);
    const prediction = backend.predict({ goal: "anything" });
    expect(prediction.confidence).toBe(0);
    expect(prediction.actions).toEqual([]);
  });

  it("prefers successful demonstrations via the success boost", () => {
    const competing: MovementDemonstration[] = [
      {
        trajectoryId: "fail",
        sessionId: "s1",
        goalText: "save the document",
        actions: [{ tool: "keyboard.type", summary: "cmd+s", relativeTs: 0 }],
        outcomeStatus: "failure",
      },
      {
        trajectoryId: "ok",
        sessionId: "s2",
        goalText: "save the document",
        actions: [{ tool: "keyboard.type", summary: "cmd+s", relativeTs: 0 }],
        outcomeStatus: "success",
      },
    ];
    const backend = new RetrievalMovementPolicyBackend(competing);
    expect(backend.predict({ goal: "save the document" }).sourceTrajectoryId).toBe("ok");
  });
});

describe("MovementPolicyRegistry", () => {
  it("creates registered backends and rejects unknown kinds", () => {
    const registry = defaultMovementPolicyRegistry();
    expect(registry.kinds()).toContain("retrieval");
    expect(registry.create("retrieval", [])).toBeInstanceOf(RetrievalMovementPolicyBackend);
    expect(() => registry.create("mlx", [])).toThrow(/unknown movement policy backend/);
  });

  it("supports registering a custom pluggable backend", () => {
    const registry = new MovementPolicyRegistry().register("mock", () => ({
      id: "mock",
      predict: () => ({ backendId: "mock", confidence: 1, generalized: false, actions: [] }),
    }));
    expect(registry.create("mock", []).id).toBe("mock");
  });
});

describe("MovementInferenceService", () => {
  it("builds a policy from a reviewed export and predicts movements", () => {
    const manifest = manifestWith([
      replay({
        sessionId: "s1",
        trajectoryId: "t1",
        goal: "open the reports folder",
        actions: [{ tool: "mouse.click", summary: "double-click reports folder", ts: 1000 }],
      }),
    ]);
    const service = MovementInferenceService.fromExport(manifest);
    expect(service.datasetSize).toBe(1);
    expect(service.backendId).toBe("retrieval-v1");
    const prediction = service.predict({ goal: "open the invoices folder" });
    expect(prediction.generalized).toBe(true);
    expect(prediction.actions[0]!.summary).toBe("double-click invoices folder");
  });
});
