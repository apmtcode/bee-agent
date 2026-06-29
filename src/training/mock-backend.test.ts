import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { MockMovementTrainingBackend, evaluateReplayFidelity, movementDatasetFromExport } from "./mock-backend.js";
import { deserializeMovementPolicy, replayMovement, type MovementTrajectory } from "./movement-policy.js";
import type { ReviewedExportManifest } from "./export-manifest.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "mock-backend-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })));
});

const trajectories: MovementTrajectory[] = [
  {
    id: "traj-1",
    events: [
      { kind: "observation", source: "browser", summary: "open the deploy dashboard" },
      { kind: "action", tool: "browser", summary: "click deploy" },
      { kind: "action", tool: "browser", summary: "confirm rollout" },
    ],
  },
];

describe("MockMovementTrainingBackend", () => {
  it("trains a policy in-process and writes a replayable artifact", async () => {
    const rootDir = await makeTempDir();
    const backend = new MockMovementTrainingBackend(rootDir);

    const result = await backend.train({
      jobId: "job-1",
      mode: "sft",
      artifactDir: "training-jobs/job-1/artifacts",
      trajectories,
    });

    expect(result.status).toBe("completed");
    expect(result.backendId).toBe("mock-movement");
    expect(result.modelPath).toBe("training-jobs/job-1/artifacts/movement-policy.json");
    expect(result.metrics).toMatchObject({
      trainedTrajectories: 1,
      trainedActions: 2,
      vocabularySize: 2,
      replayFidelity: 1,
    });

    const saved = await fs.readFile(path.join(rootDir, result.modelPath!), "utf8");
    const policy = deserializeMovementPolicy(saved);
    const replayed = replayMovement(policy, {
      observation: { source: "browser", summary: "open the deploy dashboard" },
    });
    expect(replayed).toEqual([
      { tool: "browser", summary: "click deploy" },
      { tool: "browser", summary: "confirm rollout" },
    ]);
  });

  it("honors the order hyperparameter", async () => {
    const rootDir = await makeTempDir();
    const backend = new MockMovementTrainingBackend(rootDir);
    const result = await backend.train({
      jobId: "job-order",
      mode: "sft",
      artifactDir: "artifacts",
      trajectories,
      hyperparameters: { order: 1 },
    });
    expect(result.status).toBe("completed");
    const policy = deserializeMovementPolicy(await fs.readFile(path.join(rootDir, result.modelPath!), "utf8"));
    expect(policy.order).toBe(1);
  });

  it("reports zero fidelity for an empty dataset", () => {
    expect(evaluateReplayFidelity({ version: 1, order: 3, vocabulary: [], tokenActions: {}, transitions: {}, observationCues: {}, trainedTrajectories: 0, trainedActions: 0 }, [])).toBe(0);
  });
});

describe("movementDatasetFromExport", () => {
  it("normalizes a reviewed export's replays into training trajectories", () => {
    const manifest: ReviewedExportManifest = {
      version: 1,
      createdAt: "2026-01-01T00:00:00.000Z",
      reviewedBy: "operator",
      purpose: "local fine-tuning",
      targetPlatform: "apple-silicon",
      modes: ["sft"],
      rawCaptureIncluded: false,
      executableSkills: [],
      executableSkillRuns: [],
      promotedSkills: [],
      memories: [],
      trajectories: [],
      replays: [
        {
          sessionId: "sess-1",
          trajectoryIds: ["traj-1"],
          eventCount: 2,
          events: [
            { kind: "observation", ts: 1, trajectoryId: "traj-1", source: "browser", summary: "open deploy" },
            { kind: "action", ts: 2, trajectoryId: "traj-1", tool: "browser", summary: "click deploy" },
          ],
        },
      ],
    };

    const dataset = movementDatasetFromExport(manifest);
    expect(dataset).toEqual([
      {
        id: "traj-1",
        events: [
          { kind: "observation", source: "browser", summary: "open deploy" },
          { kind: "action", tool: "browser", summary: "click deploy" },
        ],
      },
    ]);
  });
});
