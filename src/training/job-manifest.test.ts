import { describe, expect, it } from "vitest";
import { createLocalTrainingJobManifest } from "./job-manifest.js";
import type { ReviewedExportManifest } from "./export-manifest.js";

const exportManifest: ReviewedExportManifest = {
  version: 1,
  createdAt: "2026-01-01T00:00:00.000Z",
  reviewedBy: "operator",
  purpose: "local fine-tuning",
  targetPlatform: "apple-silicon",
  modes: ["sft", "rl"],
  rawCaptureIncluded: false,
  promotedSkills: [
    {
      id: "skill-1",
      title: "Deploy workflow",
      summary: "Deploy workflow repaired and ready to reuse",
      sourceCandidateId: "candidate-1",
      sourceTrajectoryIds: ["traj-1"],
      promotedAt: "2026-01-01T00:10:00.000Z",
      version: 1,
    },
  ],
  memories: [
    {
      id: "memory-1",
      type: "session-summary",
      summary: "Deploy workflow repaired and ready to reuse",
      sourceSessionId: "sess-1",
      sourceTrajectoryId: "traj-1",
      tags: ["deploy"],
    },
  ],
  trajectories: [
    {
      id: "traj-1",
      sessionId: "sess-1",
      createdAt: "2026-01-01T00:00:00.000Z",
      captureTier: "operator",
      observationCount: 1,
      actionCount: 1,
      outcomeStatus: "success",
      reward: 1,
    },
  ],
  replays: [
    {
      sessionId: "sess-1",
      trajectoryIds: ["traj-1"],
      eventCount: 4,
      events: [
        { kind: "observation", ts: 1, trajectoryId: "traj-1", source: "browser", summary: "opened deploy" },
        { kind: "action", ts: 2, trajectoryId: "traj-1", tool: "browser", summary: "clicked deploy" },
        { kind: "transcript", ts: 3, messageId: "msg-1", role: "user", content: "check deploy" },
        { kind: "transcript", ts: 4, messageId: "msg-2", role: "assistant", content: "deploy checked" },
      ],
    },
  ],
};

describe("createLocalTrainingJobManifest", () => {
  it("builds an apple-silicon SFT job manifest from a reviewed export", () => {
    const manifest = createLocalTrainingJobManifest({
      id: "job-sft-1",
      exportManifest,
      mode: "sft",
    });

    expect(manifest).toMatchObject({
      id: "job-sft-1",
      mode: "sft",
      targetPlatform: "apple-silicon",
      status: "planned",
      dataset: {
        promotedSkillCount: 1,
        memoryCount: 1,
        trajectoryCount: 1,
        replayCount: 1,
      },
      runtime: {
        environment: "local-apple-silicon",
        reviewedExportRequired: true,
        rawCaptureAllowed: false,
      },
      config: {
        epochs: 3,
        learningRate: 1e-5,
        batchSize: 1,
      },
    });
  });

  it("builds an apple-silicon RL job manifest from a reviewed export", () => {
    const manifest = createLocalTrainingJobManifest({
      id: "job-rl-1",
      exportManifest,
      mode: "rl",
    });

    expect(manifest).toMatchObject({
      id: "job-rl-1",
      mode: "rl",
      targetPlatform: "apple-silicon",
      config: {
        algorithm: "grpo",
        rewardSource: "replay-manifest",
        rolloutCount: 8,
        klPenalty: 0.02,
      },
    });
  });
});
