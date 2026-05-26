export type TrainingMode = "sft" | "rl";

export type TrainingTargetPlatform = "apple-silicon";

export type ExportedTrajectoryReference = {
  id: string;
  sessionId: string;
  createdAt: string;
  captureTier: string;
  observationCount: number;
  actionCount: number;
  outcomeStatus?: "success" | "failure" | "aborted";
  reward?: number;
  reviewedAt?: string;
  reviewedBy?: string;
};

export type ExportedMemoryReference = {
  id: string;
  type: string;
  summary: string;
  sourceSessionId?: string;
  sourceTrajectoryId?: string;
  tags: string[];
};

export type ExportedPromotedSkill = {
  id: string;
  title: string;
  summary: string;
  sourceCandidateId?: string;
  sourceTrajectoryIds: string[];
  promotedAt: string;
  version: number;
  usage?: {
    executionCount: number;
    lastExecutedAt?: string;
  };
};

export type ExportedExecutableSkill = {
  id: string;
  promotedSkillId: string;
  title: string;
  summary: string;
  version: number;
  sourceTrajectoryIds: string[];
  usage: {
    executionCount: number;
    lastExecutedAt?: string;
  };
};

export type ExportedExecutableSkillRun = {
  id: string;
  skillId: string;
  sessionId?: string;
  parentRunId?: string;
  status: "completed";
  sourceTrajectoryIds: string[];
  replayPreview: Array<{
    kind: "transcript" | "observation" | "action";
    ts: number;
    summary: string;
  }>;
  stepResults: Array<{
    stepId: string;
    title: string;
    kind: "summary" | "x-post" | "command";
    output: string;
    agentRole?: string;
    subagentRunId?: string;
    commentOutputs?: string[];
  }>;
};

export type ExportedReplayManifest = {
  sessionId: string;
  trajectoryIds: string[];
  eventCount: number;
  events: Array<
    | {
        kind: "transcript";
        ts: number;
        messageId: string;
        role: "system" | "user" | "assistant" | "tool";
        content: string;
      }
    | {
        kind: "observation";
        ts: number;
        trajectoryId: string;
        source: string;
        summary: string;
      }
    | {
        kind: "action";
        ts: number;
        trajectoryId: string;
        tool: string;
        summary: string;
      }
  >;
};

export type ReviewedExportManifest = {
  version: 1;
  createdAt: string;
  reviewedBy: string;
  purpose: string;
  targetPlatform: TrainingTargetPlatform;
  modes: TrainingMode[];
  rawCaptureIncluded: false;
  promotedSkills: ExportedPromotedSkill[];
  executableSkills: ExportedExecutableSkill[];
  executableSkillRuns: ExportedExecutableSkillRun[];
  memories: ExportedMemoryReference[];
  trajectories: ExportedTrajectoryReference[];
  replays: ExportedReplayManifest[];
};
