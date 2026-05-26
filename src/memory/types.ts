export type MemoryItemType =
  | "episodic"
  | "semantic"
  | "preference"
  | "procedure"
  | "session-summary"
  | "trajectory-summary";

export type MemoryItem = {
  id: string;
  type: MemoryItemType;
  summary: string;
  createdAt: string;
  updatedAt: string;
  sourceSessionId?: string;
  sourceTrajectoryId?: string;
  tags?: string[];
  metadata?: Record<string, unknown>;
};

export type SkillCandidateStatus = "candidate" | "reviewed" | "rejected";

export type SkillCandidate = {
  id: string;
  title: string;
  summary: string;
  sourceTrajectoryIds: string[];
  createdAt: string;
  updatedAt: string;
  confidence: number;
  status: SkillCandidateStatus;
  reviewedAt?: string;
  reviewNote?: string;
  promotedSkillId?: string;
};

export type SkillUsageStats = {
  executionCount: number;
  lastExecutedAt?: string;
};

export type PromotedSkill = {
  id: string;
  title: string;
  summary: string;
  sourceTrajectoryIds: string[];
  promotedAt: string;
  version: number;
  sourceCandidateId?: string;
  usage?: SkillUsageStats;
};

export type ExecutableSkillStepKind = "summary" | "x-post" | "command";

export type ExecutableSkillStep = {
  id: string;
  title: string;
  kind: ExecutableSkillStepKind;
  content?: string;
  command?: string;
  agentRole?: string;
  maxCharacters?: number;
  overflowToComment?: boolean;
};

export type ExecutableSkill = {
  id: string;
  promotedSkillId: string;
  title: string;
  summary: string;
  version: number;
  createdAt: string;
  updatedAt: string;
  sourceTrajectoryIds: string[];
  steps: ExecutableSkillStep[];
  usage: SkillUsageStats;
};

export type ExecutableSkillStepResult = {
  stepId: string;
  title: string;
  kind: ExecutableSkillStepKind;
  status: "completed";
  output: string;
  startedAt: string;
  completedAt: string;
  agentRole?: string;
  subagentRunId?: string;
  commentOutputs?: string[];
};

export type ExecutableSkillRun = {
  id: string;
  skillId: string;
  sessionId?: string;
  parentRunId?: string;
  startedAt: string;
  completedAt: string;
  status: "completed";
  sourceTrajectoryIds: string[];
  replayPreview: RecallReplayPreviewEvent[];
  stepResults: ExecutableSkillStepResult[];
};

export type RecallHitKind = "memory" | "skill" | "trajectory";

export type RecallMatchReason = "summary" | "title" | "tag" | "outcome" | "review" | "observation" | "action" | "transcript";

export type RecallReplayPreviewEvent = {
  kind: "transcript" | "observation" | "action";
  ts: number;
  summary: string;
};

export type MemoryRecallHit = {
  kind: "memory";
  score: number;
  reasons: RecallMatchReason[];
  item: MemoryItem;
};

export type SkillRecallHit = {
  kind: "skill";
  score: number;
  reasons: RecallMatchReason[];
  skill: PromotedSkill;
  executableSkill?: ExecutableSkill;
};

export type TrajectoryRecallHit = {
  kind: "trajectory";
  score: number;
  reasons: RecallMatchReason[];
  trajectoryId: string;
  sessionId: string;
  createdAt: string;
  captureTier: string;
  outcomeSummary?: string;
  outcomeStatus?: "success" | "failure" | "aborted";
  reward?: number;
  reviewStatus: "approved";
  reviewedAt?: string;
  reviewedBy?: string;
  summary: string;
  preview: RecallReplayPreviewEvent[];
  sourceObservationCount: number;
  sourceActionCount: number;
};

export type RecallHit = MemoryRecallHit | SkillRecallHit | TrajectoryRecallHit;

export type RecallSummary = {
  query: string;
  hits: RecallHit[];
  memories: MemoryRecallHit[];
  skills: SkillRecallHit[];
  trajectories: TrajectoryRecallHit[];
};
