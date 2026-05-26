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

export type PromotedSkill = {
  id: string;
  title: string;
  summary: string;
  sourceTrajectoryIds: string[];
  promotedAt: string;
  version: number;
  sourceCandidateId?: string;
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
