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
