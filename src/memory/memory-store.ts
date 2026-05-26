import { randomUUID } from "node:crypto";
import { readJsonFile, writeJsonAtomic } from "../shared/fs.js";
import type {
  MemoryItem,
  MemoryItemType,
  MemoryRecallHit,
  PromotedSkill,
  RecallMatchReason,
  SkillCandidate,
  SkillCandidateStatus,
  SkillRecallHit,
} from "./types.js";

export type MemoryStoreShape = {
  version: 1;
  items: MemoryItem[];
  skillCandidates: SkillCandidate[];
  promotedSkills: PromotedSkill[];
};

const EMPTY_STORE: MemoryStoreShape = {
  version: 1,
  items: [],
  skillCandidates: [],
  promotedSkills: [],
};

function nowIso(): string {
  return new Date().toISOString();
}

function includesNeedle(value: string, needle: string): boolean {
  return value.toLowerCase().includes(needle.toLowerCase());
}

function scoreMatch(value: string | undefined, needle: string): number {
  if (!value) {
    return 0;
  }
  const haystack = value.toLowerCase();
  const query = needle.toLowerCase();
  if (haystack === query) {
    return 6;
  }
  if (haystack.startsWith(query)) {
    return 4;
  }
  return haystack.includes(query) ? 2 : 0;
}

function pushReason(reasons: RecallMatchReason[], reason: RecallMatchReason, score: number): number {
  if (score > 0 && !reasons.includes(reason)) {
    reasons.push(reason);
  }
  return score;
}

export class FileMemoryStore {
  constructor(private readonly filePath: string) {}

  async load(): Promise<MemoryStoreShape> {
    return await readJsonFile(this.filePath, EMPTY_STORE);
  }

  async save(store: MemoryStoreShape): Promise<void> {
    await writeJsonAtomic(this.filePath, store);
  }

  async addItem(params: {
    type: MemoryItemType;
    summary: string;
    sourceSessionId?: string;
    sourceTrajectoryId?: string;
    tags?: string[];
    metadata?: Record<string, unknown>;
  }): Promise<MemoryItem> {
    const store = await this.load();
    const item: MemoryItem = {
      id: randomUUID(),
      type: params.type,
      summary: params.summary,
      createdAt: nowIso(),
      updatedAt: nowIso(),
      ...(params.sourceSessionId ? { sourceSessionId: params.sourceSessionId } : {}),
      ...(params.sourceTrajectoryId ? { sourceTrajectoryId: params.sourceTrajectoryId } : {}),
      ...(params.tags ? { tags: [...params.tags] } : {}),
      ...(params.metadata ? { metadata: params.metadata } : {}),
    };
    store.items.push(item);
    await this.save(store);
    return item;
  }

  async search(query: string): Promise<MemoryItem[]> {
    const results = await this.searchDetailed(query);
    return results.map((hit) => hit.item);
  }

  async searchDetailed(query: string): Promise<MemoryRecallHit[]> {
    const needle = query.trim();
    if (!needle) {
      return [];
    }
    const store = await this.load();
    return store.items
      .map((item) => {
        const reasons: RecallMatchReason[] = [];
        let score = 0;
        score += pushReason(reasons, "summary", scoreMatch(item.summary, needle));
        for (const tag of item.tags ?? []) {
          score += pushReason(reasons, "tag", scoreMatch(tag, needle));
        }
        if (score === 0) {
          return null;
        }
        return {
          kind: "memory",
          score,
          reasons,
          item,
        } satisfies MemoryRecallHit;
      })
      .filter((hit): hit is MemoryRecallHit => hit !== null)
      .sort((a, b) => b.score - a.score || b.item.updatedAt.localeCompare(a.item.updatedAt));
  }

  async searchPromotedSkills(query: string): Promise<SkillRecallHit[]> {
    const needle = query.trim();
    if (!needle) {
      return [];
    }
    const store = await this.load();
    return store.promotedSkills
      .map((skill) => {
        const reasons: RecallMatchReason[] = [];
        let score = 0;
        score += pushReason(reasons, "title", scoreMatch(skill.title, needle));
        score += pushReason(reasons, "summary", scoreMatch(skill.summary, needle));
        if (score === 0) {
          return null;
        }
        return {
          kind: "skill",
          score,
          reasons,
          skill,
        } satisfies SkillRecallHit;
      })
      .filter((hit): hit is SkillRecallHit => hit !== null)
      .sort((a, b) => b.score - a.score || b.skill.promotedAt.localeCompare(a.skill.promotedAt));
  }

  async listSkillCandidates(status?: SkillCandidateStatus): Promise<SkillCandidate[]> {
    const store = await this.load();
    const items = status ? store.skillCandidates.filter((candidate) => candidate.status === status) : store.skillCandidates;
    return [...items].sort((a, b) => a.createdAt.localeCompare(b.createdAt));
  }

  async addSkillCandidate(params: {
    title: string;
    summary: string;
    sourceTrajectoryIds: string[];
    confidence: number;
  }): Promise<SkillCandidate> {
    const store = await this.load();
    const timestamp = nowIso();
    const candidate: SkillCandidate = {
      id: randomUUID(),
      title: params.title,
      summary: params.summary,
      sourceTrajectoryIds: [...params.sourceTrajectoryIds],
      createdAt: timestamp,
      updatedAt: timestamp,
      confidence: params.confidence,
      status: "candidate",
    };
    store.skillCandidates.push(candidate);
    await this.save(store);
    return candidate;
  }

  async reviewSkillCandidate(
    candidateId: string,
    status: SkillCandidateStatus,
    reviewNote?: string,
  ): Promise<SkillCandidate | null> {
    const store = await this.load();
    const candidate = store.skillCandidates.find((entry) => entry.id === candidateId);
    if (!candidate) {
      return null;
    }
    const timestamp = nowIso();
    const reviewedAt = status === "candidate" ? undefined : timestamp;
    candidate.status = status;
    candidate.updatedAt = timestamp;
    if (reviewedAt) {
      candidate.reviewedAt = reviewedAt;
    }
    if (reviewNote) {
      candidate.reviewNote = reviewNote;
    }
    await this.save(store);
    return candidate;
  }

  async listPromotedSkills(): Promise<PromotedSkill[]> {
    const store = await this.load();
    return [...store.promotedSkills].sort((a, b) => a.promotedAt.localeCompare(b.promotedAt));
  }

  async promoteSkill(candidateId: string): Promise<PromotedSkill | null> {
    const store = await this.load();
    const candidate = store.skillCandidates.find((entry) => entry.id === candidateId);
    if (!candidate || candidate.status === "rejected") {
      return null;
    }
    if (candidate.promotedSkillId) {
      return store.promotedSkills.find((skill) => skill.id === candidate.promotedSkillId) ?? null;
    }
    const timestamp = nowIso();
    candidate.status = "reviewed";
    candidate.updatedAt = timestamp;
    candidate.reviewedAt = timestamp;
    const skill: PromotedSkill = {
      id: randomUUID(),
      title: candidate.title,
      summary: candidate.summary,
      sourceTrajectoryIds: [...candidate.sourceTrajectoryIds],
      promotedAt: timestamp,
      version: 1,
      sourceCandidateId: candidate.id,
    };
    candidate.promotedSkillId = skill.id;
    store.promotedSkills.push(skill);
    await this.save(store);
    return skill;
  }
}
