import { randomUUID } from "node:crypto";
import { readJsonFile, writeJsonAtomic } from "../shared/fs.js";
import type { MemoryItem, MemoryItemType, PromotedSkill, SkillCandidate, SkillCandidateStatus } from "./types.js";

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
    const needle = query.trim();
    if (!needle) {
      return [];
    }
    const store = await this.load();
    return store.items.filter((item) => {
      if (includesNeedle(item.summary, needle)) {
        return true;
      }
      return item.tags?.some((tag) => includesNeedle(tag, needle)) ?? false;
    });
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
