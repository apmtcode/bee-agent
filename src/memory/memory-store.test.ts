import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { FileMemoryStore } from "./memory-store.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "memory-store-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })));
});

describe("FileMemoryStore", () => {
  it("reviews, promotes, searches, and lists skill candidates", async () => {
    const dir = await makeTempDir();
    const store = new FileMemoryStore(path.join(dir, "memory.json"));

    await store.addItem({
      type: "session-summary",
      summary: "Deploy workflow repaired and ready to reuse",
      sourceSessionId: "sess-1",
      sourceTrajectoryId: "traj-1",
      tags: ["deploy", "release"],
    });
    await store.addItem({
      type: "preference",
      summary: "Prefer short release updates",
      tags: ["status"],
    });

    const candidate = await store.addSkillCandidate({
      title: "Deploy workflow",
      summary: "Deploy workflow repaired and ready to reuse",
      sourceTrajectoryIds: ["traj-1"],
      confidence: 0.6,
    });

    await expect(store.listSkillCandidates("candidate")).resolves.toEqual([
      expect.objectContaining({ id: candidate.id, status: "candidate" }),
    ]);
    await expect(store.reviewSkillCandidate(candidate.id, "reviewed", "operator approved")).resolves.toMatchObject({
      id: candidate.id,
      status: "reviewed",
      reviewNote: "operator approved",
    });

    const promoted = await store.promoteSkill(candidate.id);
    expect(promoted).toMatchObject({ sourceCandidateId: candidate.id, sourceTrajectoryIds: ["traj-1"] });
    await expect(store.listPromotedSkills()).resolves.toEqual([
      expect.objectContaining({ id: promoted?.id, sourceCandidateId: candidate.id }),
    ]);
    await expect(store.promoteSkill(candidate.id)).resolves.toMatchObject({ id: promoted?.id });
    await expect(store.searchDetailed("deploy")).resolves.toEqual([
      expect.objectContaining({ kind: "memory", item: expect.objectContaining({ sourceSessionId: "sess-1" }) }),
    ]);
    await expect(store.searchPromotedSkills("deploy")).resolves.toEqual([
      expect.objectContaining({ kind: "skill", skill: expect.objectContaining({ id: promoted?.id, title: "Deploy workflow" }) }),
    ]);
  });

  it("does not promote rejected candidates", async () => {
    const dir = await makeTempDir();
    const store = new FileMemoryStore(path.join(dir, "memory.json"));

    const candidate = await store.addSkillCandidate({
      title: "Discarded workflow",
      summary: "This should not become a promoted skill",
      sourceTrajectoryIds: ["traj-2"],
      confidence: 0.2,
    });

    await expect(store.reviewSkillCandidate(candidate.id, "rejected", "unsafe" )).resolves.toMatchObject({
      id: candidate.id,
      status: "rejected",
    });
    await expect(store.promoteSkill(candidate.id)).resolves.toBeNull();
    await expect(store.listPromotedSkills()).resolves.toEqual([]);
  });
});
