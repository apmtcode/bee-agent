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
  it("reviews, promotes, stores executable skills, and records usage", async () => {
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
    expect(promoted).toMatchObject({ sourceCandidateId: candidate.id, sourceTrajectoryIds: ["traj-1"], usage: { executionCount: 0 } });
    if (!promoted) {
      throw new Error("expected promoted skill");
    }
    const executableSkill = await store.createExecutableSkill({
      promotedSkillId: promoted.id,
      steps: [
        { id: "summary", title: "Summarize", kind: "summary", content: "deploy summary" },
        { id: "x-post", title: "Post", kind: "x-post", content: "deploy post", maxCharacters: 280, overflowToComment: true },
      ],
    });
    expect(executableSkill).toMatchObject({ promotedSkillId: promoted.id, usage: { executionCount: 0 } });
    if (!executableSkill) {
      throw new Error("expected executable skill" );
    }
    await store.addExecutableSkillRun({
      id: "run-1",
      skillId: executableSkill.id,
      sessionId: "sess-1",
      startedAt: "2026-01-01T00:00:00.000Z",
      completedAt: "2026-01-01T00:01:00.000Z",
      status: "completed",
      sourceTrajectoryIds: ["traj-1"],
      replayPreview: [{ kind: "observation", ts: 1, summary: "preview" }],
      stepResults: [
        {
          stepId: "summary",
          title: "Summarize",
          kind: "summary",
          status: "completed",
          output: "deploy summary",
          startedAt: "2026-01-01T00:00:00.000Z",
          completedAt: "2026-01-01T00:00:10.000Z",
        },
      ],
    });

    await expect(store.listPromotedSkills()).resolves.toEqual([
      expect.objectContaining({ id: promoted.id, sourceCandidateId: candidate.id, usage: expect.objectContaining({ executionCount: 1 }) }),
    ]);
    await expect(store.listExecutableSkills()).resolves.toEqual([
      expect.objectContaining({ id: executableSkill.id, promotedSkillId: promoted.id, usage: expect.objectContaining({ executionCount: 1 }) }),
    ]);
    await expect(store.listExecutableSkillRuns(executableSkill.id)).resolves.toEqual([
      expect.objectContaining({ id: "run-1", skillId: executableSkill.id }),
    ]);
    await expect(store.promoteSkill(candidate.id)).resolves.toMatchObject({ id: promoted.id });
    await expect(store.searchDetailed("deploy")).resolves.toEqual([
      expect.objectContaining({ kind: "memory", item: expect.objectContaining({ sourceSessionId: "sess-1" }) }),
    ]);
    await expect(store.searchPromotedSkills("deploy")).resolves.toEqual([
      expect.objectContaining({
        kind: "skill",
        skill: expect.objectContaining({ id: promoted.id, title: "Deploy workflow" }),
        executableSkill: expect.objectContaining({ id: executableSkill.id }),
      }),
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
