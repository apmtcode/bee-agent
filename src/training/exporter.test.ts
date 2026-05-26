import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { randomUUID } from "node:crypto";
import { afterEach, describe, expect, it } from "vitest";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import { FileTrajectoryStore } from "../capture/trajectory-store.js";
import { FileMemoryStore } from "../memory/memory-store.js";
import { LocalTrainingExporter } from "./exporter.js";
import type { SessionRecord } from "../harness/types.js";
import { JsonlTranscriptStore } from "../harness/transcript-store.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "training-export-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })));
});

describe("LocalTrainingExporter", () => {
  it("exports only reviewed promoted skills and trajectory metadata", async () => {
    const dir = await makeTempDir();
    const memory = new FileMemoryStore(path.join(dir, "memory.json"));
    const trajectories = new FileTrajectoryStore(path.join(dir, "trajectories.json"));
    const sessionId = "sess-1";
    const transcriptFile = path.join(dir, "transcripts", `${sessionId}.jsonl`);
    const transcript = new JsonlTranscriptStore(transcriptFile);
    await transcript.ensureHeader({ sessionId });
    await transcript.appendMessage({ role: "user", content: "check deploy", timestamp: 10 });
    await transcript.appendMessage({ role: "assistant", content: "deploy checked", timestamp: 30 });
    const sessions = new Map<string, SessionRecord>([
      [
        sessionId,
        {
          id: sessionId,
          createdAt: "2026-01-01T00:00:00.000Z",
          updatedAt: "2026-01-01T00:00:00.000Z",
          status: "active",
          transcriptFile,
          metadata: { title: "Deploy" },
        },
      ],
    ]);
    const exporter = new LocalTrainingExporter({
      memory,
      trajectories,
      resolveSession: async (lookupSessionId) => sessions.get(lookupSessionId),
    });

    const includedTrajectory = buildTrajectorySpan({
      id: randomUUID(),
      sessionId,
      observations: [{ kind: "observation", source: "operator", summary: "observed deploy", ts: 1 }],
      actions: [{ kind: "action", tool: "assistant-response", summary: "fixed deploy", ts: 2 }],
      outcome: { status: "success", summary: "deploy fixed", reward: 1 },
    });
    const excludedTrajectory = buildTrajectorySpan({
      id: randomUUID(),
      sessionId: "sess-2",
      observations: [{ kind: "observation", source: "operator", summary: "extra", ts: 3 }],
      actions: [{ kind: "action", tool: "assistant-response", summary: "extra", ts: 4 }],
      outcome: { status: "failure", summary: "not exported" },
    });
    await trajectories.add(includedTrajectory);
    await trajectories.add(excludedTrajectory);
    await trajectories.review(includedTrajectory.id, {
      status: "approved",
      reviewedBy: "reviewer",
      redactedObservations: [{ ts: 11, source: "operator", summary: "reviewed observation" }],
      redactedActions: [{ ts: 12, tool: "assistant-response", summary: "reviewed action" }],
      redactedTranscript: [
        { ts: 13, role: "user", content: "reviewed user" },
        { ts: 14, role: "assistant", content: "reviewed assistant" },
      ],
    });
    await trajectories.review(excludedTrajectory.id, {
      status: "rejected",
      reviewedBy: "reviewer",
      reviewNote: "unsafe",
    });

    await memory.addItem({
      type: "session-summary",
      summary: "Deploy workflow repaired and ready to reuse",
      sourceSessionId: sessionId,
      sourceTrajectoryId: includedTrajectory.id,
      tags: ["deploy"],
    });

    const includedCandidate = await memory.addSkillCandidate({
      title: "Deploy workflow",
      summary: "Deploy workflow repaired and ready to reuse",
      sourceTrajectoryIds: [includedTrajectory.id],
      confidence: 0.7,
    });
    await memory.reviewSkillCandidate(includedCandidate.id, "reviewed", "approved for export");
    await memory.promoteSkill(includedCandidate.id);

    const rejectedCandidate = await memory.addSkillCandidate({
      title: "Unsafe workflow",
      summary: "Should not export",
      sourceTrajectoryIds: [excludedTrajectory.id],
      confidence: 0.1,
    });
    await memory.reviewSkillCandidate(rejectedCandidate.id, "rejected", "unsafe");

    const outputFile = path.join(dir, "export.json");
    const manifest = await exporter.createReviewedExport({
      reviewedBy: "operator",
      purpose: "local fine-tuning",
      outputFile,
      modes: ["sft", "rl"],
    });

    expect(manifest.rawCaptureIncluded).toBe(false);
    expect(manifest.targetPlatform).toBe("apple-silicon");
    expect(manifest.modes).toEqual(["sft", "rl"]);
    expect(manifest.promotedSkills).toHaveLength(1);
    expect(manifest.promotedSkills[0]?.sourceCandidateId).toBe(includedCandidate.id);
    expect(manifest.replays).toEqual([
      expect.objectContaining({
        sessionId,
        trajectoryIds: [includedTrajectory.id],
        eventCount: 4,
      }),
    ]);
    expect(manifest.trajectories).toEqual([
      expect.objectContaining({
        id: includedTrajectory.id,
        observationCount: 1,
        actionCount: 1,
        outcomeStatus: "success",
        reward: 1,
        reviewedBy: "reviewer",
        reviewedAt: expect.any(String),
      }),
    ]);
    expect(JSON.stringify(manifest)).not.toContain("Should not export");
    expect(JSON.stringify(manifest)).not.toContain("opened deploy page");
    expect(JSON.stringify(manifest)).not.toContain("check deploy");
    expect(JSON.stringify(manifest)).not.toContain("deploy checked");
    expect(manifest.replays[0]?.events).toEqual([
      { kind: "observation", ts: 11, trajectoryId: includedTrajectory.id, source: "operator", summary: "reviewed observation" },
      { kind: "action", ts: 12, trajectoryId: includedTrajectory.id, tool: "assistant-response", summary: "reviewed action" },
      { kind: "transcript", ts: 13, messageId: expect.any(String), role: "user", content: "reviewed user" },
      { kind: "transcript", ts: 14, messageId: expect.any(String), role: "assistant", content: "reviewed assistant" },
    ]);
    expect(await fs.readFile(outputFile, "utf8")).toContain("apple-silicon");
  });
});
