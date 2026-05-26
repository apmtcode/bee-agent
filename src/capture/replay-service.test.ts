import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { FileSessionStore } from "../harness/session-store.js";
import { JsonlTranscriptStore } from "../harness/transcript-store.js";
import { ReplayRuntimeService } from "./replay-service.js";
import { FileTrajectoryStore } from "./trajectory-store.js";
import { buildTrajectorySpan } from "./trajectory.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "replay-runtime-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })));
});

describe("ReplayRuntimeService", () => {
  it("reconstructs a replay manifest for a stored session", async () => {
    const rootDir = await makeTempDir();
    const sessionStore = new FileSessionStore(path.join(rootDir, "sessions.json"));
    const trajectoryStore = new FileTrajectoryStore(path.join(rootDir, "trajectories.json"));
    const service = new ReplayRuntimeService(sessionStore, trajectoryStore);
    const transcriptFile = path.join(rootDir, "transcripts", "sess-1.jsonl");
    const transcript = new JsonlTranscriptStore(transcriptFile);

    await transcript.ensureHeader({ sessionId: "sess-1", cwd: "/tmp/repo" });
    await transcript.appendMessage({ role: "user", content: "check deploy", timestamp: 10, messageId: "msg-1" });
    await transcript.appendMessage({ role: "assistant", content: "deploy checked", timestamp: 30, messageId: "msg-2" });
    await sessionStore.upsert({
      id: "sess-1",
      transcriptFile,
      metadata: { title: "Replay" },
    });
    await trajectoryStore.add(
      buildTrajectorySpan({
        id: "traj-1",
        sessionId: "sess-1",
        observations: [{ kind: "observation", source: "browser", summary: "opened deploy", ts: 20 }],
        actions: [{ kind: "action", tool: "browser", summary: "clicked deploy", ts: 40 }],
        outcome: { status: "success", summary: "deploy repaired", reward: 1 },
      }),
    );

    await expect(service.getSessionReplay("sess-1")).resolves.toMatchObject({
      sessionId: "sess-1",
      trajectoryIds: ["traj-1"],
      eventCount: 4,
      events: [
        { kind: "transcript", ts: 10, messageId: "msg-1", role: "user", content: "check deploy" },
        { kind: "observation", ts: 20, trajectoryId: "traj-1", source: "browser", summary: "opened deploy" },
        { kind: "transcript", ts: 30, messageId: "msg-2", role: "assistant", content: "deploy checked" },
        { kind: "action", ts: 40, trajectoryId: "traj-1", tool: "browser", summary: "clicked deploy" },
      ],
    });
  });

  it("lists replays for sessions with persisted data", async () => {
    const rootDir = await makeTempDir();
    const sessionStore = new FileSessionStore(path.join(rootDir, "sessions.json"));
    const trajectoryStore = new FileTrajectoryStore(path.join(rootDir, "trajectories.json"));
    const service = new ReplayRuntimeService(sessionStore, trajectoryStore);
    const transcriptFile = path.join(rootDir, "transcripts", "sess-2.jsonl");
    const transcript = new JsonlTranscriptStore(transcriptFile);

    await transcript.ensureHeader({ sessionId: "sess-2" });
    await transcript.appendMessage({ role: "user", content: "open logs", timestamp: 5, messageId: "msg-1" });
    await sessionStore.upsert({ id: "sess-2", transcriptFile });

    await expect(service.listSessionReplays()).resolves.toEqual([
      expect.objectContaining({ sessionId: "sess-2", eventCount: 1 }),
    ]);
  });
});
