import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { FileSessionStore } from "./session-store.js";
import { JsonlTranscriptStore } from "./transcript-store.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "operator-core-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })));
});

describe("FileSessionStore", () => {
  it("creates, updates, and lists sessions", async () => {
    const dir = await makeTempDir();
    const store = new FileSessionStore(path.join(dir, "sessions.json"));

    const session = await store.upsert({
      metadata: { title: "Main session", cwd: dir },
      transcriptFile: path.join(dir, "main.jsonl"),
    });

    expect(session.status).toBe("active");
    expect(session.metadata.title).toBe("Main session");
    await expect(store.get(session.id)).resolves.toEqual(session);

    const updated = await store.update(session.id, {
      status: "idle",
      metadata: { agentId: "main" },
    });

    expect(updated).toMatchObject({
      id: session.id,
      status: "idle",
      metadata: { title: "Main session", cwd: dir, agentId: "main" },
    });
    await expect(store.list()).resolves.toEqual([updated]);
  });
});

describe("JsonlTranscriptStore", () => {
  it("writes a header, appends messages, and supports bounded reads", async () => {
    const dir = await makeTempDir();
    const transcript = new JsonlTranscriptStore(path.join(dir, "session.jsonl"));

    await transcript.ensureHeader({ sessionId: "sess-1", cwd: dir });
    await transcript.appendMessage({ role: "user", content: "hello" });
    await transcript.appendMessage({ role: "assistant", content: "hi there" });
    await transcript.appendMessage({ role: "assistant", content: "done" });

    const records = await transcript.readAll();
    expect(records).toHaveLength(4);

    await expect(transcript.readLatestAssistantText()).resolves.toBe("done");
    await expect(transcript.read({ includeHeader: false, limit: 2 })).resolves.toHaveLength(2);
    await expect(transcript.readMessages({ reverse: true, limit: 2 })).resolves.toMatchObject([
      { message: { content: "done", role: "assistant" } },
      { message: { content: "hi there", role: "assistant" } },
    ]);
    await expect(transcript.readTail(2)).resolves.toMatchObject([
      { message: { content: "hi there", role: "assistant" } },
      { message: { content: "done", role: "assistant" } },
    ]);
  });
});
