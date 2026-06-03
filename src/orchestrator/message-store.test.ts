import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { FileMessageStore } from "./message-store.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "message-store-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })));
});

describe("FileMessageStore", () => {
  it("sends, lists, and filters persisted messages", async () => {
    const rootDir = await makeTempDir();
    const store = new FileMessageStore(path.join(rootDir, "messages.json"));

    const first = await store.send({
      fromSessionId: "session-a",
      toSessionId: "session-b",
      summary: "Need logs",
      message: "Please collect the latest logs.",
      metadata: { priority: "high" },
    });
    const second = await store.send({
      fromSessionId: "session-b",
      toSessionId: "session-a",
      message: "Collected and attached.",
    });

    await expect(store.list()).resolves.toMatchObject([
      {
        id: first.id,
        fromSessionId: "session-a",
        toSessionId: "session-b",
        summary: "Need logs",
        message: "Please collect the latest logs.",
        metadata: { priority: "high" },
      },
      {
        id: second.id,
        fromSessionId: "session-b",
        toSessionId: "session-a",
        summary: "Collected and attached.",
        message: "Collected and attached.",
      },
    ]);
    await expect(store.listBySession("session-a")).resolves.toMatchObject([{ id: first.id }, { id: second.id }]);
    await expect(store.listInbox("session-a")).resolves.toMatchObject([{ id: second.id, toSessionId: "session-a" }]);
    await expect(store.listOutbox("session-a")).resolves.toMatchObject([{ id: first.id, fromSessionId: "session-a" }]);
    await expect(store.get(first.id)).resolves.toMatchObject({ id: first.id, summary: "Need logs" });
  });

  it("builds a summary from the message when omitted", async () => {
    const rootDir = await makeTempDir();
    const store = new FileMessageStore(path.join(rootDir, "messages.json"));

    const created = await store.send({
      fromSessionId: "session-a",
      toSessionId: "session-b",
      message: "   This is a long message that should be normalized into a compact summary line for inbox views.   ",
    });

    expect(created.summary).toBe("This is a long message that should be normalized into a compact summary line...");
    expect(created.message).toBe("This is a long message that should be normalized into a compact summary line for inbox views.");
  });

  it("returns undefined for unknown messages", async () => {
    const rootDir = await makeTempDir();
    const store = new FileMessageStore(path.join(rootDir, "messages.json"));

    await expect(store.get("missing")).resolves.toBeUndefined();
  });
});
