import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { FileTaskStore } from "./task-store.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "task-store-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })));
});

describe("FileTaskStore", () => {
  it("creates, lists, and filters persisted tasks", async () => {
    const rootDir = await makeTempDir();
    const store = new FileTaskStore(path.join(rootDir, "tasks.json"));

    const first = await store.create({
      sessionId: "session-a",
      subject: "Inspect logs",
      description: "Look at the latest logs.",
      activeForm: "Inspecting logs",
      metadata: { priority: "high" },
      blocks: ["2"],
    });
    const second = await store.create({
      sessionId: "session-b",
      subject: "Ship patch",
      description: "Ship the patch.",
      blockedBy: [first.id],
    });

    await expect(store.list()).resolves.toMatchObject([
      {
        id: first.id,
        sessionId: "session-a",
        subject: "Inspect logs",
        description: "Look at the latest logs.",
        activeForm: "Inspecting logs",
        status: "pending",
        metadata: { priority: "high" },
        blocks: ["2"],
        blockedBy: [],
      },
      {
        id: second.id,
        sessionId: "session-b",
        subject: "Ship patch",
        description: "Ship the patch.",
        status: "pending",
        blocks: [],
        blockedBy: [first.id],
      },
    ]);
    await expect(store.listBySession("session-a")).resolves.toMatchObject([{ id: first.id }]);
    await expect(store.get(second.id)).resolves.toMatchObject({ id: second.id, subject: "Ship patch" });
  });

  it("updates task fields and merges metadata", async () => {
    const rootDir = await makeTempDir();
    const store = new FileTaskStore(path.join(rootDir, "tasks.json"));
    const created = await store.create({
      sessionId: "session-a",
      subject: "Inspect logs",
      description: "Look at the latest logs.",
      activeForm: "Inspecting logs",
      owner: "analyst",
      metadata: { priority: "high" },
      blockedBy: ["1"],
    });

    const updated = await store.update(created.id, {
      subject: "Inspect error logs",
      description: "Look at the latest error logs.",
      activeForm: null,
      owner: null,
      status: "in_progress",
      metadata: { lane: "ops" },
      blocks: ["3"],
      blockedBy: [],
    });

    expect(updated).toMatchObject({
      id: created.id,
      subject: "Inspect error logs",
      description: "Look at the latest error logs.",
      status: "in_progress",
      metadata: {
        priority: "high",
        lane: "ops",
      },
      blocks: ["3"],
      blockedBy: [],
    });
    expect(updated).not.toHaveProperty("activeForm");
    expect(updated).not.toHaveProperty("owner");
    expect(Date.parse(updated?.updatedAt ?? "")).toBeGreaterThanOrEqual(Date.parse(created.updatedAt));
    await expect(store.get(created.id)).resolves.toMatchObject({
      metadata: {
        priority: "high",
        lane: "ops",
      },
      blocks: ["3"],
      blockedBy: [],
      status: "in_progress",
    });
  });

  it("returns undefined for unknown tasks", async () => {
    const rootDir = await makeTempDir();
    const store = new FileTaskStore(path.join(rootDir, "tasks.json"));

    await expect(store.get("missing")).resolves.toBeUndefined();
    await expect(store.update("missing", { status: "completed" })).resolves.toBeUndefined();
  });
});
