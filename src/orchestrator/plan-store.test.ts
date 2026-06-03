import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { FilePlanStore } from "./plan-store.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "plan-store-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })));
});

describe("FilePlanStore", () => {
  it("creates, lists, and looks up plans by session", async () => {
    const rootDir = await makeTempDir();
    const store = new FilePlanStore(path.join(rootDir, "plans.json"));

    const first = await store.upsert({
      sessionId: "session-a",
      content: "1. Inspect logs\n2. Ship patch",
      verification: { status: "pending" },
    });
    const second = await store.upsert({
      sessionId: "session-b",
      content: "1. Review replay",
      status: "awaiting_approval",
      approval: {
        requestId: "req-1",
        requestedBySessionId: "session-a",
        approverSessionId: "session-reviewer",
        requestedAt: "2026-06-03T00:00:00.000Z",
      },
    });

    await expect(store.list()).resolves.toMatchObject([
      {
        id: first.id,
        sessionId: "session-a",
        content: "1. Inspect logs\n2. Ship patch",
        status: "draft",
        verification: { status: "pending" },
      },
      {
        id: second.id,
        sessionId: "session-b",
        content: "1. Review replay",
        status: "awaiting_approval",
        approval: {
          requestId: "req-1",
          requestedBySessionId: "session-a",
          requestedAt: "2026-06-03T00:00:00.000Z",
        },
      },
    ]);
    await expect(store.get(first.id)).resolves.toMatchObject({ id: first.id, sessionId: "session-a" });
    await expect(store.getBySession("session-b")).resolves.toMatchObject({ id: second.id, status: "awaiting_approval" });
  });

  it("upserts the same session and updates approval and verification state", async () => {
    const rootDir = await makeTempDir();
    const store = new FilePlanStore(path.join(rootDir, "plans.json"));

    const created = await store.upsert({
      sessionId: "session-a",
      content: "Initial draft",
      verification: { status: "pending" },
    });
    const updated = await store.upsert({
      sessionId: "session-a",
      content: "Approved draft",
      status: "approved",
      approval: {
        requestId: "req-2",
        requestedBySessionId: "session-a",
        approverSessionId: "reviewer",
        requestedAt: "2026-06-03T01:00:00.000Z",
        resolvedAt: "2026-06-03T01:05:00.000Z",
        approved: true,
        feedback: "Looks good",
      },
      verification: {
        status: "requested",
        reminderAt: "2026-06-03T02:00:00.000Z",
        notes: "Verify after tests",
      },
    });

    expect(updated).toMatchObject({
      id: created.id,
      sessionId: "session-a",
      content: "Approved draft",
      status: "approved",
      approval: {
        requestId: "req-2",
        approved: true,
        feedback: "Looks good",
      },
      verification: {
        status: "requested",
        reminderAt: "2026-06-03T02:00:00.000Z",
        notes: "Verify after tests",
      },
    });
    expect(Date.parse(updated.updatedAt)).toBeGreaterThanOrEqual(Date.parse(created.updatedAt));
    await expect(store.list()).resolves.toHaveLength(1);
  });

  it("clears approval and verification fields and returns undefined for unknown plans", async () => {
    const rootDir = await makeTempDir();
    const store = new FilePlanStore(path.join(rootDir, "plans.json"));

    const created = await store.upsert({
      sessionId: "session-a",
      content: "Draft",
      approval: {
        requestId: "req-3",
        requestedBySessionId: "session-a",
        approverSessionId: "reviewer",
        requestedAt: "2026-06-03T01:00:00.000Z",
      },
      verification: { status: "pending", notes: "Soon" },
    });
    const updated = await store.update(created.id, {
      status: "rejected",
      approval: null,
      verification: null,
    });

    expect(updated).toMatchObject({
      id: created.id,
      status: "rejected",
    });
    expect(updated).not.toHaveProperty("approval");
    expect(updated).not.toHaveProperty("verification");
    await expect(store.get("missing")).resolves.toBeUndefined();
    await expect(store.update("missing", { status: "completed" })).resolves.toBeUndefined();
  });
});
