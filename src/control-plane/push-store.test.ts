import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  FilePushStore,
  normalizeWebPushSubscriptionPayload,
} from "./push-store.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "operator-push-store-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true, maxRetries: 5, retryDelay: 50 })));
});

describe("FilePushStore", () => {
  it("creates and replaces a session-scoped subscription by endpoint", async () => {
    const store = new FilePushStore(path.join(await makeTempDir(), "push-subscriptions.json"));

    const created = await store.upsertSubscription({
      sessionId: "session-1",
      subscription: {
        endpoint: "https://push.example.test/sub-1",
        keys: { p256dh: "p256dh-1", auth: "auth-1" },
      },
      label: "Chrome",
      userAgent: "Browser/1.0",
    });
    const replaced = await store.upsertSubscription({
      sessionId: "session-1",
      subscription: {
        endpoint: "https://push.example.test/sub-1",
        keys: { p256dh: "p256dh-2", auth: "auth-2" },
      },
      label: "Chrome stable",
    });

    expect(replaced).toMatchObject({
      id: created.id,
      sessionId: "session-1",
      endpoint: "https://push.example.test/sub-1",
      keys: { p256dh: "p256dh-2", auth: "auth-2" },
      label: "Chrome stable",
    });
    expect(replaced.updatedAt >= created.updatedAt).toBe(true);

    await expect(store.listSubscriptions("session-1")).resolves.toEqual([
      expect.objectContaining({ id: created.id, sessionId: "session-1", label: "Chrome stable" }),
    ]);
  });

  it("lists by session and deletes by id or endpoint", async () => {
    const store = new FilePushStore(path.join(await makeTempDir(), "push-subscriptions.json"));
    const first = await store.upsertSubscription({
      sessionId: "session-1",
      subscription: {
        endpoint: "https://push.example.test/sub-1",
        keys: { p256dh: "p256dh-1", auth: "auth-1" },
      },
    });
    await store.upsertSubscription({
      sessionId: "session-2",
      subscription: {
        endpoint: "https://push.example.test/sub-2",
        keys: { p256dh: "p256dh-2", auth: "auth-2" },
      },
    });

    await expect(store.listSubscriptions("session-1")).resolves.toEqual([
      expect.objectContaining({ id: first.id, sessionId: "session-1" }),
    ]);
    await expect(store.deleteSubscription({ id: first.id })).resolves.toBe(true);
    await expect(store.listSubscriptions("session-1")).resolves.toEqual([]);
    await expect(store.deleteSubscription({ endpoint: "https://push.example.test/sub-2", sessionId: "session-2" })).resolves.toBe(true);
    await expect(store.listSubscriptions("session-2")).resolves.toEqual([]);
  });

  it("validates subscription payloads", () => {
    expect(normalizeWebPushSubscriptionPayload({
      endpoint: "https://push.example.test/sub-1",
      keys: { p256dh: "p256dh-1", auth: "auth-1" },
    })).toEqual({
      endpoint: "https://push.example.test/sub-1",
      keys: { p256dh: "p256dh-1", auth: "auth-1" },
    });
    expect(() => normalizeWebPushSubscriptionPayload({ endpoint: "http://push.example.test/sub-1", keys: { p256dh: "a", auth: "b" } })).toThrow("Invalid push subscription endpoint");
    expect(() => normalizeWebPushSubscriptionPayload({ endpoint: "https://push.example.test/sub-1", keys: { p256dh: "", auth: "b" } })).toThrow("Invalid push subscription p256dh");
  });
});
