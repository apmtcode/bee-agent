import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { FileCaptureConsentStore } from "./consent-store.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "capture-consent-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })));
});

describe("FileCaptureConsentStore", () => {
  it("creates, filters, and revokes consent grants", async () => {
    const dir = await makeTempDir();
    const store = new FileCaptureConsentStore(path.join(dir, "consent.json"));

    const expired = await store.create({
      tier: "app",
      purpose: "short-lived capture",
      expiresAt: "2026-01-01T01:00:00.000Z",
    });
    const active = await store.create({
      tier: "full",
      purpose: "visible supervised session",
      expiresAt: "2026-01-02T01:00:00.000Z",
    });

    await expect(store.list()).resolves.toHaveLength(2);
    await expect(store.listActive(new Date("2026-01-01T12:00:00.000Z"))).resolves.toEqual([
      expect.objectContaining({ id: active.id, tier: "full" }),
    ]);
    await expect(store.findActiveByTier("full", new Date("2026-01-01T12:00:00.000Z"))).resolves.toMatchObject({
      id: active.id,
      purpose: "visible supervised session",
    });

    const revoked = await store.revoke(active.id);
    expect(revoked).toMatchObject({ id: active.id });
    await expect(
      store.listActive(new Date(revoked?.revokedAt ?? "2100-01-01T00:00:00.000Z")),
    ).resolves.toEqual([]);
    await expect(store.get(expired.id)).resolves.toMatchObject({ id: expired.id, tier: "app" });
  });
});
