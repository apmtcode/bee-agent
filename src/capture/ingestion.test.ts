import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { FileCaptureConsentStore } from "./consent-store.js";
import { CaptureIngestionService } from "./ingestion.js";
import { FileTrajectoryStore } from "./trajectory-store.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "capture-ingestion-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })));
});

describe("CaptureIngestionService", () => {
  it("creates consent and routes browser, device, and os events through the recorder", async () => {
    const dir = await makeTempDir();
    const consentStore = new FileCaptureConsentStore(path.join(dir, "consent.json"));
    const trajectoryStore = new FileTrajectoryStore(path.join(dir, "trajectories.json"));
    const service = new CaptureIngestionService({ consentStore, trajectoryStore });

    const consent = await service.createConsent({ tier: "operator", purpose: "capture workflow" });
    await expect(service.listActiveConsents()).resolves.toEqual([
      expect.objectContaining({ id: consent.id, tier: "operator" }),
    ]);

    await expect(
      service.recordBrowser({
        sessionId: "sess-capture",
        pageUrl: "https://example.test",
        pageTitle: "Home",
        visibleIndicator: true,
        ts: 1,
        action: { kind: "navigate", target: "https://example.test", ts: 2 },
      }),
    ).resolves.toMatchObject({ recorded: true, trajectory: { sessionId: "sess-capture" } });

    await expect(
      service.recordDevice({
        sessionId: "sess-capture",
        deviceId: "sim-1",
        platform: "ios",
        appId: "mobile-app",
        appName: "Mobile App",
        visibleIndicator: true,
        ts: 3,
        gesture: { kind: "tap", target: "Pay", ts: 4 },
      }),
    ).resolves.toMatchObject({ recorded: true, trajectory: { sessionId: "sess-capture" } });

    await expect(
      service.observeOs({
        sessionId: "sess-capture",
        appId: "terminal",
        visibleIndicator: true,
        ts: 5,
        event: "command-ran",
        commandSummary: "pnpm build",
      }),
    ).resolves.toMatchObject({ recorded: true, trajectory: { sessionId: "sess-capture" } });

    await expect(trajectoryStore.listBySession("sess-capture")).resolves.toHaveLength(3);

    await expect(service.revokeConsent(consent.id)).resolves.toMatchObject({ id: consent.id, revokedAt: expect.any(String) });
    await expect(
      service.observeOs({
        sessionId: "sess-capture",
        appId: "terminal",
        visibleIndicator: true,
        ts: 6,
        event: "command-ran",
        commandSummary: "pnpm test",
      }),
    ).resolves.toEqual({ recorded: false, reason: "missing-consent" });
  });
});
