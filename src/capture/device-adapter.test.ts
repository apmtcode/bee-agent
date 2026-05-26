import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { FileCaptureConsentStore } from "./consent-store.js";
import { ConsentAwareCaptureRecorder } from "./recorder.js";
import { FileTrajectoryStore } from "./trajectory-store.js";
import { DeviceCaptureAdapter } from "./device-adapter.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "device-adapter-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })));
});

describe("DeviceCaptureAdapter", () => {
  it("records structured device gestures", async () => {
    const dir = await makeTempDir();
    const consentStore = new FileCaptureConsentStore(path.join(dir, "consent.json"));
    const trajectoryStore = new FileTrajectoryStore(path.join(dir, "trajectories.json"));
    const recorder = new ConsentAwareCaptureRecorder({ consentStore, trajectoryStore });
    const adapter = new DeviceCaptureAdapter(recorder);

    await consentStore.create({ tier: "app", purpose: "device capture" });

    const result = await adapter.record({
      sessionId: "sess-device",
      deviceId: "sim-1",
      platform: "ios",
      appId: "mobile-app",
      appName: "Mobile App",
      screenTitle: "Checkout",
      selectionSummary: "Confirm button visible",
      captureTier: "app",
      visibleIndicator: true,
      ts: 10,
      gesture: { kind: "tap", target: "Confirm", ts: 11 },
    });

    expect(result).toMatchObject({ recorded: true, trajectory: { sessionId: "sess-device", captureTier: "app" } });
    await expect(trajectoryStore.listBySession("sess-device")).resolves.toEqual([
      expect.objectContaining({
        observations: [expect.objectContaining({ source: "device", summary: "Mobile App on Checkout" })],
        actions: [expect.objectContaining({ tool: "device", summary: "tapped Confirm" })],
      }),
    ]);
  });
});
