import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { FileCaptureConsentStore } from "./consent-store.js";
import { ConsentAwareCaptureRecorder } from "./recorder.js";
import { FileTrajectoryStore } from "./trajectory-store.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "capture-recorder-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })));
});

describe("ConsentAwareCaptureRecorder", () => {
  it("records trajectories only when consent and visible indicator are present", async () => {
    const dir = await makeTempDir();
    const consentStore = new FileCaptureConsentStore(path.join(dir, "consent.json"));
    const trajectoryStore = new FileTrajectoryStore(path.join(dir, "trajectories.json"));
    const recorder = new ConsentAwareCaptureRecorder({ consentStore, trajectoryStore });

    await expect(
      recorder.record({
        sessionId: "sess-1",
        visibleIndicator: true,
        observations: [{ kind: "observation", source: "browser", summary: "opened deploy page", ts: 1 }],
      }),
    ).resolves.toEqual({ recorded: false, reason: "missing-consent" });

    await consentStore.create({ tier: "operator", purpose: "supervised capture" });

    await expect(
      recorder.record({
        sessionId: "sess-1",
        observations: [{ kind: "observation", source: "browser", summary: "opened deploy page", ts: 1 }],
      }),
    ).resolves.toEqual({ recorded: false, reason: "missing-visible-indicator" });

    const recorded = await recorder.record({
      sessionId: "sess-1",
      visibleIndicator: true,
      observations: [{ kind: "observation", source: "browser", summary: "opened deploy page", ts: 1 }],
      actions: [{ kind: "action", tool: "browser", summary: "clicked deploy", ts: 2 }],
      outcome: { status: "success", summary: "deploy queued", reward: 1 },
    });

    expect(recorded).toMatchObject({ recorded: true, trajectory: { sessionId: "sess-1", captureTier: "operator" } });
    await expect(trajectoryStore.listBySession("sess-1")).resolves.toEqual([
      expect.objectContaining({ consentGrantId: expect.any(String), captureTier: "operator" }),
    ]);
  });

  it("suppresses capture for sensitive apps", async () => {
    const dir = await makeTempDir();
    const consentStore = new FileCaptureConsentStore(path.join(dir, "consent.json"));
    const trajectoryStore = new FileTrajectoryStore(path.join(dir, "trajectories.json"));
    const recorder = new ConsentAwareCaptureRecorder({ consentStore, trajectoryStore });

    await consentStore.create({ tier: "operator", purpose: "supervised capture" });

    await expect(
      recorder.record({
        sessionId: "sess-2",
        appId: "1Password",
        visibleIndicator: true,
        observations: [{ kind: "observation", source: "os", summary: "focused password vault", ts: 1 }],
      }),
    ).resolves.toEqual({ recorded: false, reason: "sensitive-app" });
    await expect(trajectoryStore.list()).resolves.toEqual([]);
  });

  it("allows app-tier capture with matching consent", async () => {
    const dir = await makeTempDir();
    const consentStore = new FileCaptureConsentStore(path.join(dir, "consent.json"));
    const trajectoryStore = new FileTrajectoryStore(path.join(dir, "trajectories.json"));
    const recorder = new ConsentAwareCaptureRecorder({ consentStore, trajectoryStore });

    await consentStore.create({ tier: "app", purpose: "mobile capture" });

    const recorded = await recorder.record({
      sessionId: "sess-3",
      appId: "mobile-app",
      captureTier: "app",
      visibleIndicator: true,
      observations: [{ kind: "observation", source: "device", summary: "opened checkout", ts: 5 }],
      actions: [{ kind: "action", tool: "device", summary: "tapped pay", ts: 6 }],
    });

    expect(recorded).toMatchObject({ recorded: true, trajectory: { sessionId: "sess-3", captureTier: "app" } });
  });
});
