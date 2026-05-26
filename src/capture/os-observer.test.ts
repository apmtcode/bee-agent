import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { FileCaptureConsentStore } from "./consent-store.js";
import { ConsentAwareCaptureRecorder } from "./recorder.js";
import { FileTrajectoryStore } from "./trajectory-store.js";
import { OsCaptureObserver } from "./os-observer.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "os-observer-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })));
});

describe("OsCaptureObserver", () => {
  it("records structured OS-level observations", async () => {
    const dir = await makeTempDir();
    const consentStore = new FileCaptureConsentStore(path.join(dir, "consent.json"));
    const trajectoryStore = new FileTrajectoryStore(path.join(dir, "trajectories.json"));
    const recorder = new ConsentAwareCaptureRecorder({ consentStore, trajectoryStore });
    const observer = new OsCaptureObserver(recorder);

    await consentStore.create({ tier: "operator", purpose: "os observation" });

    const result = await observer.observe({
      sessionId: "sess-os",
      appId: "terminal",
      visibleIndicator: true,
      ts: 20,
      event: "command-ran",
      commandSummary: "pnpm build",
      outcome: { status: "success", summary: "build completed" },
    });

    expect(result).toMatchObject({ recorded: true, trajectory: { sessionId: "sess-os" } });
    await expect(trajectoryStore.listBySession("sess-os")).resolves.toEqual([
      expect.objectContaining({
        observations: [expect.objectContaining({ source: "os", summary: "ran pnpm build" })],
      }),
    ]);
  });

  it("suppresses sensitive OS observations", async () => {
    const dir = await makeTempDir();
    const consentStore = new FileCaptureConsentStore(path.join(dir, "consent.json"));
    const trajectoryStore = new FileTrajectoryStore(path.join(dir, "trajectories.json"));
    const recorder = new ConsentAwareCaptureRecorder({ consentStore, trajectoryStore });
    const observer = new OsCaptureObserver(recorder);

    await consentStore.create({ tier: "operator", purpose: "os observation" });

    await expect(
      observer.observe({
        sessionId: "sess-os-2",
        appId: "system-settings",
        visibleIndicator: true,
        ts: 30,
        event: "window-opened",
        windowTitle: "Passwords",
      }),
    ).resolves.toEqual({ recorded: false, reason: "sensitive-app" });
  });
});
