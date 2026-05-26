import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { FileCaptureConsentStore } from "./consent-store.js";
import { ConsentAwareCaptureRecorder } from "./recorder.js";
import { FileTrajectoryStore } from "./trajectory-store.js";
import { BrowserCaptureAdapter } from "./browser-adapter.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "browser-adapter-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })));
});

describe("BrowserCaptureAdapter", () => {
  it("records structured browser observations and actions", async () => {
    const dir = await makeTempDir();
    const consentStore = new FileCaptureConsentStore(path.join(dir, "consent.json"));
    const trajectoryStore = new FileTrajectoryStore(path.join(dir, "trajectories.json"));
    const recorder = new ConsentAwareCaptureRecorder({ consentStore, trajectoryStore });
    const adapter = new BrowserCaptureAdapter(recorder);

    await consentStore.create({ tier: "operator", purpose: "browser capture" });

    const result = await adapter.record({
      sessionId: "sess-browser",
      pageUrl: "https://example.test/deploy",
      pageTitle: "Deploy Dashboard",
      domSummary: "deploy button visible",
      activeElementLabel: "Deploy",
      screenshotRef: "shots/deploy.png",
      visibleIndicator: true,
      ts: 1,
      action: { kind: "click", target: "Deploy", ts: 2 },
      outcome: { status: "success", summary: "deploy queued", reward: 1 },
    });

    expect(result).toMatchObject({ recorded: true, trajectory: { sessionId: "sess-browser" } });
    await expect(trajectoryStore.listBySession("sess-browser")).resolves.toEqual([
      expect.objectContaining({
        observations: [
          expect.objectContaining({
            source: "browser",
            summary: "browser at Deploy Dashboard",
            metadata: expect.objectContaining({
              pageUrl: "https://example.test/deploy",
              activeElementLabel: "Deploy",
            }),
          }),
        ],
        actions: [
          expect.objectContaining({
            tool: "browser",
            summary: "clicked Deploy",
          }),
        ],
      }),
    ]);
  });
});
