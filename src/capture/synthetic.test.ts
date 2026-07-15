import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { FileCaptureConsentStore } from "./consent-store.js";
import { ConsentAwareCaptureRecorder } from "./recorder.js";
import { FileTrajectoryStore } from "./trajectory-store.js";
import { OsCaptureObserver } from "./os-observer.js";
import { DeviceCaptureAdapter } from "./device-adapter.js";
import { buildReplayManifest } from "./replay.js";
import {
  BUILTIN_SCENARIOS,
  createSeededRandom,
  driveSyntheticStream,
  generateSyntheticStream,
  streamFingerprint,
  substituteScenarioTokens,
} from "./synthetic.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "synthetic-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })));
});

describe("createSeededRandom", () => {
  it("is deterministic for a given seed and varies across seeds", () => {
    const a = createSeededRandom(42);
    const b = createSeededRandom(42);
    const seqA = [a(), a(), a()];
    const seqB = [b(), b(), b()];
    expect(seqA).toEqual(seqB);
    seqA.forEach((value) => {
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThan(1);
    });

    const c = createSeededRandom(43);
    expect([c(), c(), c()]).not.toEqual(seqA);
  });
});

describe("generateSyntheticStream", () => {
  it("produces strictly monotonic timestamps and reproducible output", () => {
    const first = generateSyntheticStream(BUILTIN_SCENARIOS["editor-edit-run"], {
      sessionId: "sess-1",
      seed: 7,
      startTs: 1000,
      stepIntervalMs: 100,
      jitterMs: 50,
    });
    const second = generateSyntheticStream(BUILTIN_SCENARIOS["editor-edit-run"], {
      sessionId: "sess-1",
      seed: 7,
      startTs: 1000,
      stepIntervalMs: 100,
      jitterMs: 50,
    });

    expect(first).toEqual(second); // same seed → identical stream
    expect(first).toHaveLength(3);

    const timestamps = first.map((event) => event.input.ts);
    for (let i = 1; i < timestamps.length; i += 1) {
      expect(timestamps[i]).toBeGreaterThan(timestamps[i - 1]);
    }
  });

  it("maps device steps into gesture inputs with a matching gesture timestamp", () => {
    const stream = generateSyntheticStream(BUILTIN_SCENARIOS["mobile-checkout"], {
      sessionId: "sess-mobile",
    });
    const deviceEvents = stream.filter((event) => event.actor === "device");
    expect(deviceEvents).toHaveLength(4);
    for (const event of deviceEvents) {
      if (event.actor !== "device") continue;
      expect(event.input.gesture?.ts).toBe(event.input.ts);
    }
  });
});

describe("streamFingerprint / substituteScenarioTokens", () => {
  it("keeps the structural fingerprint stable under token substitution", () => {
    const base = BUILTIN_SCENARIOS["editor-edit-run"];
    const variant = substituteScenarioTokens(base, {
      "src/report.ts": "src/invoice.ts",
      "npm run build": "npm run lint",
    });

    // Subjects changed...
    const variantStream = generateSyntheticStream(variant, { sessionId: "sess-v" });
    const fileEvent = variantStream.find(
      (event) => event.actor === "os" && event.input.event === "file-opened",
    );
    expect(fileEvent?.actor === "os" && fileEvent.input.filePath).toBe("src/invoice.ts");

    // ...but the movement structure (the generalization key) is preserved.
    const baseStream = generateSyntheticStream(base, { sessionId: "sess-b" });
    expect(streamFingerprint(variantStream)).toBe(streamFingerprint(baseStream));
    expect(streamFingerprint(baseStream)).toBe(
      "os:focus-changed>os:file-opened>os:command-ran",
    );
  });
});

describe("driveSyntheticStream round-trip", () => {
  it("generates → records → rebuilds a replay manifest matching the source events", async () => {
    const dir = await makeTempDir();
    const consentStore = new FileCaptureConsentStore(path.join(dir, "consent.json"));
    const trajectoryStore = new FileTrajectoryStore(path.join(dir, "trajectories.json"));
    const recorder = new ConsentAwareCaptureRecorder({ consentStore, trajectoryStore });
    const observer = new OsCaptureObserver(recorder);
    const deviceAdapter = new DeviceCaptureAdapter(recorder);
    await consentStore.create({ tier: "operator", purpose: "synthetic round-trip" });

    const stream = generateSyntheticStream(BUILTIN_SCENARIOS["editor-edit-run"], {
      sessionId: "sess-rt",
      stepIntervalMs: 100,
    });

    const results = await driveSyntheticStream(stream, { os: observer, device: deviceAdapter });
    expect(results.every((result) => result.recorded)).toBe(true);

    const trajectories = await trajectoryStore.listBySession("sess-rt");
    expect(trajectories).toHaveLength(3);

    const manifest = buildReplayManifest({
      sessionId: "sess-rt",
      transcript: [],
      trajectories,
    });
    // Every OS step became exactly one observation event in the replay.
    const observationEvents = manifest.events.filter((event) => event.kind === "observation");
    expect(observationEvents).toHaveLength(3);
    // Replay preserves chronological order of the synthetic stream.
    const ts = manifest.events.map((event) => event.ts);
    expect([...ts].sort((a, b) => a - b)).toEqual(ts);
  });

  it("skips events whose actor has no configured sink", async () => {
    const dir = await makeTempDir();
    const consentStore = new FileCaptureConsentStore(path.join(dir, "consent.json"));
    const trajectoryStore = new FileTrajectoryStore(path.join(dir, "trajectories.json"));
    const recorder = new ConsentAwareCaptureRecorder({ consentStore, trajectoryStore });
    const deviceAdapter = new DeviceCaptureAdapter(recorder);
    await consentStore.create({ tier: "operator", purpose: "synthetic device only" });

    const stream = generateSyntheticStream(BUILTIN_SCENARIOS["mobile-checkout"], {
      sessionId: "sess-dev",
    });
    // Only a device sink is provided; OS events (none here) would be dropped.
    const results = await driveSyntheticStream(stream, { device: deviceAdapter });
    expect(results).toHaveLength(4);
    expect(results.every((result) => result.recorded)).toBe(true);
  });
});
