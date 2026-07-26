import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { FileCaptureConsentStore } from "./consent-store.js";
import { DeviceCaptureAdapter } from "./device-adapter.js";
import { OsCaptureObserver } from "./os-observer.js";
import { ConsentAwareCaptureRecorder } from "./recorder.js";
import { buildReplayManifest } from "./replay.js";
import { FileTrajectoryStore } from "./trajectory-store.js";
import {
  createRng,
  generateSyntheticStream,
  generateSyntheticStreamFamily,
  listSyntheticScenarios,
  streamActionSignature,
} from "./synthetic-stream.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "synthetic-stream-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })));
});

describe("createRng", () => {
  it("is deterministic for a given seed and diverges across seeds", () => {
    const a = createRng(42);
    const b = createRng(42);
    const c = createRng(43);
    const seqA = [a(), a(), a()];
    const seqB = [b(), b(), b()];
    const seqC = [c(), c(), c()];
    expect(seqA).toEqual(seqB);
    expect(seqA).not.toEqual(seqC);
    for (const value of [...seqA, ...seqC]) {
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThan(1);
    }
  });
});

describe("generateSyntheticStream", () => {
  it("produces byte-identical streams for the same (scenario, seed, startTs)", () => {
    const a = generateSyntheticStream({ scenario: "edit-file", seed: 7, startTs: 1000 });
    const b = generateSyntheticStream({ scenario: "edit-file", seed: 7, startTs: 1000 });
    expect(a).toEqual(b);
  });

  it("varies concrete targets across seeds while keeping a monotonic clock", () => {
    const a = generateSyntheticStream({ scenario: "web-search", seed: 1 });
    const b = generateSyntheticStream({ scenario: "web-search", seed: 2 });
    expect(a.events).not.toEqual(b.events);

    for (const stream of [a, b]) {
      const timestamps = stream.events.map((event) => event.input.ts);
      const sorted = [...timestamps].sort((x, y) => x - y);
      expect(timestamps).toEqual(sorted);
      expect(stream.endTs).toBeGreaterThanOrEqual(stream.startTs);
      expect(stream.endTs).toBe(timestamps[timestamps.length - 1]);
    }
  });

  it("emits a seed-invariant action signature per scenario", () => {
    for (const scenario of listSyntheticScenarios()) {
      const signatures = [1, 2, 3, 99].map((seed) =>
        streamActionSignature(generateSyntheticStream({ scenario, seed })),
      );
      for (const signature of signatures) {
        expect(signature).toEqual(signatures[0]);
        expect(signature.length).toBeGreaterThan(0);
      }
    }
  });

  it("attaches a success outcome to the final event of every scenario", () => {
    for (const scenario of listSyntheticScenarios()) {
      const stream = generateSyntheticStream({ scenario, seed: 5 });
      const last = stream.events[stream.events.length - 1]!;
      expect(last.input.outcome?.status).toBe("success");
      // Only the terminal event carries an outcome.
      const withOutcome = stream.events.filter((event) => event.input.outcome);
      expect(withOutcome).toHaveLength(1);
    }
  });

  it("throws on an unknown scenario", () => {
    expect(() => generateSyntheticStream({ scenario: "nope", seed: 1 })).toThrow(/unknown synthetic scenario/);
  });
});

describe("generateSyntheticStreamFamily", () => {
  it("builds a contiguous seed range sharing one signature (train/held-out split)", () => {
    const family = generateSyntheticStreamFamily({ scenario: "run-command", count: 4, baseSeed: 10 });
    expect(family.map((stream) => stream.seed)).toEqual([10, 11, 12, 13]);
    const signatures = family.map(streamActionSignature);
    for (const signature of signatures) {
      expect(signature).toEqual(signatures[0]);
    }
    // Related-but-distinct: the concrete event payloads differ across the family.
    const distinct = new Set(family.map((stream) => JSON.stringify(stream.events)));
    expect(distinct.size).toBeGreaterThan(1);
  });
});

describe("synthetic stream round-trip through the capture pipeline", () => {
  it("records generated events into trajectories and rebuilds a replay manifest", async () => {
    const dir = await makeTempDir();
    const consentStore = new FileCaptureConsentStore(path.join(dir, "consent.json"));
    const trajectoryStore = new FileTrajectoryStore(path.join(dir, "trajectories.json"));
    const recorder = new ConsentAwareCaptureRecorder({ consentStore, trajectoryStore });
    const deviceAdapter = new DeviceCaptureAdapter(recorder);
    const osObserver = new OsCaptureObserver(recorder);

    await consentStore.create({ tier: "app", purpose: "synthetic replay validation" });

    const stream = generateSyntheticStream({ scenario: "edit-file", seed: 3 });

    let recorded = 0;
    for (const event of stream.events) {
      const result =
        event.channel === "device"
          ? await deviceAdapter.record(event.input)
          : await osObserver.observe(event.input);
      expect(result.recorded).toBe(true);
      recorded += 1;
    }
    expect(recorded).toBe(stream.events.length);

    const trajectories = await trajectoryStore.listBySession(stream.sessionId);
    expect(trajectories).toHaveLength(stream.events.length);

    const manifest = buildReplayManifest({
      sessionId: stream.sessionId,
      transcript: [],
      trajectories,
    });

    // Every device gesture becomes a replay action; total events are preserved.
    const deviceEvents = stream.events.filter((event) => event.channel === "device").length;
    const actionEvents = manifest.events.filter((event) => event.kind === "action").length;
    expect(actionEvents).toBe(deviceEvents);
    expect(manifest.eventCount).toBe(manifest.events.length);

    // The replay timeline is globally time-ordered.
    const timeline = manifest.events.map((event) => event.ts);
    expect(timeline).toEqual([...timeline].sort((a, b) => a - b));
  });

  it("respects consent gating: no active grant means nothing is recorded", async () => {
    const dir = await makeTempDir();
    const consentStore = new FileCaptureConsentStore(path.join(dir, "consent.json"));
    const trajectoryStore = new FileTrajectoryStore(path.join(dir, "trajectories.json"));
    const recorder = new ConsentAwareCaptureRecorder({ consentStore, trajectoryStore });
    const osObserver = new OsCaptureObserver(recorder);

    const stream = generateSyntheticStream({ scenario: "run-command", seed: 1 });
    const first = stream.events.find((event) => event.channel === "os")!;
    const result = await osObserver.observe(first.input);
    expect(result).toEqual({ recorded: false, reason: "missing-consent" });
  });
});
