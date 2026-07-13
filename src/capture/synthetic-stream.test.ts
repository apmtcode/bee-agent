import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { FileCaptureConsentStore } from "./consent-store.js";
import { DeviceCaptureAdapter } from "./device-adapter.js";
import { OsCaptureObserver } from "./os-observer.js";
import { ConsentAwareCaptureRecorder } from "./recorder.js";
import { buildReplayManifest } from "./replay.js";
import {
  SYNTHETIC_SCENARIO_LIBRARY,
  createSyntheticRng,
  deriveRelatedScenario,
  generateMovementStream,
  type SyntheticMovementEvent,
} from "./synthetic-stream.js";
import { FileTrajectoryStore } from "./trajectory-store.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "synthetic-stream-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })));
});

describe("createSyntheticRng", () => {
  it("is deterministic and seed-dependent", () => {
    const a = createSyntheticRng(42);
    const b = createSyntheticRng(42);
    const c = createSyntheticRng(7);
    const seqA = [a(), a(), a()];
    const seqB = [b(), b(), b()];
    expect(seqA).toEqual(seqB);
    expect(seqA).not.toEqual([c(), c(), c()]);
    for (const value of seqA) {
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThan(1);
    }
  });
});

describe("generateMovementStream", () => {
  const scenario = SYNTHETIC_SCENARIO_LIBRARY["search-and-open"]!;

  it("produces one event per step with the right channels", () => {
    const stream = generateMovementStream(scenario, { sessionId: "s1", seed: 1 });
    expect(stream).toHaveLength(scenario.steps.length);
    expect(stream.map((event) => event.channel)).toEqual(
      scenario.steps.map((step) => step.channel),
    );
  });

  it("is fully deterministic for a given seed", () => {
    const first = generateMovementStream(scenario, { sessionId: "s1", seed: 99 });
    const second = generateMovementStream(scenario, { sessionId: "s1", seed: 99 });
    expect(second).toEqual(first);
  });

  it("assigns strictly monotonic timestamps within the configured gap bounds", () => {
    const stream = generateMovementStream(scenario, {
      sessionId: "s1",
      seed: 3,
      startTs: 1000,
      minStepGapMs: 10,
      maxStepGapMs: 20,
    });
    const timestamps = stream.map(eventTs);
    expect(timestamps[0]).toBe(1000);
    for (let i = 1; i < timestamps.length; i += 1) {
      const gap = timestamps[i]! - timestamps[i - 1]!;
      expect(gap).toBeGreaterThanOrEqual(10);
      expect(gap).toBeLessThanOrEqual(20);
    }
  });

  it("attaches the scenario outcome only to the final event", () => {
    const stream = generateMovementStream(scenario, { sessionId: "s1", seed: 5 });
    stream.forEach((event, index) => {
      const outcome = event.channel === "os" ? event.input.outcome : event.input.outcome;
      if (index === stream.length - 1) {
        expect(outcome).toEqual(scenario.outcome);
      } else {
        expect(outcome).toBeUndefined();
      }
    });
  });

  it("threads sessionId and capture tier onto every event", () => {
    const stream = generateMovementStream(scenario, {
      sessionId: "sess-x",
      seed: 1,
      captureTier: "app",
    });
    for (const event of stream) {
      expect(event.input.sessionId).toBe("sess-x");
      expect(event.input.captureTier).toBe("app");
    }
  });
});

describe("deriveRelatedScenario", () => {
  const base = SYNTHETIC_SCENARIO_LIBRARY["compose-message"]!;

  it("preserves task structure (step kinds and order)", () => {
    const derived = deriveRelatedScenario(base, 11);
    expect(derived.steps.map((step) => step.channel)).toEqual(
      base.steps.map((step) => step.channel),
    );
    derived.steps.forEach((step, index) => {
      const original = base.steps[index]!;
      if (step.channel === "device" && original.channel === "device") {
        expect(step.gesture).toBe(original.gesture);
      }
      if (step.channel === "os" && original.channel === "os") {
        expect(step.event).toBe(original.event);
      }
    });
  });

  it("perturbs concrete targets so the variant is held-out", () => {
    const derived = deriveRelatedScenario(base, 11);
    const baseTargets = base.steps.flatMap((step) => (step.channel === "device" && step.target ? [step.target] : []));
    const derivedTargets = derived.steps.flatMap((step) => (step.channel === "device" && step.target ? [step.target] : []));
    expect(derivedTargets).not.toEqual(baseTargets);
    expect(derived.name).not.toBe(base.name);
  });

  it("is deterministic per variant seed", () => {
    expect(deriveRelatedScenario(base, 21)).toEqual(deriveRelatedScenario(base, 21));
    expect(deriveRelatedScenario(base, 21)).not.toEqual(deriveRelatedScenario(base, 22));
  });
});

describe("capture pipeline round-trip", () => {
  it("records a synthetic stream into trajectories and rebuilds a replay manifest", async () => {
    const dir = await makeTempDir();
    const consentStore = new FileCaptureConsentStore(path.join(dir, "consent.json"));
    const trajectoryStore = new FileTrajectoryStore(path.join(dir, "trajectories.json"));
    const recorder = new ConsentAwareCaptureRecorder({ consentStore, trajectoryStore });
    const deviceAdapter = new DeviceCaptureAdapter(recorder);
    const osObserver = new OsCaptureObserver(recorder);
    await consentStore.create({ tier: "app", purpose: "synthetic capture" });

    const scenario = SYNTHETIC_SCENARIO_LIBRARY["navigate-settings"]!;
    const sessionId = "sess-roundtrip";
    const stream = generateMovementStream(scenario, {
      sessionId,
      seed: 4,
      captureTier: "app",
    });

    for (const event of stream) {
      const result =
        event.channel === "device"
          ? await deviceAdapter.record(event.input)
          : await osObserver.observe(event.input);
      expect(result.recorded).toBe(true);
    }

    const trajectories = await trajectoryStore.listBySession(sessionId);
    expect(trajectories).toHaveLength(stream.length);

    const manifest = buildReplayManifest({ sessionId, transcript: [], trajectories });
    // Every generated event surfaces as at least one timeline event.
    expect(manifest.eventCount).toBeGreaterThanOrEqual(stream.length);
    // The replay timeline is ordered by timestamp — the recorded movements
    // replay in the same order they were generated.
    const timestamps = manifest.events.map((event) => event.ts);
    const sorted = [...timestamps].sort((a, b) => a - b);
    expect(timestamps).toEqual(sorted);

    const finalOutcome = trajectories.at(-1)?.outcome;
    expect(finalOutcome).toEqual(scenario.outcome);
  });
});

function eventTs(event: SyntheticMovementEvent): number {
  return event.input.ts;
}
