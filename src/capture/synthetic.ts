import type { OsEventKind, OsObservationInput } from "./os-observer.js";
import type {
  DeviceCaptureInput,
  DeviceGestureKind,
  DevicePlatform,
} from "./device-adapter.js";
import type { CaptureRecordResult } from "./recorder.js";
import type { CaptureTier, TrajectorySpan } from "./trajectory.js";

/**
 * Synthetic movement event-stream generator.
 *
 * bee-agent runs in the cloud with no access to a real machine, so the
 * local-movement learning subsystem is validated against *simulated* input.
 * This module produces deterministic, replayable streams of OS- and
 * device-level capture events from declarative scenario templates. The output
 * feeds the same recorder/observer path as real capture, so the whole
 * capture → dataset → replay pipeline can be exercised end-to-end without any
 * OS integration.
 *
 * Determinism (seeded PRNG, no wall-clock reads) makes generated streams
 * reproducible across runs — a prerequisite for regression tests and for the
 * generalization eval harness, which compares a replayed/generalized stream's
 * structure against the base trajectory it was derived from.
 */

export type SyntheticActor = "os" | "device";

/** One OS-level step (window/focus/file/command) in a scenario template. */
export type SyntheticOsStep = {
  actor: "os";
  event: OsEventKind;
  appId: string;
  windowTitle?: string;
  filePath?: string;
  commandSummary?: string;
  outcome?: TrajectorySpan["outcome"];
};

/** One device-level step (tap/swipe/scroll/type/shortcut) in a scenario. */
export type SyntheticDeviceStep = {
  actor: "device";
  deviceId: string;
  platform: DevicePlatform;
  appId: string;
  appName: string;
  screenTitle?: string;
  selectionSummary?: string;
  gesture?: {
    kind: DeviceGestureKind;
    target?: string;
    direction?: "up" | "down" | "left" | "right";
    valueSummary?: string;
  };
  outcome?: TrajectorySpan["outcome"];
};

export type SyntheticStep = SyntheticOsStep | SyntheticDeviceStep;

export type SyntheticScenario = {
  name: string;
  description?: string;
  steps: SyntheticStep[];
};

/** A single generated event, tagged with the adapter that should consume it. */
export type SyntheticCaptureEvent =
  | { actor: "os"; input: OsObservationInput }
  | { actor: "device"; input: DeviceCaptureInput };

export type GenerateStreamOptions = {
  sessionId: string;
  /** Seed for the deterministic jitter PRNG. Same seed → identical stream. */
  seed?: number;
  /** Timestamp (ms) the first event is offset from. Default 0. */
  startTs?: number;
  /** Base spacing (ms) between consecutive events. Default 250. */
  stepIntervalMs?: number;
  /** Upper bound (ms, inclusive) of deterministic per-step jitter. Default 0. */
  jitterMs?: number;
  captureTier?: CaptureTier;
  visibleIndicator?: boolean;
};

/**
 * Deterministic PRNG (mulberry32). Returns a function yielding floats in
 * [0, 1). Same seed always produces the same sequence — no global state, no
 * wall-clock, safe for reproducible tests.
 */
export function createSeededRandom(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Expand a scenario template into a concrete, time-ordered event stream.
 * Timestamps are strictly monotonic: each step advances the clock by
 * `stepIntervalMs` plus a deterministic jitter in [0, jitterMs].
 */
export function generateSyntheticStream(
  scenario: SyntheticScenario,
  options: GenerateStreamOptions,
): SyntheticCaptureEvent[] {
  const random = createSeededRandom(options.seed ?? 1);
  const startTs = options.startTs ?? 0;
  const stepInterval = options.stepIntervalMs ?? 250;
  const jitter = options.jitterMs ?? 0;
  const tier = options.captureTier ?? "operator";
  const visibleIndicator = options.visibleIndicator ?? true;

  let ts = startTs;
  const events: SyntheticCaptureEvent[] = [];

  for (const step of scenario.steps) {
    const jitterOffset = jitter > 0 ? Math.floor(random() * (jitter + 1)) : 0;
    ts += stepInterval + jitterOffset;

    if (step.actor === "os") {
      events.push({
        actor: "os",
        input: {
          sessionId: options.sessionId,
          appId: step.appId,
          captureTier: tier,
          visibleIndicator,
          ts,
          event: step.event,
          ...(step.windowTitle ? { windowTitle: step.windowTitle } : {}),
          ...(step.filePath ? { filePath: step.filePath } : {}),
          ...(step.commandSummary ? { commandSummary: step.commandSummary } : {}),
          ...(step.outcome ? { outcome: step.outcome } : {}),
        },
      });
      continue;
    }

    events.push({
      actor: "device",
      input: {
        sessionId: options.sessionId,
        deviceId: step.deviceId,
        platform: step.platform,
        appId: step.appId,
        appName: step.appName,
        captureTier: tier,
        visibleIndicator,
        ts,
        ...(step.screenTitle ? { screenTitle: step.screenTitle } : {}),
        ...(step.selectionSummary ? { selectionSummary: step.selectionSummary } : {}),
        ...(step.gesture
          ? {
              gesture: {
                kind: step.gesture.kind,
                ts,
                ...(step.gesture.target ? { target: step.gesture.target } : {}),
                ...(step.gesture.direction ? { direction: step.gesture.direction } : {}),
                ...(step.gesture.valueSummary ? { valueSummary: step.gesture.valueSummary } : {}),
              },
            }
          : {}),
        ...(step.outcome ? { outcome: step.outcome } : {}),
      },
    });
  }

  return events;
}

/**
 * Structural fingerprint of a stream: the ordered sequence of actor + event
 * (OS) / gesture (device) kinds, ignoring timestamps, target strings, and
 * app identifiers. Two streams that perform the *same shape* of task — even on
 * different apps/files/targets — share a fingerprint. This is the equivalence
 * key the generalization eval harness uses to check that a generalized
 * movement preserves the base trajectory's structure.
 */
export function streamFingerprint(events: SyntheticCaptureEvent[]): string {
  return events
    .map((event) =>
      event.actor === "os"
        ? `os:${event.input.event}`
        : `device:${event.input.gesture?.kind ?? "observe"}`,
    )
    .join(">");
}

/**
 * Produce a "related but new" scenario by substituting tokens across every
 * string-valued field of every step. Used to synthesize generalization targets
 * (same structure, different subjects) from a base scenario — e.g. swap the
 * edited file and the built target while keeping the focus → open → run shape.
 * Substrings are replaced, so callers can template with distinctive tokens.
 */
export function substituteScenarioTokens(
  scenario: SyntheticScenario,
  substitutions: Record<string, string>,
): SyntheticScenario {
  const apply = (value: string | undefined): string | undefined => {
    if (value === undefined) {
      return undefined;
    }
    let next = value;
    for (const [from, to] of Object.entries(substitutions)) {
      if (from.length > 0) {
        next = next.split(from).join(to);
      }
    }
    return next;
  };

  const steps = scenario.steps.map((step): SyntheticStep => {
    if (step.actor === "os") {
      return {
        ...step,
        appId: apply(step.appId) ?? step.appId,
        ...(step.windowTitle !== undefined ? { windowTitle: apply(step.windowTitle) } : {}),
        ...(step.filePath !== undefined ? { filePath: apply(step.filePath) } : {}),
        ...(step.commandSummary !== undefined ? { commandSummary: apply(step.commandSummary) } : {}),
      };
    }
    return {
      ...step,
      appId: apply(step.appId) ?? step.appId,
      appName: apply(step.appName) ?? step.appName,
      ...(step.screenTitle !== undefined ? { screenTitle: apply(step.screenTitle) } : {}),
      ...(step.selectionSummary !== undefined ? { selectionSummary: apply(step.selectionSummary) } : {}),
      ...(step.gesture
        ? {
            gesture: {
              ...step.gesture,
              ...(step.gesture.target !== undefined ? { target: apply(step.gesture.target) } : {}),
              ...(step.gesture.valueSummary !== undefined
                ? { valueSummary: apply(step.gesture.valueSummary) }
                : {}),
            },
          }
        : {}),
    };
  });

  return { ...scenario, steps };
}

/** Structural sink the driver dispatches OS events to (matches OsCaptureObserver). */
export type SyntheticOsSink = {
  observe(input: OsObservationInput): Promise<CaptureRecordResult>;
};

/** Structural sink the driver dispatches device events to (matches DeviceCaptureAdapter). */
export type SyntheticDeviceSink = {
  record(input: DeviceCaptureInput): Promise<CaptureRecordResult>;
};

export type SyntheticStreamSink = {
  os?: SyntheticOsSink;
  device?: SyntheticDeviceSink;
};

/**
 * Feed a generated stream through real capture adapters, in order, returning
 * each adapter's record result. Events whose actor has no configured sink are
 * skipped. This closes the loop: generate → record → (store) → replay/export.
 */
export async function driveSyntheticStream(
  events: SyntheticCaptureEvent[],
  sink: SyntheticStreamSink,
): Promise<CaptureRecordResult[]> {
  const results: CaptureRecordResult[] = [];
  for (const event of events) {
    if (event.actor === "os") {
      if (sink.os) {
        results.push(await sink.os.observe(event.input));
      }
      continue;
    }
    if (sink.device) {
      results.push(await sink.device.record(event.input));
    }
  }
  return results;
}

/**
 * Built-in scenario library covering the two capture surfaces. Templates use
 * distinctive tokens (e.g. `report.ts`, `checkout`) so `substituteScenarioTokens`
 * can derive generalization variants cleanly.
 */
export const BUILTIN_SCENARIOS: Record<string, SyntheticScenario> = {
  "editor-edit-run": {
    name: "editor-edit-run",
    description: "Focus an editor, open a file, run the build, observe success.",
    steps: [
      { actor: "os", event: "focus-changed", appId: "code-editor", windowTitle: "workspace" },
      { actor: "os", event: "file-opened", appId: "code-editor", filePath: "src/report.ts" },
      {
        actor: "os",
        event: "command-ran",
        appId: "terminal",
        commandSummary: "npm run build",
        outcome: { status: "success", summary: "build passed" },
      },
    ],
  },
  "mobile-checkout": {
    name: "mobile-checkout",
    description: "Open a shopping app, search, scroll results, tap checkout.",
    steps: [
      {
        actor: "device",
        deviceId: "phone-1",
        platform: "ios",
        appId: "shop",
        appName: "Shop",
        screenTitle: "home",
        gesture: { kind: "tap", target: "search-field" },
      },
      {
        actor: "device",
        deviceId: "phone-1",
        platform: "ios",
        appId: "shop",
        appName: "Shop",
        screenTitle: "search",
        gesture: { kind: "type", target: "search-field", valueSummary: "running shoes" },
      },
      {
        actor: "device",
        deviceId: "phone-1",
        platform: "ios",
        appId: "shop",
        appName: "Shop",
        screenTitle: "results",
        gesture: { kind: "scroll", direction: "down" },
      },
      {
        actor: "device",
        deviceId: "phone-1",
        platform: "ios",
        appId: "shop",
        appName: "Shop",
        screenTitle: "product",
        gesture: { kind: "tap", target: "checkout" },
        outcome: { status: "success", summary: "reached checkout" },
      },
    ],
  },
};
