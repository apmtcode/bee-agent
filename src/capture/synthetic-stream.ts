import type { DeviceCaptureInput, DeviceGestureKind, DevicePlatform } from "./device-adapter.js";
import type { OsEventKind, OsObservationInput } from "./os-observer.js";
import type { CaptureTier, TrajectorySpan } from "./trajectory.js";

/**
 * Synthetic movement event-stream generator.
 *
 * bee-agent's local-movement learning subsystem must be validated without
 * access to a real machine's mouse/keyboard/window events. This module produces
 * deterministic, replayable streams of movement events (OS observations +
 * device gestures) from declarative scenario templates, so the whole
 * capture -> dataset -> replay -> train/infer pipeline can be exercised in the
 * cloud/CI with reproducible fixtures.
 *
 * All randomness flows through a seeded PRNG so the same (scenario, seed) always
 * yields the same stream — a hard requirement for regression tests and for the
 * generalization eval harness, which compares held-out variants against a base.
 */

/** One templated step in a movement scenario. Timestamps are assigned by the generator. */
export type MovementStepTemplate =
  | {
      channel: "os";
      event: OsEventKind;
      windowTitle?: string;
      filePath?: string;
      commandSummary?: string;
    }
  | {
      channel: "device";
      gesture: DeviceGestureKind;
      target?: string;
      direction?: "up" | "down" | "left" | "right";
      valueSummary?: string;
      screenTitle?: string;
      selectionSummary?: string;
    };

/** A declarative description of a coherent local-movement task. */
export type SyntheticMovementScenario = {
  name: string;
  platform: DevicePlatform;
  deviceId: string;
  appId: string;
  appName: string;
  screenTitle?: string;
  steps: MovementStepTemplate[];
  outcome?: NonNullable<TrajectorySpan["outcome"]>;
};

/** One generated event, ready to feed the matching capture adapter. */
export type SyntheticMovementEvent =
  | { channel: "os"; input: OsObservationInput }
  | { channel: "device"; input: DeviceCaptureInput };

export type SyntheticStreamOptions = {
  sessionId: string;
  /** Seed for the deterministic PRNG. Same seed + scenario => identical stream. */
  seed?: number;
  /** Timestamp (ms) of the first event. Defaults to 0. */
  startTs?: number;
  /** Minimum gap (ms) between consecutive events. Defaults to 40. */
  minStepGapMs?: number;
  /** Maximum gap (ms) between consecutive events. Defaults to 260. */
  maxStepGapMs?: number;
  captureTier?: CaptureTier;
  /** Whether the visible-indicator flag is set on emitted events. Defaults to true. */
  visibleIndicator?: boolean;
};

/**
 * Deterministic 32-bit PRNG (mulberry32). Fast, seedable, and reproducible —
 * intentionally not `Math.random()`, so streams are stable across runs and
 * across environments that forbid non-deterministic time/randomness.
 */
export function createSyntheticRng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Generate a full movement stream from a scenario. Pure and deterministic. */
export function generateMovementStream(
  scenario: SyntheticMovementScenario,
  options: SyntheticStreamOptions,
): SyntheticMovementEvent[] {
  const rng = createSyntheticRng(options.seed ?? 1);
  const minGap = Math.max(1, options.minStepGapMs ?? 40);
  const maxGap = Math.max(minGap, options.maxStepGapMs ?? 260);
  const visibleIndicator = options.visibleIndicator ?? true;

  let ts = options.startTs ?? 0;
  const events: SyntheticMovementEvent[] = [];

  scenario.steps.forEach((step, index) => {
    if (index > 0) {
      ts += minGap + Math.floor(rng() * (maxGap - minGap + 1));
    }
    const isLast = index === scenario.steps.length - 1;
    const outcome = isLast ? scenario.outcome : undefined;

    if (step.channel === "os") {
      const input: OsObservationInput = {
        sessionId: options.sessionId,
        appId: scenario.appId,
        visibleIndicator,
        ts,
        event: step.event,
        ...(options.captureTier ? { captureTier: options.captureTier } : {}),
        ...(step.windowTitle ? { windowTitle: step.windowTitle } : {}),
        ...(step.filePath ? { filePath: step.filePath } : {}),
        ...(step.commandSummary ? { commandSummary: step.commandSummary } : {}),
        ...(outcome ? { outcome } : {}),
      };
      events.push({ channel: "os", input });
      return;
    }

    const input: DeviceCaptureInput = {
      sessionId: options.sessionId,
      deviceId: scenario.deviceId,
      platform: scenario.platform,
      appId: scenario.appId,
      appName: scenario.appName,
      visibleIndicator,
      ts,
      gesture: {
        kind: step.gesture,
        ts,
        ...(step.target ? { target: step.target } : {}),
        ...(step.direction ? { direction: step.direction } : {}),
        ...(step.valueSummary ? { valueSummary: step.valueSummary } : {}),
      },
      ...(options.captureTier ? { captureTier: options.captureTier } : {}),
      ...(step.screenTitle ?? scenario.screenTitle
        ? { screenTitle: step.screenTitle ?? scenario.screenTitle }
        : {}),
      ...(step.selectionSummary ? { selectionSummary: step.selectionSummary } : {}),
      ...(outcome ? { outcome } : {}),
    };
    events.push({ channel: "device", input });
  });

  return events;
}

/**
 * Derive a new-but-related scenario from a base one for generalization testing.
 *
 * The task structure (step kinds and order) is preserved so a model trained on
 * the base should transfer, while the concrete targets/titles are perturbed so
 * the derived scenario is genuinely held-out rather than an exact copy. Seeded,
 * so a variant index maps to a stable perturbation.
 */
export function deriveRelatedScenario(
  base: SyntheticMovementScenario,
  variantSeed: number,
): SyntheticMovementScenario {
  const rng = createSyntheticRng(variantSeed);
  const suffix = variantLabel(variantSeed);

  const steps = base.steps.map((step) => {
    if (step.channel === "os") {
      return {
        ...step,
        ...(step.windowTitle ? { windowTitle: `${step.windowTitle} ${suffix}` } : {}),
        ...(step.filePath ? { filePath: perturbPath(step.filePath, suffix) } : {}),
        ...(step.commandSummary ? { commandSummary: `${step.commandSummary} ${suffix}` } : {}),
      } satisfies MovementStepTemplate;
    }
    return {
      ...step,
      ...(step.target ? { target: `${step.target} ${suffix}` } : {}),
      ...(step.direction ? { direction: maybeRotateDirection(step.direction, rng) } : {}),
      ...(step.valueSummary ? { valueSummary: `${step.valueSummary} (${suffix})` } : {}),
    } satisfies MovementStepTemplate;
  });

  return {
    ...base,
    name: `${base.name}#${suffix}`,
    ...(base.screenTitle ? { screenTitle: `${base.screenTitle} ${suffix}` } : {}),
    steps,
  };
}

function variantLabel(seed: number): string {
  const alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ";
  const rng = createSyntheticRng(seed);
  const a = alphabet[Math.floor(rng() * alphabet.length)] ?? "A";
  const b = alphabet[Math.floor(rng() * alphabet.length)] ?? "B";
  return `v${a}${b}`;
}

function perturbPath(filePath: string, suffix: string): string {
  const dot = filePath.lastIndexOf(".");
  if (dot <= 0) {
    return `${filePath}-${suffix}`;
  }
  return `${filePath.slice(0, dot)}-${suffix}${filePath.slice(dot)}`;
}

function maybeRotateDirection(
  direction: "up" | "down" | "left" | "right",
  rng: () => number,
): "up" | "down" | "left" | "right" {
  // Keep the axis (a related, not arbitrary, movement) but occasionally flip it.
  if (rng() < 0.5) {
    return direction;
  }
  const opposites = { up: "down", down: "up", left: "right", right: "left" } as const;
  return opposites[direction];
}

/**
 * A small library of built-in movement scenarios covering common local tasks.
 * Useful as ready-made fixtures for pipeline round-trip and generalization tests.
 */
export const SYNTHETIC_SCENARIO_LIBRARY: Record<string, SyntheticMovementScenario> = {
  "search-and-open": {
    name: "search-and-open",
    platform: "macos",
    deviceId: "sim-desktop",
    appId: "finder",
    appName: "Finder",
    screenTitle: "Documents",
    steps: [
      { channel: "os", event: "focus-changed", windowTitle: "Finder" },
      { channel: "device", gesture: "shortcut", target: "Search" },
      { channel: "device", gesture: "type", target: "Search field", valueSummary: "quarterly report" },
      { channel: "device", gesture: "tap", target: "First result" },
      { channel: "os", event: "file-opened", filePath: "/docs/quarterly-report.pdf" },
    ],
    outcome: { status: "success", summary: "opened the searched document", reward: 1 },
  },
  "compose-message": {
    name: "compose-message",
    platform: "ios",
    deviceId: "sim-phone",
    appId: "messenger",
    appName: "Messenger",
    screenTitle: "Chats",
    steps: [
      { channel: "os", event: "window-opened", windowTitle: "Messenger" },
      { channel: "device", gesture: "tap", target: "New message" },
      { channel: "device", gesture: "type", target: "Recipient", valueSummary: "Alex" },
      { channel: "device", gesture: "type", target: "Body", valueSummary: "running five minutes late" },
      { channel: "device", gesture: "tap", target: "Send" },
    ],
    outcome: { status: "success", summary: "sent the composed message", reward: 1 },
  },
  "navigate-settings": {
    name: "navigate-settings",
    platform: "android",
    deviceId: "sim-tablet",
    appId: "settings",
    appName: "Settings",
    screenTitle: "System",
    steps: [
      { channel: "os", event: "focus-changed", windowTitle: "Settings" },
      { channel: "device", gesture: "scroll", direction: "down" },
      { channel: "device", gesture: "tap", target: "Display" },
      { channel: "device", gesture: "swipe", direction: "left", target: "Brightness" },
      { channel: "os", event: "command-ran", commandSummary: "applied display preference" },
    ],
    outcome: { status: "success", summary: "adjusted a display setting", reward: 1 },
  },
};
