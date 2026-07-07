import type { DeviceCaptureInput, DeviceGestureKind, DevicePlatform } from "../capture/device-adapter.js";
import { buildTrajectorySpan, type TrajectoryAction, type TrajectorySpan } from "../capture/trajectory.js";

/**
 * Synthetic movement-stream generator (standing objective #2). The engine runs
 * in the cloud with no access to a real machine, so capture -> dataset ->
 * train -> replay -> generalize is validated against *simulated* input.
 *
 * Deterministic: a numeric `seed` fully determines the corpus (via a mulberry32
 * PRNG), so tests and evals are reproducible without `Math.random`.
 */

export type SyntheticFlowStep = {
  kind: DeviceGestureKind;
  direction?: "up" | "down" | "left" | "right";
  /** Candidate targets — one is sampled per instantiation. */
  targets: string[];
};

export type SyntheticFlow = {
  id: string;
  appId: string;
  appName: string;
  platform: DevicePlatform;
  steps: SyntheticFlowStep[];
};

export type SyntheticCorpusOptions = {
  seed: number;
  /** Flows to draw from. Defaults to {@link DEFAULT_SYNTHETIC_FLOWS}. */
  flows?: SyntheticFlow[];
  /** Trajectories in the training split. */
  trainCount: number;
  /** Trajectories in the held-out split. */
  heldOutCount: number;
  /**
   * Alternate target vocabulary used for the held-out split so it exercises
   * *new-but-related* movements (same gesture grammar, unseen targets). Defaults
   * to a deterministic `${target}-alt` mapping.
   */
  heldOutTargets?: (baseTarget: string) => string;
};

export type SyntheticCorpus = {
  train: TrajectorySpan[];
  heldOut: TrajectorySpan[];
};

/** mulberry32 — a tiny, fast, deterministic PRNG seeded by a 32-bit integer. */
export function createSeededRng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function pick<T>(rng: () => number, items: readonly T[]): T {
  return items[Math.floor(rng() * items.length) % items.length];
}

export const DEFAULT_SYNTHETIC_FLOWS: SyntheticFlow[] = [
  {
    id: "search-and-open",
    appId: "notes",
    appName: "Notes",
    platform: "macos",
    steps: [
      { kind: "tap", targets: ["search-field", "search-box"] },
      { kind: "type", targets: ["query-input"] },
      { kind: "tap", targets: ["first-result", "top-result"] },
      { kind: "scroll", direction: "down", targets: ["result-body"] },
    ],
  },
  {
    id: "compose-and-send",
    appId: "chat",
    appName: "Chat",
    platform: "macos",
    steps: [
      { kind: "tap", targets: ["new-message", "compose-button"] },
      { kind: "type", targets: ["message-body"] },
      { kind: "shortcut", targets: ["send"] },
    ],
  },
  {
    id: "browse-and-select",
    appId: "files",
    appName: "Files",
    platform: "macos",
    steps: [
      { kind: "swipe", direction: "up", targets: ["file-list"] },
      { kind: "tap", targets: ["folder", "directory"] },
      { kind: "swipe", direction: "down", targets: ["file-list"] },
      { kind: "tap", targets: ["document", "item"] },
    ],
  },
];

function instantiateFlow(
  flow: SyntheticFlow,
  index: number,
  rng: () => number,
  mapTarget: (target: string) => string,
): TrajectorySpan {
  const baseTs = 1_000_000 + index * 10_000;
  const actions: TrajectoryAction[] = flow.steps.map((step, stepIndex) => {
    const target = mapTarget(pick(rng, step.targets));
    return {
      kind: "action",
      tool: "device",
      summary: `${step.kind} ${target}`,
      ts: baseTs + stepIndex * 100,
      metadata: {
        gesture: step.kind,
        target,
        ...(step.direction ? { direction: step.direction } : {}),
      },
    } satisfies TrajectoryAction;
  });

  return buildTrajectorySpan({
    id: `${flow.id}-${index}`,
    sessionId: `synthetic-${flow.appId}-${index}`,
    captureTier: "app",
    observations: [
      {
        kind: "observation",
        source: "device",
        summary: `${flow.appName} active`,
        ts: baseTs,
        metadata: { appId: flow.appId, platform: flow.platform },
      },
    ],
    actions,
    outcome: { status: "success", summary: `completed ${flow.id}`, reward: 1 },
  });
}

/**
 * Build a reproducible corpus split into a training set and a held-out set. The
 * held-out set reuses the same flow grammars but remaps targets, so a model
 * that has learned the movement *structure* should generalize to it.
 */
export function generateSyntheticCorpus(options: SyntheticCorpusOptions): SyntheticCorpus {
  const flows = options.flows ?? DEFAULT_SYNTHETIC_FLOWS;
  const rng = createSeededRng(options.seed);
  const heldOutTargets = options.heldOutTargets ?? ((target) => `${target}-alt`);

  const train: TrajectorySpan[] = [];
  for (let index = 0; index < options.trainCount; index += 1) {
    train.push(instantiateFlow(pick(rng, flows), index, rng, (target) => target));
  }

  const heldOut: TrajectorySpan[] = [];
  for (let index = 0; index < options.heldOutCount; index += 1) {
    heldOut.push(instantiateFlow(pick(rng, flows), 1000 + index, rng, heldOutTargets));
  }

  return { train, heldOut };
}

/** Convenience: a single device-capture input matching a synthetic action. */
export function syntheticDeviceInput(
  overrides: Partial<DeviceCaptureInput> & Pick<DeviceCaptureInput, "sessionId" | "deviceId" | "appId" | "appName">,
): DeviceCaptureInput {
  return {
    platform: "macos",
    visibleIndicator: true,
    ts: 1_000_000,
    ...overrides,
  } satisfies DeviceCaptureInput;
}
