import { buildTrajectorySpan, type TrajectoryAction, type TrajectorySpan } from "../capture/trajectory.js";
import type { DeviceGestureKind } from "../capture/device-adapter.js";

/**
 * Synthetic movement-stream generator.
 *
 * The real capture pipeline needs an actual machine (mouse/keyboard/UI events).
 * In the cloud we have none, so this module fabricates realistic, structured
 * gesture trajectories from named "motifs" (repeatable movement patterns) with
 * deterministic, seedable variation. It lets us validate the whole
 * capture → dataset → train → replay → generalize round-trip in CI, and produce
 * held-out-but-related sequences for the generalization eval harness.
 */

/** One step of a movement motif, mirroring `DeviceCaptureAdapter` gesture shape. */
export type MovementStep = {
  gesture: DeviceGestureKind;
  direction?: "up" | "down" | "left" | "right";
  target?: string;
};

/** A named, repeatable movement pattern (e.g. "open, scroll, tap, type"). */
export type MovementMotif = {
  name: string;
  steps: MovementStep[];
};

export type SyntheticMovementOptions = {
  /** Number of trajectories to generate. */
  count: number;
  /** Deterministic seed. Same seed + options => byte-identical output. */
  seed?: number;
  sessionId?: string;
  /** Motifs to sample from. Defaults to {@link DEFAULT_MOVEMENT_MOTIFS}. */
  motifs?: MovementMotif[];
  /**
   * Per-step probability of a benign variation (repeat a step or drop a
   * trailing step) so trajectories are related but not identical. 0 disables.
   */
  variation?: number;
  /** Base timestamp (ms) for the first action. Defaults to a fixed epoch. */
  baseTs?: number;
};

/** A small library of default motifs covering common local-movement patterns. */
export const DEFAULT_MOVEMENT_MOTIFS: MovementMotif[] = [
  {
    name: "browse-and-select",
    steps: [
      { gesture: "scroll", direction: "down", target: "list" },
      { gesture: "scroll", direction: "down", target: "list" },
      { gesture: "tap", target: "row" },
    ],
  },
  {
    name: "search-and-type",
    steps: [
      { gesture: "tap", target: "search-field" },
      { gesture: "type", target: "search-field" },
      { gesture: "shortcut", target: "submit" },
    ],
  },
  {
    name: "swipe-navigate",
    steps: [
      { gesture: "swipe", direction: "left", target: "carousel" },
      { gesture: "swipe", direction: "left", target: "carousel" },
      { gesture: "tap", target: "cta" },
    ],
  },
];

/** Deterministic mulberry32 PRNG so tests need no `Math.random`. */
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

/** Generate deterministic synthetic movement trajectories from motifs. */
export function generateSyntheticMovementTrajectories(options: SyntheticMovementOptions): TrajectorySpan[] {
  const motifs = options.motifs && options.motifs.length > 0 ? options.motifs : DEFAULT_MOVEMENT_MOTIFS;
  const rng = createSeededRng(options.seed ?? 1);
  const variation = Math.min(Math.max(options.variation ?? 0.25, 0), 1);
  const baseTs = options.baseTs ?? 1_700_000_000_000;
  const sessionId = options.sessionId ?? "synthetic-session";

  const trajectories: TrajectorySpan[] = [];
  for (let index = 0; index < options.count; index += 1) {
    const motif = motifs[Math.floor(rng() * motifs.length) % motifs.length] as MovementMotif;
    const steps = expandSteps(motif.steps, variation, rng);
    let ts = baseTs + index * 100_000;
    const actions: TrajectoryAction[] = steps.map((step) => {
      ts += 250;
      return {
        kind: "action",
        tool: "device",
        summary: `${step.gesture}${step.direction ? ` ${step.direction}` : ""}${step.target ? ` ${step.target}` : ""}`.trim(),
        ts,
        metadata: {
          gesture: step.gesture,
          ...(step.direction ? { direction: step.direction } : {}),
          ...(step.target ? { target: step.target } : {}),
        },
      };
    });

    trajectories.push(
      buildTrajectorySpan({
        id: `synthetic-${motif.name}-${index}`,
        sessionId,
        captureTier: "app",
        actions,
        outcome: { status: "success", summary: `completed ${motif.name}` },
      }),
    );
  }
  return trajectories;
}

function expandSteps(steps: MovementStep[], variation: number, rng: () => number): MovementStep[] {
  if (variation <= 0) {
    return [...steps];
  }
  const expanded: MovementStep[] = [];
  for (const step of steps) {
    expanded.push(step);
    // Occasionally repeat a step to create length/rhythm variation.
    if (rng() < variation) {
      expanded.push(step);
    }
  }
  // Occasionally drop the trailing step so sequences differ in their tail.
  if (expanded.length > 1 && rng() < variation / 2) {
    expanded.pop();
  }
  return expanded;
}
