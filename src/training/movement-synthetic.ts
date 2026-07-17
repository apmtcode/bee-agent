import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";

/**
 * Deterministic synthetic movement-stream generator for the local-movement
 * learning subsystem. Because the engine runs in the cloud with no access to a
 * real machine's mouse/keyboard/window events, this fabricates realistic,
 * *related* movement trajectories so the capture→dataset→train→eval pipeline can
 * be validated end-to-end without real OS input.
 *
 * Everything is seeded and pure — no `Math.random`, no `Date` — so a given seed
 * always yields the same trajectories, keeping tests hermetic.
 */

/** One movement step: the structured fields `tokenizeMovement` consumes. */
export type MovementStep = {
  tool: string;
  gesture: string;
  direction?: "up" | "down" | "left" | "right";
  target: string;
};

export type SyntheticMovementOptions = {
  seed: number;
  /** Number of trajectories to generate. */
  count: number;
  /** The canonical "skill" pattern trajectories are variations of. */
  template: MovementStep[];
  /** Per-step probability (0..1) of a perturbation (substitution/insertion). */
  noise?: number;
  /** Alternate targets used when a step is perturbed by substitution. */
  targetPool?: string[];
  /** Wall-clock base timestamp for the first event (ms). Passed in for determinism. */
  startTs?: number;
  /** Milliseconds between successive movements. */
  stepIntervalMs?: number;
  /** Prefix for generated trajectory ids. */
  idPrefix?: string;
  /** Session id assigned to every generated trajectory. */
  sessionId?: string;
};

/** Small, fast, fully-deterministic PRNG (mulberry32) — avoids `Math.random`. */
function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function summarizeStep(step: MovementStep): string {
  const direction = step.direction ? ` ${step.direction}` : "";
  return `${step.gesture}${direction} on ${step.target}`;
}

function stepToAction(step: MovementStep, ts: number): TrajectoryAction {
  return {
    kind: "action",
    tool: step.tool,
    summary: summarizeStep(step),
    ts,
    metadata: {
      gesture: step.gesture,
      ...(step.direction ? { direction: step.direction } : {}),
      target: step.target,
    },
  };
}

/**
 * Generate `count` trajectories that are noisy variations of `template`. Each
 * step may be substituted (different target) or duplicated (extra movement),
 * producing "new but related" sequences to exercise generalization.
 */
export function generateSyntheticMovementTrajectories(
  options: SyntheticMovementOptions,
): TrajectorySpan[] {
  const {
    seed,
    count,
    template,
    noise = 0.15,
    targetPool = ["panel", "menu", "field-a", "field-b", "toolbar"],
    startTs = 0,
    stepIntervalMs = 250,
    idPrefix = "synthetic",
    sessionId = "synthetic-session",
  } = options;

  const random = mulberry32(seed);
  const trajectories: TrajectorySpan[] = [];

  for (let index = 0; index < count; index += 1) {
    const actions: TrajectoryAction[] = [];
    let ts = startTs + index * template.length * stepIntervalMs * 4;
    for (const step of template) {
      const roll = random();
      if (roll < noise / 2 && targetPool.length > 0) {
        // Substitution: same gesture, different target → related movement.
        const alt = targetPool[Math.floor(random() * targetPool.length) % targetPool.length]!;
        actions.push(stepToAction({ ...step, target: alt }, ts));
      } else if (roll < noise) {
        // Insertion: the canonical step plus an extra scroll movement.
        actions.push(stepToAction(step, ts));
        ts += stepIntervalMs;
        actions.push(
          stepToAction({ tool: step.tool, gesture: "scroll", direction: "down", target: step.target }, ts),
        );
      } else {
        actions.push(stepToAction(step, ts));
      }
      ts += stepIntervalMs;
    }
    trajectories.push(
      buildTrajectorySpan({
        id: `${idPrefix}-${index}`,
        sessionId,
        captureTier: "app",
        actions,
        outcome: { status: "success", summary: "synthetic movement trajectory", reward: 1 },
      }),
    );
  }

  return trajectories;
}
