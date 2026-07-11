import type { ReplayTimelineEvent } from "../capture/replay.js";
import type { MovementDataset } from "./model-backend.js";

/**
 * Deterministic synthetic movement-stream generator.
 *
 * bee-agent runs in the cloud with no access to the user's real machine, so the
 * capture -> dataset -> replay -> train -> infer pipeline cannot be exercised on
 * genuine OS input here. This generator produces reproducible observation/action
 * streams (a seeded LCG, no wall-clock, no Math.random) so tests can validate
 * every stage of the pipeline offline.
 */

export type SyntheticStep = {
  /** What the operator "sees". */
  observation: string;
  /** The tool used in response. */
  tool: string;
  /** What the action does. */
  action: string;
};

export type SyntheticMovementOptions = {
  /** Ordered routine to repeat (defaults to a small deploy-like workflow). */
  steps?: SyntheticStep[];
  /** How many times to repeat the routine. */
  repeats?: number;
  /** Deterministic seed for optional step shuffling / noise. */
  seed?: number;
  /** Session id stamped onto the dataset. */
  jobId?: string;
  /** Starting timestamp; each event increments by one. */
  startTs?: number;
};

const DEFAULT_STEPS: SyntheticStep[] = [
  { observation: "deploy dashboard is open", tool: "browser", action: "click the deploy button" },
  { observation: "confirmation dialog appeared", tool: "browser", action: "confirm the deployment" },
  { observation: "build log is streaming", tool: "terminal", action: "watch the build log" },
  { observation: "deployment finished successfully", tool: "browser", action: "close the dialog" },
];

/** A tiny seeded linear-congruential generator — deterministic, dependency-free. */
export function createSeededRandom(seed: number): () => number {
  let state = (seed >>> 0) || 1;
  return () => {
    // Numerical Recipes LCG constants.
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

/**
 * Build a deterministic movement dataset by repeating a routine. With `seed`
 * set, the order of routine repetitions is shuffled deterministically so the
 * learned transition model still recovers the per-observation action mapping —
 * useful for asserting order-invariance.
 */
export function generateSyntheticMovementDataset(options: SyntheticMovementOptions = {}): MovementDataset {
  const steps = options.steps ?? DEFAULT_STEPS;
  const repeats = Math.max(1, options.repeats ?? 3);
  const startTs = options.startTs ?? 1;

  const order: number[] = [];
  for (let repeat = 0; repeat < repeats; repeat += 1) {
    order.push(repeat);
  }
  if (options.seed !== undefined) {
    const random = createSeededRandom(options.seed);
    for (let index = order.length - 1; index > 0; index -= 1) {
      const swap = Math.floor(random() * (index + 1));
      const tmp = order[index]!;
      order[index] = order[swap]!;
      order[swap] = tmp;
    }
  }

  const events: ReplayTimelineEvent[] = [];
  let ts = startTs;
  for (const repeat of order) {
    const trajectoryId = `synthetic-traj-${repeat}`;
    for (const step of steps) {
      events.push({
        kind: "observation",
        ts: ts++,
        trajectoryId,
        source: "synthetic",
        summary: step.observation,
      });
      events.push({
        kind: "action",
        ts: ts++,
        trajectoryId,
        tool: step.tool,
        summary: step.action,
      });
    }
  }

  return { jobId: options.jobId ?? "synthetic-job", events };
}
