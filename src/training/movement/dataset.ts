// Replayable dataset format for the movement-learning subsystem (objective
// #2b/#2c). A dataset turns raw `MovementTrajectory` recordings into
// model-ready training examples: each example captures the gesture's start,
// goal, and a *normalized* motion profile (the fraction of the start→target
// vector covered at each normalized step). Normalization is what makes the
// learned profile transferable to unseen start/target pairs.

import {
  normalizeMovementTrajectory,
  type MovementTrajectory,
  type Point,
} from "./event-schema.js";

export type MotionSample = {
  /** Normalized step index in [0, 1]. */
  t: number;
  /** Fraction of the start→target displacement covered on each axis. */
  fx: number;
  fy: number;
};

export type MovementTrainingExample = {
  id: string;
  label: string;
  start: Point;
  target: Point;
  /** Number of pointer-move samples in the source gesture. */
  stepCount: number;
  /** Median inter-sample interval in ms (0 when fewer than 2 samples). */
  stepMs: number;
  profile: MotionSample[];
};

export type MovementDataset = {
  version: 1;
  examples: MovementTrainingExample[];
};

/**
 * Convert a single trajectory into a training example. Trajectories without a
 * declared `target` fall back to their final pointer position as the goal.
 * Returns `undefined` for gestures with no pointer motion (e.g. pure typing),
 * which the model layer does not learn from.
 */
export function trajectoryToExample(
  trajectory: MovementTrajectory,
): MovementTrainingExample | undefined {
  const normalized = normalizeMovementTrajectory(trajectory);
  // Learn the motion profile from continuous pointer *movement* only. Button
  // press/release events sit at the target and would otherwise flatten the tail
  // of the profile and inflate the step count.
  const pointerMoves = normalized.events.filter((event) => event.kind === "pointer-move");
  if (pointerMoves.length < 2) {
    return undefined;
  }
  const path: Point[] = pointerMoves.map((event) => ({ x: event.x, y: event.y }));
  const start = path[0]!;
  const target = trajectory.target ?? path[path.length - 1]!;
  const dx = target.x - start.x;
  const dy = target.y - start.y;

  const profile: MotionSample[] = path.map((point, index) => ({
    t: index / (path.length - 1),
    fx: dx === 0 ? (index === path.length - 1 ? 1 : 0) : (point.x - start.x) / dx,
    fy: dy === 0 ? (index === path.length - 1 ? 1 : 0) : (point.y - start.y) / dy,
  }));

  const intervals: number[] = [];
  for (let i = 1; i < pointerMoves.length; i += 1) {
    intervals.push(pointerMoves[i]!.ts - pointerMoves[i - 1]!.ts);
  }

  return {
    id: trajectory.id,
    label: trajectory.label,
    start: { ...start },
    target: { ...target },
    stepCount: path.length,
    stepMs: median(intervals),
    profile,
  };
}

export function buildMovementDataset(trajectories: MovementTrajectory[]): MovementDataset {
  const examples: MovementTrainingExample[] = [];
  for (const trajectory of trajectories) {
    const example = trajectoryToExample(trajectory);
    if (example) {
      examples.push(example);
    }
  }
  return { version: 1, examples };
}

/** Serialize as JSONL (one example per line) — the on-disk/replayable format. */
export function serializeDatasetJsonl(dataset: MovementDataset): string {
  return dataset.examples.map((example) => JSON.stringify(example)).join("\n");
}

export function parseDatasetJsonl(text: string): MovementDataset {
  const examples = text
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .map((line) => JSON.parse(line) as MovementTrainingExample);
  return { version: 1, examples };
}

function median(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1]! + sorted[mid]!) / 2 : sorted[mid]!;
}
