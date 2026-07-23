// Generalization eval harness for the movement-learning subsystem (roadmap:
// "measure replay fidelity on held-out but related synthetic trajectories").
// Given a trained model and a set of reference gestures the model did NOT train
// on, it measures how faithfully the model's inferred gesture matches the
// reference — both the final landing accuracy and the along-path deviation.

import type { MovementModelArtifact, MovementModelBackend } from "./backend.js";
import { pointerPath, type MovementTrajectory, type Point } from "./event-schema.js";

export type TrajectoryFidelity = {
  id: string;
  label: string;
  /** Distance between the inferred gesture's landing point and the target. */
  finalError: number;
  /** Mean per-sample distance between inferred and reference pointer paths. */
  meanPathError: number;
  /** Max per-sample distance between inferred and reference pointer paths. */
  maxPathError: number;
  hit: boolean;
};

export type GeneralizationEvalResult = {
  count: number;
  meanFinalError: number;
  meanPathError: number;
  maxPathError: number;
  /** Fraction of gestures whose landing point is within `tolerance` px. */
  hitRate: number;
  perTrajectory: TrajectoryFidelity[];
};

export type EvaluateOptions = {
  tolerance?: number;
};

export function euclidean(a: Point, b: Point): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

/**
 * Resample a pointer path to `count` points by normalized arc-position, so two
 * paths with different sample counts can be compared point-to-point.
 */
export function resamplePath(path: Point[], count: number): Point[] {
  if (path.length === 0) {
    return [];
  }
  if (path.length === 1) {
    return Array.from({ length: count }, () => ({ ...path[0]! }));
  }
  const result: Point[] = [];
  for (let i = 0; i < count; i += 1) {
    const t = count === 1 ? 0 : i / (count - 1);
    const scaled = t * (path.length - 1);
    const lower = Math.floor(scaled);
    const upper = Math.min(path.length - 1, lower + 1);
    const ratio = scaled - lower;
    const a = path[lower]!;
    const b = path[upper]!;
    result.push({ x: a.x + (b.x - a.x) * ratio, y: a.y + (b.y - a.y) * ratio });
  }
  return result;
}

function scoreAgainstReference(
  inferred: MovementTrajectory,
  reference: MovementTrajectory,
  tolerance: number,
): TrajectoryFidelity {
  const target = reference.target ?? pointerPath(reference).at(-1) ?? { x: 0, y: 0 };
  const inferredPath = pointerPath(inferred);
  const landing = inferredPath.at(-1) ?? { x: 0, y: 0 };
  const finalError = euclidean(landing, target);

  const referencePath = pointerPath(reference);
  const sampleCount = Math.max(2, Math.min(inferredPath.length, referencePath.length));
  const a = resamplePath(inferredPath, sampleCount);
  const b = resamplePath(referencePath, sampleCount);
  let sum = 0;
  let max = 0;
  for (let i = 0; i < sampleCount; i += 1) {
    const distance = euclidean(a[i]!, b[i]!);
    sum += distance;
    max = Math.max(max, distance);
  }
  return {
    id: reference.id,
    label: reference.label,
    finalError,
    meanPathError: sampleCount === 0 ? 0 : sum / sampleCount,
    maxPathError: max,
    hit: finalError <= tolerance,
  };
}

/**
 * Train-free evaluation: for each held-out reference gesture, ask the model to
 * infer a gesture toward the same start/target and compare it to the reference.
 */
export async function evaluateGeneralization(
  backend: MovementModelBackend,
  artifact: MovementModelArtifact,
  heldOut: MovementTrajectory[],
  options: EvaluateOptions = {},
): Promise<GeneralizationEvalResult> {
  const tolerance = options.tolerance ?? 2;
  const perTrajectory: TrajectoryFidelity[] = [];
  for (const reference of heldOut) {
    const path = pointerPath(reference);
    if (path.length < 2) {
      continue;
    }
    const start = path[0]!;
    const target = reference.target ?? path.at(-1)!;
    const inferred = await backend.infer(artifact, {
      label: reference.label,
      start,
      target,
      id: `eval-${reference.id}`,
    });
    perTrajectory.push(scoreAgainstReference(inferred, reference, tolerance));
  }

  const count = perTrajectory.length;
  const hits = perTrajectory.filter((entry) => entry.hit).length;
  return {
    count,
    meanFinalError: average(perTrajectory.map((entry) => entry.finalError)),
    meanPathError: average(perTrajectory.map((entry) => entry.meanPathError)),
    maxPathError: perTrajectory.reduce((max, entry) => Math.max(max, entry.maxPathError), 0),
    hitRate: count === 0 ? 0 : hits / count,
    perTrajectory,
  };
}

function average(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}
