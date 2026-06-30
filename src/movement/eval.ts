/**
 * Generalization eval harness.
 *
 * Measures how faithfully a {@link TrainedMovementModel} reproduces a target
 * movement it was *not* trained on. Each eval case pairs a request context with
 * the ground-truth event stream that performs it; the harness predicts, then
 * compares the predicted pointer path to the ground truth.
 *
 * Fidelity metrics:
 *  - `endpointError`  — distance between the predicted final pointer position
 *                       and the requested `end` anchor (how well it lands).
 *  - `pathRmse`       — root-mean-square distance between resampled predicted
 *                       and expected paths (how well the shape matches).
 */

import {
  euclideanDistance,
  pointerPath,
  type MovementEvent,
  type Point,
} from "./events.js";
import type { MovementContext, TrainedMovementModel } from "./model.js";

export type MovementEvalCase = {
  context: MovementContext;
  expected: MovementEvent[];
};

export type MovementEvalResult = {
  intent: string;
  predictedEvents: number;
  endpointError: number;
  pathRmse: number;
  /** True when the model produced no prediction (e.g. unknown intent). */
  empty: boolean;
};

export type MovementEvalSummary = {
  cases: number;
  evaluated: number;
  meanEndpointError: number;
  meanPathRmse: number;
  maxEndpointError: number;
  maxPathRmse: number;
  results: MovementEvalResult[];
};

/** Resample a polyline to exactly `samples` points by arc-length interpolation. */
export function resamplePath(path: Point[], samples: number): Point[] {
  if (path.length === 0 || samples <= 0) {
    return [];
  }
  if (path.length === 1) {
    return Array.from({ length: samples }, () => ({ ...path[0]! }));
  }

  const cumulative: number[] = [0];
  for (let i = 1; i < path.length; i += 1) {
    cumulative.push(cumulative[i - 1]! + euclideanDistance(path[i - 1]!, path[i]!));
  }
  const total = cumulative[cumulative.length - 1]!;
  if (total === 0) {
    return Array.from({ length: samples }, () => ({ ...path[0]! }));
  }

  const out: Point[] = [];
  for (let s = 0; s < samples; s += 1) {
    const target = (s / (samples - 1)) * total;
    let i = 1;
    while (i < cumulative.length && cumulative[i]! < target) {
      i += 1;
    }
    const segStart = cumulative[i - 1]!;
    const segEnd = cumulative[i]!;
    const segLen = segEnd - segStart || 1;
    const t = (target - segStart) / segLen;
    const a = path[i - 1]!;
    const b = path[i]!;
    out.push({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t });
  }
  return out;
}

export function pathRmse(predicted: Point[], expected: Point[], samples = 32): number {
  const a = resamplePath(predicted, samples);
  const b = resamplePath(expected, samples);
  if (a.length === 0 || b.length === 0) {
    return Number.POSITIVE_INFINITY;
  }
  let sumSq = 0;
  for (let i = 0; i < samples; i += 1) {
    const d = euclideanDistance(a[i]!, b[i]!);
    sumSq += d * d;
  }
  return Math.sqrt(sumSq / samples);
}

export function evaluateCase(
  model: TrainedMovementModel,
  evalCase: MovementEvalCase,
): MovementEvalResult {
  const predictedEvents = model.predict(evalCase.context);
  const predictedPath = pointerPath(predictedEvents);
  const expectedPath = pointerPath(evalCase.expected);
  const empty = predictedPath.length === 0;

  const finalPoint = predictedPath[predictedPath.length - 1];
  const endpointError = finalPoint
    ? euclideanDistance(finalPoint, evalCase.context.end)
    : Number.POSITIVE_INFINITY;

  return {
    intent: evalCase.context.intent,
    predictedEvents: predictedEvents.length,
    endpointError,
    pathRmse: empty ? Number.POSITIVE_INFINITY : pathRmse(predictedPath, expectedPath),
    empty,
  };
}

export function evaluateGeneralization(
  model: TrainedMovementModel,
  cases: MovementEvalCase[],
): MovementEvalSummary {
  const results = cases.map((evalCase) => evaluateCase(model, evalCase));
  const evaluated = results.filter((result) => !result.empty);

  const mean = (values: number[]): number =>
    values.length === 0 ? Number.POSITIVE_INFINITY : values.reduce((sum, v) => sum + v, 0) / values.length;
  const max = (values: number[]): number =>
    values.length === 0 ? Number.POSITIVE_INFINITY : Math.max(...values);

  return {
    cases: cases.length,
    evaluated: evaluated.length,
    meanEndpointError: mean(evaluated.map((r) => r.endpointError)),
    meanPathRmse: mean(evaluated.map((r) => r.pathRmse)),
    maxEndpointError: max(evaluated.map((r) => r.endpointError)),
    maxPathRmse: max(evaluated.map((r) => r.pathRmse)),
    results,
  };
}
