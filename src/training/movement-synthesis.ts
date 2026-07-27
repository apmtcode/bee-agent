/**
 * Local-movement learning subsystem — synthetic data + evaluation.
 *
 * bee-agent has no access to the user's real machine in the cloud, so we
 * validate the capture→dataset→train→predict→generalize loop with deterministic
 * synthetic movement streams. This module generates parametric demonstrations
 * and scores a trained {@link MovementModel} on held-out (but related) targets —
 * the "generalization eval harness" from the roadmap.
 */

import type {
  MovementDataset,
  MovementEvent,
  MovementModel,
  MovementSequence,
} from "./movement-model.js";

// ---------------------------------------------------------------------------
// Synthetic gesture generation (deterministic — no Math.random)
// ---------------------------------------------------------------------------

export type SyntheticTarget = {
  x: number;
  y: number;
};

export type SyntheticGestureOptions = {
  label: string;
  /** Fixed pointer start for every demonstration (normalized). */
  start: SyntheticTarget;
  /** Targets to demonstrate reaching — one sequence per target. */
  targets: SyntheticTarget[];
  /** Intermediate mouse-move samples between start and target. */
  steps?: number;
  /** Per-step time delta in ms. */
  stepMs?: number;
  /** Emit a mouse-down/up click at the target. */
  click?: boolean;
};

/**
 * Build point-and-(optionally)-click demonstrations: the pointer glides from a
 * fixed start to each target in a straight line, so the endpoint is a pure
 * linear function of the target params — the relation a good learner recovers.
 */
export function generatePointerGesture(options: SyntheticGestureOptions): MovementSequence[] {
  const steps = Math.max(1, options.steps ?? 4);
  const stepMs = Math.max(1, options.stepMs ?? 16);
  const click = options.click ?? false;

  return options.targets.map((target, targetIndex) => {
    const events: MovementEvent[] = [];
    let t = 0;
    for (let step = 0; step <= steps; step += 1) {
      const fraction = step / steps;
      events.push({
        t,
        type: "mouse-move",
        x: round(options.start.x + (target.x - options.start.x) * fraction),
        y: round(options.start.y + (target.y - options.start.y) * fraction),
      });
      t += stepMs;
    }
    if (click) {
      events.push({ t, type: "mouse-down", x: round(target.x), y: round(target.y), button: "left" });
      t += stepMs;
      events.push({ t, type: "mouse-up", x: round(target.x), y: round(target.y), button: "left" });
    }
    return {
      id: `${options.label}-${targetIndex}`,
      label: options.label,
      events,
      params: { targetX: round(target.x), targetY: round(target.y) },
    };
  });
}

export function buildMovementDataset(sequences: MovementSequence[]): MovementDataset {
  return { version: 1, sequences };
}

// ---------------------------------------------------------------------------
// Generalization evaluation
// ---------------------------------------------------------------------------

export type MovementEvalCase = {
  label: string;
  params: Record<string, number>;
  /** Ground-truth sequence to compare the prediction against. */
  expected: MovementSequence;
};

export type MovementEvalResult = {
  cases: number;
  /** Mean Euclidean pointer error over aligned mouse events (normalized units). */
  meanPointerError: number;
  maxPointerError: number;
  /** Mean absolute error of the final pointer position vs. the requested target. */
  meanEndpointError: number;
  /** Fraction of cases whose endpoint error is within `endpointTolerance`. */
  passRate: number;
};

export type MovementEvalOptions = {
  /** Endpoint error (normalized) considered a pass. Default 0.02 (~2% of screen). */
  endpointTolerance?: number;
};

/**
 * Score a trained model against held-out cases. Endpoint error measures whether
 * the model re-aimed the gesture at the requested target (generalization);
 * pointer error measures trajectory fidelity across the whole movement.
 */
export function evaluateMovementModel(
  model: MovementModel,
  cases: MovementEvalCase[],
  options?: MovementEvalOptions,
): MovementEvalResult {
  const tolerance = options?.endpointTolerance ?? 0.02;
  if (cases.length === 0) {
    return { cases: 0, meanPointerError: 0, maxPointerError: 0, meanEndpointError: 0, passRate: 1 };
  }

  let pointerErrorSum = 0;
  let pointerErrorCount = 0;
  let maxPointerError = 0;
  let endpointErrorSum = 0;
  let passCount = 0;

  for (const testCase of cases) {
    const predicted = model.predict({ label: testCase.label, params: testCase.params });

    const aligned = Math.min(predicted.events.length, testCase.expected.events.length);
    for (let index = 0; index < aligned; index += 1) {
      const a = predicted.events[index]!;
      const b = testCase.expected.events[index]!;
      if (typeof a.x === "number" && typeof a.y === "number" && typeof b.x === "number" && typeof b.y === "number") {
        const error = Math.hypot(a.x - b.x, a.y - b.y);
        pointerErrorSum += error;
        pointerErrorCount += 1;
        maxPointerError = Math.max(maxPointerError, error);
      }
    }

    const endpointError = endpointDistance(predicted, testCase.expected);
    endpointErrorSum += endpointError;
    if (endpointError <= tolerance) {
      passCount += 1;
    }
  }

  return {
    cases: cases.length,
    meanPointerError: pointerErrorCount > 0 ? pointerErrorSum / pointerErrorCount : 0,
    maxPointerError,
    meanEndpointError: endpointErrorSum / cases.length,
    passRate: passCount / cases.length,
  };
}

function endpointDistance(a: MovementSequence, b: MovementSequence): number {
  const pa = lastPointer(a);
  const pb = lastPointer(b);
  if (!pa || !pb) {
    return 0;
  }
  return Math.hypot(pa.x - pb.x, pa.y - pb.y);
}

function lastPointer(sequence: MovementSequence): { x: number; y: number } | undefined {
  for (let index = sequence.events.length - 1; index >= 0; index -= 1) {
    const event = sequence.events[index]!;
    if (typeof event.x === "number" && typeof event.y === "number") {
      return { x: event.x, y: event.y };
    }
  }
  return undefined;
}

function round(value: number): number {
  return Math.round(value * 1e6) / 1e6;
}
