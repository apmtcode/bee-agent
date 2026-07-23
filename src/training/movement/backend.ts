// Pluggable local-model backend for the movement-learning subsystem (objective
// #2d — "post-train a local model ... and generalize", and the roadmap's
// "pluggable local-model backend interface ... with a deterministic mock
// backend"). The backend contract is intentionally narrow — train a dataset
// into an artifact, then infer a gesture for a (possibly unseen) start/target —
// so a real on-device small model can be dropped in behind the same interface
// while cloud/CI relies on the deterministic mock below.

import type { MovementDataset } from "./dataset.js";
import type { MovementEvent, MovementTrajectory, Point } from "./event-schema.js";

export type MovementModelArtifact = {
  version: 1;
  backend: string;
  exampleCount: number;
  /** Averaged, resampled motion profile the model learned. */
  knots: { t: number; fx: number; fy: number }[];
  meanStepMs: number;
  /** Learned pointer-move count, so replay reproduces the recorded resolution. */
  meanStepCount: number;
  /** Per-label overrides, so a model can specialize gestures it has seen. */
  labels: Record<
    string,
    { knots: { t: number; fx: number; fy: number }[]; meanStepMs: number; meanStepCount: number }
  >;
  metadata?: Record<string, unknown>;
};

export type MovementInferenceRequest = {
  label?: string;
  start: Point;
  target: Point;
  /** Override the learned step count; defaults to the model's knot resolution. */
  steps?: number;
  startTs?: number;
  stepMs?: number;
  button?: "left" | "right" | "middle";
  id?: string;
};

export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset): Promise<MovementModelArtifact>;
  infer(
    artifact: MovementModelArtifact,
    request: MovementInferenceRequest,
  ): Promise<MovementTrajectory>;
}

const DEFAULT_KNOTS = 16;

/**
 * A deterministic, dependency-free backend that "learns" the average normalized
 * motion profile of a dataset. It has no real neural weights but exhibits the
 * behaviour the subsystem needs to validate: it reproduces recorded gestures
 * (replay fidelity) and produces smooth, plausible paths to unseen targets
 * (generalization), because the learned profile is expressed in normalized
 * start→target space and re-scaled at inference time.
 */
export class MockMovementModelBackend implements MovementModelBackend {
  readonly name = "mock-profile-v1";

  constructor(private readonly knotCount: number = DEFAULT_KNOTS) {}

  async train(dataset: MovementDataset): Promise<MovementModelArtifact> {
    const global = averageProfile(
      dataset.examples.map((example) => example.profile),
      this.knotCount,
    );
    const meanStepMs = mean(dataset.examples.map((example) => example.stepMs));

    const labels: MovementModelArtifact["labels"] = {};
    const byLabel = new Map<string, typeof dataset.examples>();
    for (const example of dataset.examples) {
      const bucket = byLabel.get(example.label) ?? [];
      bucket.push(example);
      byLabel.set(example.label, bucket);
    }
    for (const [label, examples] of byLabel) {
      labels[label] = {
        knots: averageProfile(examples.map((example) => example.profile), this.knotCount),
        meanStepMs: mean(examples.map((example) => example.stepMs)),
        meanStepCount: mean(examples.map((example) => example.stepCount)),
      };
    }

    return {
      version: 1,
      backend: this.name,
      exampleCount: dataset.examples.length,
      knots: global,
      meanStepMs,
      meanStepCount: mean(dataset.examples.map((example) => example.stepCount)),
      labels,
    };
  }

  async infer(
    artifact: MovementModelArtifact,
    request: MovementInferenceRequest,
  ): Promise<MovementTrajectory> {
    const specialization =
      request.label && artifact.labels[request.label] ? artifact.labels[request.label]! : undefined;
    const knots = specialization?.knots ?? artifact.knots;
    const learnedStepCount = Math.round(specialization?.meanStepCount ?? artifact.meanStepCount);
    const steps = Math.max(2, request.steps ?? (learnedStepCount > 1 ? learnedStepCount : knots.length));
    const stepMs = request.stepMs ?? Math.max(1, Math.round(specialization?.meanStepMs ?? artifact.meanStepMs) || 16);
    const startTs = request.startTs ?? 0;
    const button = request.button ?? "left";
    const dx = request.target.x - request.start.x;
    const dy = request.target.y - request.start.y;

    const events: MovementEvent[] = [];
    for (let step = 0; step < steps; step += 1) {
      const t = step / (steps - 1);
      const fx = sampleProfile(knots, t, "fx");
      const fy = sampleProfile(knots, t, "fy");
      events.push({
        kind: "pointer-move",
        ts: startTs + step * stepMs,
        x: request.start.x + dx * fx,
        y: request.start.y + dy * fy,
      });
    }
    // Guarantee the gesture lands exactly on the requested target before clicking.
    const last = events[events.length - 1];
    if (last && last.kind === "pointer-move") {
      last.x = request.target.x;
      last.y = request.target.y;
    }
    const clickTs = startTs + steps * stepMs;
    events.push({ kind: "pointer-down", ts: clickTs, x: request.target.x, y: request.target.y, button });
    events.push({ kind: "pointer-up", ts: clickTs + stepMs, x: request.target.x, y: request.target.y, button });

    return {
      id: request.id ?? `inferred-${Math.round(request.target.x)},${Math.round(request.target.y)}`,
      label: request.label ?? "inferred",
      target: { ...request.target },
      events,
    };
  }
}

/** Resample a profile onto `knotCount` evenly spaced knots and average across examples. */
function averageProfile(
  profiles: { t: number; fx: number; fy: number }[][],
  knotCount: number,
): { t: number; fx: number; fy: number }[] {
  const knots: { t: number; fx: number; fy: number }[] = [];
  for (let k = 0; k < knotCount; k += 1) {
    const t = knotCount === 1 ? 1 : k / (knotCount - 1);
    let sumFx = 0;
    let sumFy = 0;
    let count = 0;
    for (const profile of profiles) {
      if (profile.length === 0) {
        continue;
      }
      sumFx += sampleProfile(profile, t, "fx");
      sumFy += sampleProfile(profile, t, "fy");
      count += 1;
    }
    knots.push({
      t,
      fx: count === 0 ? t : sumFx / count,
      fy: count === 0 ? t : sumFy / count,
    });
  }
  return knots;
}

/** Linearly interpolate a profile's `fx`/`fy` at normalized position `t`. */
function sampleProfile(
  profile: { t: number; fx: number; fy: number }[],
  t: number,
  axis: "fx" | "fy",
): number {
  if (profile.length === 0) {
    return t;
  }
  if (t <= profile[0]!.t) {
    return profile[0]![axis];
  }
  const last = profile[profile.length - 1]!;
  if (t >= last.t) {
    return last[axis];
  }
  for (let i = 1; i < profile.length; i += 1) {
    const prev = profile[i - 1]!;
    const next = profile[i]!;
    if (t <= next.t) {
      const span = next.t - prev.t;
      const ratio = span === 0 ? 0 : (t - prev.t) / span;
      return prev[axis] + (next[axis] - prev[axis]) * ratio;
    }
  }
  return last[axis];
}

function mean(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}
