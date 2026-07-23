// Deterministic synthetic movement-stream generator (standing objective #2 —
// "use synthetic/simulated event streams to validate your code"). Because the
// engine runs in the cloud with no real mouse/keyboard, every capture → dataset
// → train → replay round-trip is validated against streams produced here. All
// randomness flows through a seeded PRNG so runs are reproducible and tests are
// stable (no `Math.random`).

import type { MovementEvent, MovementTrajectory, Point } from "./event-schema.js";

/** mulberry32 — a tiny, fast, seedable PRNG. Returns floats in [0, 1). */
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

/** Smooth ease-in-out over a normalized position `t` in [0, 1]. */
export function easeInOut(t: number): number {
  const clamped = Math.min(1, Math.max(0, t));
  return clamped < 0.5
    ? 2 * clamped * clamped
    : 1 - Math.pow(-2 * clamped + 2, 2) / 2;
}

export type ClickGestureParams = {
  id: string;
  label?: string;
  start: Point;
  target: Point;
  steps?: number;
  startTs?: number;
  stepMs?: number;
  /** Peak pixel jitter added to intermediate points (seeded). 0 = a clean path. */
  jitter?: number;
  seed?: number;
  button?: "left" | "right" | "middle";
};

/**
 * A pointer glide from `start` to `target` following an ease-in-out profile,
 * ending in a button down/up (a click). This is the canonical gesture the mock
 * model learns to reproduce and generalize.
 */
export function generateClickGesture(params: ClickGestureParams): MovementTrajectory {
  const steps = Math.max(2, params.steps ?? 12);
  const stepMs = params.stepMs ?? 16;
  const startTs = params.startTs ?? 0;
  const jitter = params.jitter ?? 0;
  const button = params.button ?? "left";
  const rng = createSeededRng(params.seed ?? 1);

  const events: MovementEvent[] = [];
  for (let step = 0; step < steps; step += 1) {
    const t = step / (steps - 1);
    const eased = easeInOut(t);
    // Jitter fades to zero at the endpoints so the gesture still starts and
    // lands exactly (no floating-point residue at the target).
    const atEndpoint = step === 0 || step === steps - 1;
    const jitterScale = atEndpoint ? 0 : jitter * Math.sin(Math.PI * t);
    const jx = jitter > 0 ? (rng() - 0.5) * 2 * jitterScale : 0;
    const jy = jitter > 0 ? (rng() - 0.5) * 2 * jitterScale : 0;
    events.push({
      kind: "pointer-move",
      ts: startTs + step * stepMs,
      x: params.start.x + (params.target.x - params.start.x) * eased + jx,
      y: params.start.y + (params.target.y - params.start.y) * eased + jy,
    });
  }
  const clickTs = startTs + steps * stepMs;
  events.push({ kind: "pointer-down", ts: clickTs, x: params.target.x, y: params.target.y, button });
  events.push({ kind: "pointer-up", ts: clickTs + stepMs, x: params.target.x, y: params.target.y, button });

  return {
    id: params.id,
    label: params.label ?? `click:${Math.round(params.target.x)},${Math.round(params.target.y)}`,
    target: { ...params.target },
    events,
  };
}

export type TypingGestureParams = {
  id: string;
  label?: string;
  text: string;
  startTs?: number;
  /** Mean per-key dwell in ms; seeded variance is added when `jitterMs` > 0. */
  keyMs?: number;
  jitterMs?: number;
  seed?: number;
};

/**
 * A keystroke sequence (key-down/key-up per character) for typing gestures.
 * Complements pointer gestures so the schema and dataset exercise both input
 * modalities.
 */
export function generateTypingGesture(params: TypingGestureParams): MovementTrajectory {
  const keyMs = params.keyMs ?? 90;
  const jitterMs = params.jitterMs ?? 0;
  const rng = createSeededRng(params.seed ?? 1);
  let ts = params.startTs ?? 0;
  const events: MovementEvent[] = [];
  for (const char of params.text) {
    const dwell = keyMs + (jitterMs > 0 ? (rng() - 0.5) * 2 * jitterMs : 0);
    events.push({ kind: "key-down", ts, key: char });
    events.push({ kind: "key-up", ts: ts + Math.max(1, dwell / 2), key: char });
    ts += Math.max(1, dwell);
  }
  return {
    id: params.id,
    label: params.label ?? `type:${params.text.length}`,
    events,
  };
}

export type GestureDatasetParams = {
  /** Targets to generate click gestures toward. */
  targets: Point[];
  start?: Point;
  steps?: number;
  jitter?: number;
  seed?: number;
  idPrefix?: string;
};

/**
 * Generate a batch of click gestures — one per target — sharing a common start
 * point. Seeds are derived deterministically per index so the whole batch is
 * reproducible from a single seed.
 */
export function generateGestureBatch(params: GestureDatasetParams): MovementTrajectory[] {
  const start = params.start ?? { x: 0, y: 0 };
  const baseSeed = params.seed ?? 1;
  const prefix = params.idPrefix ?? "gesture";
  return params.targets.map((target, index) =>
    generateClickGesture({
      id: `${prefix}-${index}`,
      start,
      target,
      steps: params.steps,
      jitter: params.jitter,
      seed: baseSeed + index * 1013,
    }),
  );
}
