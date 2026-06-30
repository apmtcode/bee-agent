/**
 * Synthetic movement-stream generator.
 *
 * The engine runs in the cloud with no real mouse/keyboard, so the capture →
 * dataset → train → replay pipeline is validated against deterministic synthetic
 * demonstrations produced here. Generation is seeded (a tiny LCG) so tests are
 * reproducible without `Math.random`.
 */

import {
  buildMovementDataset,
  buildMovementDemonstration,
  type MovementDataset,
  type MovementDemonstration,
  type MovementEvent,
  type Point,
} from "./events.js";

/** Deterministic linear-congruential generator (Numerical Recipes constants). */
export function createSeededRandom(seed: number): () => number {
  let state = (seed >>> 0) || 1;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

export type DragShape = "line" | "arc";

export type SyntheticDragParams = {
  id: string;
  intent: string;
  start: Point;
  end: Point;
  /** Number of intermediate pointer-move samples (excludes down/up). */
  steps?: number;
  shape?: DragShape;
  /** Lateral bow magnitude (px) at the path midpoint when `shape: "arc"`. */
  arcHeight?: number;
  startTs?: number;
  /** Milliseconds between consecutive samples. */
  stepMs?: number;
};

/**
 * Build a pointer drag: pointer-down at `start`, `steps` interpolated
 * pointer-move samples (optionally bowed into an arc), pointer-up at `end`.
 * The demonstration frame's anchors are exactly `start`/`end`, so a model can
 * recover the path's reference frame for retargeting.
 */
export function generateDragDemonstration(params: SyntheticDragParams): MovementDemonstration {
  const steps = params.steps ?? 8;
  const shape = params.shape ?? "line";
  const arcHeight = params.arcHeight ?? 0;
  const startTs = params.startTs ?? 0;
  const stepMs = params.stepMs ?? 16;

  const dx = params.end.x - params.start.x;
  const dy = params.end.y - params.start.y;
  const length = Math.hypot(dx, dy) || 1;
  // Unit normal to the start→end direction, for the arc bow.
  const nx = -dy / length;
  const ny = dx / length;

  const events: MovementEvent[] = [];
  let ts = startTs;
  events.push({ kind: "pointer-down", ts, x: params.start.x, y: params.start.y, button: "left" });

  for (let i = 1; i <= steps; i += 1) {
    const t = i / (steps + 1);
    let x = params.start.x + dx * t;
    let y = params.start.y + dy * t;
    if (shape === "arc" && arcHeight !== 0) {
      // Parabolic bow peaking at the midpoint (4·t·(1−t) is 0 at ends, 1 at t=0.5).
      const bow = arcHeight * 4 * t * (1 - t);
      x += nx * bow;
      y += ny * bow;
    }
    ts += stepMs;
    events.push({ kind: "pointer-move", ts, x, y });
  }

  ts += stepMs;
  events.push({ kind: "pointer-up", ts, x: params.end.x, y: params.end.y, button: "left" });

  return buildMovementDemonstration({
    id: params.id,
    intent: params.intent,
    events,
    frame: { start: params.start, end: params.end },
    metadata: { synthetic: true, shape, steps },
  });
}

export type SyntheticDatasetParams = {
  intent?: string;
  /** Number of demonstrations to generate. */
  count?: number;
  seed?: number;
  steps?: number;
  shape?: DragShape;
  /** Coordinate bounds within which start/end anchors are sampled. */
  bounds?: { width: number; height: number };
};

/**
 * Generate a labelled dataset of drag demonstrations with randomized (seeded)
 * start/end anchors. Useful as training data and for held-out eval targets.
 */
export function generateSyntheticDataset(params: SyntheticDatasetParams = {}): MovementDataset {
  const intent = params.intent ?? "drag";
  const count = params.count ?? 6;
  const bounds = params.bounds ?? { width: 1280, height: 800 };
  const rng = createSeededRandom(params.seed ?? 42);

  const demonstrations: MovementDemonstration[] = [];
  for (let i = 0; i < count; i += 1) {
    const start = { x: rng() * bounds.width, y: rng() * bounds.height };
    const end = { x: rng() * bounds.width, y: rng() * bounds.height };
    demonstrations.push(
      generateDragDemonstration({
        id: `${intent}-${i}`,
        intent,
        start,
        end,
        steps: params.steps,
        shape: params.shape,
        startTs: i * 10_000,
      }),
    );
  }
  return buildMovementDataset(demonstrations);
}
