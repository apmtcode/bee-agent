// Deterministic synthetic movement-stream generator.
//
// The real capture pipeline runs on the user's machine; in the cloud we have no
// access to real mouse/keyboard input, so we synthesize realistic movement
// trajectories from parametrized "tasks". Everything is driven by a seeded PRNG
// so datasets are reproducible across runs and CI. This lets us validate the
// capture -> dataset -> train -> replay round-trip and measure generalization on
// held-out but related trajectories without any OS involvement.

import {
  createCoordinateQuantizer,
  type CoordinateQuantizer,
  type MovementDataset,
  type MovementEvent,
  type MovementModifier,
  type MovementTrajectory,
} from "./events.js";

/** Small deterministic PRNG (mulberry32) — avoids Math.random for reproducibility. */
export function createSeededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export type MovementTaskName = "open-and-type" | "menu-select" | "scroll-and-click";

export type SyntheticTarget = {
  /** Screen point the task acts on, in quantizer coordinates. */
  x: number;
  y: number;
  windowId: string;
};

export type SyntheticTaskSpec = {
  task: MovementTaskName;
  target: SyntheticTarget;
  /** For "open-and-type": how many characters to type. */
  typeLength?: number;
  /** For "scroll-and-click": scroll direction. */
  scrollDown?: boolean;
};

export type GenerateDatasetOptions = {
  seed: number;
  specs: SyntheticTaskSpec[];
  quantizer?: CoordinateQuantizer;
  /** Positional jitter (pixels) added to each pointer coordinate. */
  jitter?: number;
  /** Milliseconds between successive events. */
  stepMs?: number;
};

const DEFAULT_QUANTIZER = createCoordinateQuantizer({ width: 1280, height: 800, cols: 16, rows: 10 });

/**
 * Renders one task spec into a concrete movement trajectory. The event *shape*
 * for a given task is fixed (so the model can learn structure) while coordinates
 * and typed content vary per spec (so held-out specs are genuinely novel).
 */
export function generateTaskTrajectory(
  id: string,
  spec: SyntheticTaskSpec,
  rng: () => number,
  options: { quantizer: CoordinateQuantizer; jitter: number; stepMs: number },
): MovementTrajectory {
  const { jitter, stepMs } = options;
  const events: MovementEvent[] = [];
  let ts = 0;
  const nextTs = () => {
    const value = ts;
    ts += stepMs;
    return value;
  };
  const jittered = (base: number) => base + (rng() - 0.5) * 2 * jitter;
  const px = () => jittered(spec.target.x);
  const py = () => jittered(spec.target.y);

  events.push({ kind: "window-focus", ts: nextTs(), windowId: spec.target.windowId });

  switch (spec.task) {
    case "open-and-type": {
      events.push({ kind: "mouse-move", ts: nextTs(), x: px(), y: py() });
      events.push({ kind: "mouse-down", ts: nextTs(), x: px(), y: py(), button: "left" });
      events.push({ kind: "mouse-up", ts: nextTs(), x: px(), y: py(), button: "left" });
      const length = spec.typeLength ?? 3;
      for (let i = 0; i < length; i += 1) {
        const modifiers: MovementModifier[] = rng() < 0.2 ? ["shift"] : [];
        events.push({ kind: "key-down", ts: nextTs(), key: "a", modifiers });
        events.push({ kind: "key-up", ts: nextTs(), key: "a", modifiers });
      }
      break;
    }
    case "menu-select": {
      events.push({ kind: "mouse-move", ts: nextTs(), x: px(), y: py() });
      events.push({ kind: "mouse-down", ts: nextTs(), x: px(), y: py(), button: "left" });
      events.push({ kind: "mouse-up", ts: nextTs(), x: px(), y: py(), button: "left" });
      // Move down to a menu item then click it.
      const itemY = spec.target.y + 40;
      events.push({ kind: "mouse-move", ts: nextTs(), x: jittered(spec.target.x), y: jittered(itemY) });
      events.push({ kind: "mouse-down", ts: nextTs(), x: jittered(spec.target.x), y: jittered(itemY), button: "left" });
      events.push({ kind: "mouse-up", ts: nextTs(), x: jittered(spec.target.x), y: jittered(itemY), button: "left" });
      break;
    }
    case "scroll-and-click": {
      const dy = spec.scrollDown === false ? -3 : 3;
      events.push({ kind: "scroll", ts: nextTs(), x: px(), y: py(), dx: 0, dy });
      events.push({ kind: "scroll", ts: nextTs(), x: px(), y: py(), dx: 0, dy });
      events.push({ kind: "mouse-move", ts: nextTs(), x: px(), y: py() });
      events.push({ kind: "mouse-down", ts: nextTs(), x: px(), y: py(), button: "left" });
      events.push({ kind: "mouse-up", ts: nextTs(), x: px(), y: py(), button: "left" });
      break;
    }
  }

  return { id, label: spec.task, events };
}

export function generateMovementDataset(options: GenerateDatasetOptions): MovementDataset {
  const quantizer = options.quantizer ?? DEFAULT_QUANTIZER;
  const jitter = options.jitter ?? 6;
  const stepMs = options.stepMs ?? 25;
  const rng = createSeededRandom(options.seed);
  const trajectories = options.specs.map((spec, index) =>
    generateTaskTrajectory(`syn-${index}`, spec, rng, { quantizer, jitter, stepMs }),
  );
  return { quantizer, trajectories };
}
