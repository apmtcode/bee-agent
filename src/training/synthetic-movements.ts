// Synthetic movement-stream generator.
//
// bee-agent runs in the cloud with no access to a real machine's input devices,
// so the capture -> dataset -> train -> infer pipeline is validated against
// *simulated* movement streams. This module produces deterministic movement
// sequences from a small task grammar, with seeded parameter jitter, so tests
// can build a training set and a held-out-but-related set to measure
// generalization (see `evaluateNextStepAccuracy`).

import type { MovementSequence, MovementStep } from "./movement-model.js";

/** A named task template that expands into a canonical movement sequence. */
export type SyntheticTask = {
  name: string;
  /** Ordered action verbs the task performs. */
  actions: string[];
};

export type SyntheticMovementOptions = {
  /** Seed for the deterministic PRNG; same seed -> identical dataset. */
  seed: number;
  /** Number of sequences to emit. */
  sequenceCount: number;
  /** Task grammar to sample from; defaults to `DEFAULT_SYNTHETIC_TASKS`. */
  tasks?: SyntheticTask[];
  /** Magnitude of positional jitter applied to movement deltas. Defaults to 4. */
  jitter?: number;
};

/**
 * A default grammar of common desktop interaction tasks. Each expands to a
 * canonical, repeatable movement sequence — the kind of structure a movement
 * model should learn to reproduce and generalize.
 */
export const DEFAULT_SYNTHETIC_TASKS: SyntheticTask[] = [
  { name: "open-app-menu", actions: ["mouse.move", "mouse.click", "window.focus", "key.press"] },
  { name: "drag-file", actions: ["mouse.move", "mouse.down", "mouse.move", "mouse.up"] },
  { name: "type-text", actions: ["window.focus", "key.press", "key.press", "key.press"] },
  { name: "switch-window", actions: ["key.down", "key.press", "key.up", "window.focus"] },
];

/** Generate a deterministic list of synthetic movement sequences. */
export function generateSyntheticMovements(options: SyntheticMovementOptions): MovementSequence[] {
  const tasks = options.tasks ?? DEFAULT_SYNTHETIC_TASKS;
  if (tasks.length === 0) {
    return [];
  }
  const jitter = options.jitter ?? 4;
  const random = mulberry32(options.seed >>> 0);
  const sequences: MovementSequence[] = [];

  for (let index = 0; index < options.sequenceCount; index += 1) {
    const task = tasks[Math.floor(random() * tasks.length)] ?? tasks[0];
    const steps: MovementStep[] = task.actions.map((action, stepIndex) =>
      buildStep(action, stepIndex, jitter, random),
    );
    sequences.push({ id: `${task.name}-${options.seed}-${index}`, steps });
  }

  return sequences;
}

function buildStep(action: string, stepIndex: number, jitter: number, random: () => number): MovementStep {
  switch (action) {
    case "mouse.move":
    case "mouse.down":
    case "mouse.up":
      return {
        action,
        params: {
          x: jitterValue(100 + stepIndex * 20, jitter, random),
          y: jitterValue(80 + stepIndex * 15, jitter, random),
        },
      };
    case "mouse.click":
      return { action, params: { button: "left" } };
    case "key.press":
    case "key.down":
    case "key.up":
      return { action, params: { key: "meta" } };
    case "window.focus":
      return { action, params: { target: "active" } };
    default:
      return { action };
  }
}

function jitterValue(base: number, jitter: number, random: () => number): number {
  const offset = Math.round((random() * 2 - 1) * jitter);
  return base + offset;
}

/** Deterministic 32-bit PRNG (mulberry32); avoids Math.random for reproducibility. */
function mulberry32(seed: number): () => number {
  let state = seed || 1;
  return () => {
    state |= 0;
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
