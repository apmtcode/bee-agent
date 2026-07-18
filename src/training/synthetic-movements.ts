// Deterministic synthetic movement-stream generator.
//
// The real capture pipeline records movements from a user's machine, which is
// unavailable in the cloud. This module fabricates structured, repeatable
// movement streams from small "programs" (templated flows with slotted targets)
// so the capture → dataset → train → infer → eval loop can be validated in CI.
// It uses a seeded PRNG (no Math.random), so a given seed always yields the same
// dataset — essential for stable tests.

import type { MovementDataset, MovementEvent, MovementSequence } from "./movement-model.js";

/** mulberry32 — a tiny, fast, seedable PRNG. Deterministic per seed. */
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

function pick<T>(rng: () => number, items: readonly T[]): T {
  return items[Math.floor(rng() * items.length) % items.length]!;
}

/** A named flow: an ordered list of movement steps with placeholder targets. */
export type MovementProgram = {
  name: string;
  steps: Omit<MovementEvent, "ts">[];
};

/** Built-in programs covering distinct interaction shapes. */
export const BUILT_IN_MOVEMENT_PROGRAMS: MovementProgram[] = [
  {
    name: "login",
    steps: [
      { kind: "focus", target: "login-window" },
      { kind: "click", target: "username-field" },
      { kind: "type", target: "username-field", value: "user" },
      { kind: "click", target: "password-field" },
      { kind: "type", target: "password-field", value: "secret" },
      { kind: "click", target: "submit-button" },
    ],
  },
  {
    name: "compose",
    steps: [
      { kind: "focus", target: "editor" },
      { kind: "shortcut", target: "new-document" },
      { kind: "type", target: "editor", value: "body" },
      { kind: "shortcut", target: "save" },
    ],
  },
  {
    name: "browse",
    steps: [
      { kind: "focus", target: "browser" },
      { kind: "click", target: "address-bar" },
      { kind: "type", target: "address-bar", value: "url" },
      { kind: "keypress", target: "address-bar", value: "enter" },
      { kind: "scroll", target: "page", direction: "down" },
      { kind: "scroll", target: "page", direction: "down" },
      { kind: "click", target: "result-link" },
    ],
  },
];

export type SyntheticMovementOptions = {
  seed?: number;
  /** Programs to sample from (defaults to the built-ins). */
  programs?: MovementProgram[];
  /** Number of sequences to emit. */
  sequenceCount?: number;
  /** Base timestamp; each event advances by `stepMs`. */
  startTs?: number;
  stepMs?: number;
  /** Restrict generation to a subset of program names (for held-out eval splits). */
  onlyPrograms?: string[];
};

/**
 * Materialise a synthetic dataset by sampling programs and stamping timestamps.
 * The same seed + options always produce an identical dataset.
 */
export function generateSyntheticMovementDataset(options: SyntheticMovementOptions = {}): MovementDataset {
  const rng = createSeededRng(options.seed ?? 1);
  const allPrograms = options.programs ?? BUILT_IN_MOVEMENT_PROGRAMS;
  const programs = options.onlyPrograms
    ? allPrograms.filter((program) => options.onlyPrograms!.includes(program.name))
    : allPrograms;
  if (programs.length === 0) {
    return { version: 1, sequences: [] };
  }
  const sequenceCount = options.sequenceCount ?? 12;
  const stepMs = options.stepMs ?? 100;
  let ts = options.startTs ?? 0;

  const sequences: MovementSequence[] = [];
  for (let i = 0; i < sequenceCount; i += 1) {
    const program = pick(rng, programs);
    const events: MovementEvent[] = program.steps.map((step) => {
      ts += stepMs;
      return { ...step, ts };
    });
    sequences.push({ id: `${program.name}-${i}`, events });
  }
  return { version: 1, sequences };
}
