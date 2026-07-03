/**
 * Synthetic movement-stream generator.
 *
 * bee-agent runs in the cloud with no access to a real machine, so the
 * local-movement subsystem is validated against *simulated* event streams. This
 * module produces deterministic, replayable {@link MovementSequence}s from a
 * small grammar of realistic UI patterns (open → navigate → act), seeded by an
 * integer so tests are reproducible without `Math.random`.
 *
 * The generator can emit a held-out split of *related-but-unseen* sequences
 * (same patterns, fresh target labels) to exercise the model's generalization.
 */

import type { MovementFeature, MovementSequence } from "./movement-model.js";

/** Deterministic 32-bit LCG (Numerical Recipes constants). No global RNG. */
class SeededRng {
  private state: number;

  constructor(seed: number) {
    // Avoid a zero state, which would stick the LCG.
    this.state = (Math.floor(seed) % 2147483647) || 1;
    if (this.state < 0) {
      this.state += 2147483647;
    }
  }

  nextInt(maxExclusive: number): number {
    this.state = (this.state * 1103515245 + 12345) & 0x7fffffff;
    return maxExclusive <= 0 ? 0 : this.state % maxExclusive;
  }

  pick<T>(items: readonly T[]): T {
    return items[this.nextInt(items.length)] as T;
  }
}

type MovementPattern = {
  name: string;
  build: (rng: SeededRng, targets: SyntheticTargets) => MovementFeature[];
};

export type SyntheticTargets = {
  apps: readonly string[];
  buttons: readonly string[];
  fields: readonly string[];
};

const DEFAULT_TARGETS: SyntheticTargets = {
  apps: ["mail", "calendar", "notes", "browser", "chat"],
  buttons: ["send", "save", "compose", "reply", "confirm"],
  fields: ["subject", "body", "search", "title", "note"],
};

/** Related-but-unseen targets for the held-out split (same shapes, new labels). */
const HELD_OUT_TARGETS: SyntheticTargets = {
  apps: ["docs", "sheets", "player", "settings", "gallery"],
  buttons: ["publish", "archive", "share", "pin", "star"],
  fields: ["heading", "caption", "query", "label", "memo"],
};

const PATTERNS: MovementPattern[] = [
  {
    name: "compose-and-send",
    build: (rng, targets) => [
      { tool: "device", gesture: "tap", target: rng.pick(targets.apps) },
      { tool: "device", gesture: "tap", target: rng.pick(targets.buttons) },
      { tool: "device", gesture: "type", target: rng.pick(targets.fields) },
      { tool: "device", gesture: "type", target: rng.pick(targets.fields) },
      { tool: "device", gesture: "tap", target: "send" },
    ],
  },
  {
    name: "scroll-and-select",
    build: (rng, targets) => [
      { tool: "device", gesture: "tap", target: rng.pick(targets.apps) },
      { tool: "device", gesture: "scroll", direction: "down" },
      { tool: "device", gesture: "scroll", direction: "down" },
      { tool: "device", gesture: "tap", target: rng.pick(targets.buttons) },
    ],
  },
  {
    name: "swipe-navigate",
    build: (rng, targets) => [
      { tool: "device", gesture: "tap", target: rng.pick(targets.apps) },
      { tool: "device", gesture: "swipe", direction: "left" },
      { tool: "device", gesture: "swipe", direction: "left" },
      { tool: "device", gesture: "tap", target: rng.pick(targets.buttons) },
      { tool: "device", gesture: "shortcut", target: "save" },
    ],
  },
];

export type GenerateSyntheticMovementsParams = {
  seed: number;
  count: number;
  /** Restrict to a subset of pattern names; defaults to all patterns. */
  patterns?: string[];
  /** Use the held-out target vocabulary (related but unseen labels). */
  heldOut?: boolean;
  targets?: SyntheticTargets;
};

/** Generate `count` synthetic movement sequences deterministically from `seed`. */
export function generateSyntheticMovementSequences(params: GenerateSyntheticMovementsParams): MovementSequence[] {
  const rng = new SeededRng(params.seed);
  const targets = params.targets ?? (params.heldOut ? HELD_OUT_TARGETS : DEFAULT_TARGETS);
  const activePatterns =
    params.patterns && params.patterns.length > 0
      ? PATTERNS.filter((pattern) => params.patterns!.includes(pattern.name))
      : PATTERNS;
  const pool = activePatterns.length > 0 ? activePatterns : PATTERNS;

  const sequences: MovementSequence[] = [];
  for (let i = 0; i < Math.max(0, Math.floor(params.count)); i += 1) {
    const pattern = pool[i % pool.length]!;
    sequences.push({
      id: `${params.heldOut ? "heldout" : "train"}-${pattern.name}-${i}`,
      features: pattern.build(rng, targets),
    });
  }
  return sequences;
}

/**
 * Produce a train / held-out split. The held-out sequences reuse the same
 * movement *patterns* with a disjoint target vocabulary, so evaluating on them
 * measures whether the model generalizes movement shape rather than memorizing
 * exact targets.
 */
export function generateSyntheticMovementSplit(params: {
  seed: number;
  trainCount: number;
  heldOutCount: number;
  patterns?: string[];
}): { train: MovementSequence[]; heldOut: MovementSequence[] } {
  return {
    train: generateSyntheticMovementSequences({
      seed: params.seed,
      count: params.trainCount,
      patterns: params.patterns,
      heldOut: false,
    }),
    heldOut: generateSyntheticMovementSequences({
      seed: params.seed + 1,
      count: params.heldOutCount,
      patterns: params.patterns,
      heldOut: true,
    }),
  };
}

export const SYNTHETIC_MOVEMENT_PATTERN_NAMES = PATTERNS.map((pattern) => pattern.name);
