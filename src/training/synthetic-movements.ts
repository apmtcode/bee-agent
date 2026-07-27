import type { MovementSequence, MovementStep } from "./movement-model.js";

/**
 * Deterministic synthetic movement-stream generator (standing objective 2 —
 * "use synthetic/simulated event streams to validate your code"). Produces
 * families of *related* movement sequences from parameterized task templates so
 * the model's train → replay → generalize pipeline can be validated in the
 * cloud without any real OS input capture.
 *
 * All randomness comes from a seeded LCG, so a given seed always yields the same
 * dataset — no `Math.random`, no wall clock, fully reproducible in CI.
 */

/** Small seeded pseudo-random generator (mulberry32). */
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

/**
 * A task template: an ordered list of step factories. Slot values are drawn
 * from a labelled pool so the held-out split can reuse the *shape* of the task
 * with novel-but-related slot values (true generalization, not memorization).
 */
export type MovementTemplate = {
  name: string;
  build: (draw: (pool: string) => string) => MovementStep[];
};

export type SyntheticPools = Record<string, string[]>;

/** The canonical "operate an app" task family used across tests. */
export const DEFAULT_MOVEMENT_TEMPLATES: MovementTemplate[] = [
  {
    name: "open-search-select",
    build: (draw) => [
      { gesture: "tap", target: draw("appIcon") },
      { gesture: "tap", target: "search-field" },
      { gesture: "type", target: "search-field", valueSummary: draw("query") },
      { gesture: "tap", target: draw("result") },
    ],
  },
  {
    name: "scroll-and-open",
    build: (draw) => [
      { gesture: "tap", target: draw("appIcon") },
      { gesture: "scroll", direction: "down" },
      { gesture: "scroll", direction: "down" },
      { gesture: "tap", target: draw("result") },
    ],
  },
  {
    name: "form-fill-submit",
    build: (draw) => [
      { gesture: "tap", target: draw("field") },
      { gesture: "type", target: draw("field"), valueSummary: draw("query") },
      { gesture: "shortcut", target: "submit" },
    ],
  },
];

export const DEFAULT_TRAIN_POOLS: SyntheticPools = {
  appIcon: ["mail-app", "notes-app", "browser-app"],
  query: ["invoice", "roadmap", "budget"],
  result: ["row-1", "row-2", "row-3"],
  field: ["name-field", "email-field", "amount-field"],
};

/** Related-but-novel slot values — same shapes, values unseen in training. */
export const DEFAULT_HELDOUT_POOLS: SyntheticPools = {
  appIcon: ["calendar-app", "photos-app"],
  query: ["quarterly", "milestone"],
  result: ["row-4", "row-5"],
  field: ["phone-field", "address-field"],
};

export type SyntheticDatasetOptions = {
  seed?: number;
  /** Sequences per template in each split. */
  perTemplate?: number;
  templates?: MovementTemplate[];
  trainPools?: SyntheticPools;
  heldOutPools?: SyntheticPools;
};

export type SyntheticMovementDataset = {
  train: MovementSequence[];
  heldOut: MovementSequence[];
};

function buildSplit(
  label: string,
  templates: MovementTemplate[],
  pools: SyntheticPools,
  perTemplate: number,
  rng: () => number,
): MovementSequence[] {
  const sequences: MovementSequence[] = [];
  for (const template of templates) {
    for (let i = 0; i < perTemplate; i += 1) {
      const draw = (pool: string): string => {
        const values = pools[pool];
        return values && values.length > 0 ? pick(rng, values) : pool;
      };
      sequences.push({
        id: `${label}-${template.name}-${i}`,
        goal: template.name,
        steps: template.build(draw),
      });
    }
  }
  return sequences;
}

/**
 * Generate a train / held-out split. The held-out split draws from
 * {@link DEFAULT_HELDOUT_POOLS} — same task shapes, values never seen in
 * training — so an eval on it measures generalization rather than recall.
 */
export function generateSyntheticMovementDataset(
  options: SyntheticDatasetOptions = {},
): SyntheticMovementDataset {
  const templates = options.templates ?? DEFAULT_MOVEMENT_TEMPLATES;
  const perTemplate = options.perTemplate ?? 8;
  const rng = createSeededRng(options.seed ?? 1);
  return {
    train: buildSplit("train", templates, options.trainPools ?? DEFAULT_TRAIN_POOLS, perTemplate, rng),
    heldOut: buildSplit("held", templates, options.heldOutPools ?? DEFAULT_HELDOUT_POOLS, perTemplate, rng),
  };
}
