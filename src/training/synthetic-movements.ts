import type { MovementDataset, MovementSequence, MovementToken } from "./movement-policy.js";

/**
 * Deterministic synthetic movement-stream generator.
 *
 * The engine runs in Anthropic's cloud with NO access to the user's machine, so
 * real mouse/keyboard/window streams are unavailable. This generator produces
 * reproducible movement sequences from a tiny "movement grammar" — a set of
 * task templates whose steps share local transition structure but differ in
 * ordering and parameters. That gives us:
 *   - repeatable data to validate the capture → dataset → train → replay loop,
 *   - a *train / held-out* split of related-but-novel sequences to measure
 *     generalization (objective #2(d)).
 *
 * No Date/Math.random (forbidden in the sandbox): a seeded LCG drives all
 * choices, so a given seed always yields the same dataset.
 */

/** A named task made of ordered movement steps, each expanded to a token. */
type MovementTemplate = {
  task: string;
  steps: Array<(rng: Lcg) => MovementToken>;
};

export type SyntheticMovementOptions = {
  seed?: number;
  /** Number of sequences to emit. */
  count?: number;
  /** Restrict to these task names (defaults to all built-in tasks). */
  tasks?: string[];
};

/**
 * A small deterministic PRNG (Numerical Recipes LCG). Pure, seedable, and
 * independent of the forbidden global RNG so datasets are fully reproducible.
 */
export class Lcg {
  private state: number;
  constructor(seed: number) {
    this.state = (Math.floor(seed) >>> 0) || 1;
  }
  next(): number {
    this.state = (Math.imul(this.state, 1664525) + 1013904223) >>> 0;
    return this.state / 0x100000000;
  }
  int(maxExclusive: number): number {
    return Math.floor(this.next() * maxExclusive);
  }
  pick<T>(items: readonly T[]): T {
    return items[this.int(items.length)]!;
  }
}

const DIRECTIONS = ["up", "down", "left", "right"] as const;

// Each template is anchored by a task-specific focus token and uses a disjoint
// parameter vocabulary, so cross-task transitions never collide. The structural
// steps (focus → first action, final commit) are deterministic while the
// parameter steps vary — a realistic mix a policy can learn to generalize over.
const TEMPLATES: MovementTemplate[] = [
  {
    task: "open-and-edit",
    steps: [
      () => `os:focus:editor`,
      () => `device:tap:menu`,
      (rng) => `device:type:${rng.pick(["title", "body", "search"])}`,
      (rng) => `device:scroll:${rng.pick(DIRECTIONS)}`,
      () => `device:shortcut:save`,
    ],
  },
  {
    task: "navigate-browser",
    steps: [
      () => `os:focus:browser`,
      (rng) => `device:tap:${rng.pick(["address-bar", "tab", "bookmark"])}`,
      (rng) => `device:type:${rng.pick(["query", "url"])}`,
      () => `device:tap:go`,
      (rng) => `device:scroll:${rng.pick(DIRECTIONS)}`,
    ],
  },
  {
    task: "run-command",
    steps: [
      () => `os:focus:terminal`,
      (rng) => `device:type:${rng.pick(["build", "test", "lint", "status"])}`,
      () => `device:shortcut:enter`,
      (rng) => `device:scroll:${rng.pick(DIRECTIONS)}`,
    ],
  },
];

/**
 * Generate a deterministic movement dataset. Same options ⇒ same dataset.
 */
export function generateSyntheticMovementDataset(options: SyntheticMovementOptions = {}): MovementDataset {
  const rng = new Lcg(options.seed ?? 1);
  const count = Math.max(1, Math.floor(options.count ?? 12));
  const templates = options.tasks
    ? TEMPLATES.filter((template) => options.tasks!.includes(template.task))
    : TEMPLATES;
  const pool = templates.length > 0 ? templates : TEMPLATES;
  const sequences: MovementSequence[] = [];
  for (let i = 0; i < count; i += 1) {
    const template = pool[i % pool.length]!;
    sequences.push({
      id: `${template.task}-${i}`,
      tokens: template.steps.map((step) => step(rng)),
    });
  }
  return { sequences };
}

/**
 * Deterministically split a dataset into train / held-out partitions by index,
 * so the held-out set contains related-but-unseen sequences for a
 * generalization eval.
 */
export function splitMovementDataset(
  dataset: MovementDataset,
  options: { holdOutEvery?: number } = {},
): { train: MovementDataset; heldOut: MovementDataset } {
  const holdOutEvery = Math.max(2, Math.floor(options.holdOutEvery ?? 3));
  const train: MovementSequence[] = [];
  const heldOut: MovementSequence[] = [];
  dataset.sequences.forEach((sequence, index) => {
    if ((index + 1) % holdOutEvery === 0) {
      heldOut.push(sequence);
    } else {
      train.push(sequence);
    }
  });
  return { train: { sequences: train }, heldOut: { sequences: heldOut } };
}
