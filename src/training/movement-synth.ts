import type { MovementSequence, MovementToken } from "./movement-model.js";

/**
 * Deterministic synthetic movement-stream generator.
 *
 * bee-agent runs in the cloud with no access to a real machine's input
 * devices, so the movement-learning pipeline is validated against synthetic
 * event streams instead. This generates families of *related* movement
 * sequences from a small set of templates plus a seeded PRNG, so tests can:
 *   - train on one split and generalize to a held-out split of the same family
 *   - assert byte-stable output (seeded → reproducible), never touching a clock
 *     or a real RNG.
 */

export type MovementTemplate = {
  name: string;
  /**
   * Ordered steps. A step is either a fixed token or a choice among variants
   * (the PRNG picks one), letting related sequences share structure while
   * differing in a variable — exactly the "new but related" case.
   */
  steps: Array<MovementToken | { choices: MovementToken[] }>;
};

export type SynthesizeMovementOptions = {
  templates: MovementTemplate[];
  /** Sequences to emit per template. Defaults to 4. */
  perTemplate?: number;
  /** PRNG seed for reproducibility. Defaults to 1. */
  seed?: number;
};

/** Small, dependency-free deterministic PRNG (mulberry32). */
function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function synthesizeMovementSequences(options: SynthesizeMovementOptions): MovementSequence[] {
  const perTemplate = Math.max(1, Math.floor(options.perTemplate ?? 4));
  const random = mulberry32(options.seed ?? 1);
  const sequences: MovementSequence[] = [];

  for (const template of options.templates) {
    for (let i = 0; i < perTemplate; i += 1) {
      const tokens: MovementToken[] = template.steps.map((step) => {
        if (typeof step === "string") {
          return step;
        }
        const index = Math.floor(random() * step.choices.length) % step.choices.length;
        return step.choices[index];
      });
      sequences.push({ id: `${template.name}-${i}`, tokens });
    }
  }

  return sequences;
}

/** A default template family useful for smoke tests and demos. */
export const DEFAULT_MOVEMENT_TEMPLATES: MovementTemplate[] = [
  {
    name: "open-and-deploy",
    steps: [
      "browser#open <n>",
      "browser#click deploy",
      { choices: ["browser#confirm dialog", "browser#confirm modal"] },
      "browser#observe status",
    ],
  },
  {
    name: "edit-and-save",
    steps: [
      "editor#focus file",
      { choices: ["editor#type <n> chars", "editor#type text"] },
      "editor#shortcut save",
      "editor#observe saved",
    ],
  },
];
