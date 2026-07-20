import type { MovementDataset, MovementEvent, MovementSequence } from "./model-backend.js";

/**
 * Synthetic movement-stream generator.
 *
 * bee-agent runs in the cloud with no access to the user's real machine, so the
 * capture → dataset → train → infer loop must be exercised without real OS
 * input. This generator produces deterministic, structured movement sequences
 * from a small "intent grammar": each intent is an observation that reliably
 * triggers a particular action. A seeded PRNG makes every stream reproducible
 * (no Date.now / Math.random), so tests and generalization evals are stable.
 */

export type SyntheticIntent = {
  /** Observation source token that cues this intent (e.g. "browser:deploy-page"). */
  observation: string;
  /** Action tool the intent triggers. */
  tool: string;
  /** Action summary the intent triggers. */
  summary: string;
};

export type SyntheticMovementOptions = {
  seed?: number;
  sequenceCount: number;
  minSteps?: number;
  maxSteps?: number;
  intents?: SyntheticIntent[];
  /** Probability [0,1] of inserting a neutral "noise" observation before a step. */
  noise?: number;
};

export const DEFAULT_SYNTHETIC_INTENTS: SyntheticIntent[] = [
  { observation: "browser:deploy-page", tool: "browser", summary: "click deploy button" },
  { observation: "editor:file-tree", tool: "editor", summary: "open changed file" },
  { observation: "terminal:prompt", tool: "terminal", summary: "run test suite" },
  { observation: "window:notification", tool: "window", summary: "dismiss notification" },
];

/** Deterministic 32-bit LCG (numerical-recipes constants). */
function createRng(seed: number): () => number {
  let state = (seed >>> 0) || 0x9e3779b9;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

function pick<T>(rng: () => number, items: T[]): T {
  const index = Math.min(items.length - 1, Math.floor(rng() * items.length));
  return items[index]!;
}

/**
 * Generate a set of synthetic movement sequences. Each step emits an
 * observation for the chosen intent immediately followed by the intent's
 * action, so the observation → action mapping is learnable. Optional noise
 * observations exercise the policy's back-off/generalization path.
 */
export function generateSyntheticMovementDataset(options: SyntheticMovementOptions): MovementDataset {
  const rng = createRng(options.seed ?? 1);
  const intents = options.intents && options.intents.length > 0 ? options.intents : DEFAULT_SYNTHETIC_INTENTS;
  const minSteps = Math.max(1, options.minSteps ?? 3);
  const maxSteps = Math.max(minSteps, options.maxSteps ?? 6);
  const noise = Math.min(1, Math.max(0, options.noise ?? 0));

  const sequences: MovementSequence[] = [];
  for (let i = 0; i < options.sequenceCount; i += 1) {
    const steps = minSteps + Math.floor(rng() * (maxSteps - minSteps + 1));
    const events: MovementEvent[] = [];
    for (let step = 0; step < steps; step += 1) {
      if (noise > 0 && rng() < noise) {
        events.push({ kind: "observation", source: "window:idle", summary: "ambient activity" });
      }
      const intent = pick(rng, intents);
      events.push({ kind: "observation", source: intent.observation, summary: `observed ${intent.observation}` });
      events.push({ kind: "action", tool: intent.tool, summary: intent.summary });
    }
    sequences.push({ id: `synthetic-${i}`, events });
  }

  return { sequences };
}

/**
 * Build a held-out generalization set: the same intents (so the learned
 * observation → action mapping should transfer) but generated from a different
 * seed, yielding novel orderings the policy never saw during training.
 */
export function generateHeldOutMovements(options: SyntheticMovementOptions): MovementDataset {
  return generateSyntheticMovementDataset({ ...options, seed: (options.seed ?? 1) + 104729 });
}
