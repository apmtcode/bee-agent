import {
  buildMovementToken,
  type MovementDataset,
  type MovementEvent,
  type MovementSequence,
} from "./movement-event.js";

/**
 * Deterministic PRNG (mulberry32). We generate synthetic movement streams to
 * validate the capture→dataset→train→replay round-trip without any real OS
 * input, and determinism keeps the tests reproducible across machines.
 */
export function createSeededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state |= 0;
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function pick<T>(random: () => number, items: readonly T[]): T {
  return items[Math.floor(random() * items.length) % items.length];
}

function randomInt(random: () => number, min: number, max: number): number {
  return min + Math.floor(random() * (max - min + 1));
}

/**
 * A task grammar describes a family of related movement sequences. Each task
 * expands into an ordered list of steps; variants (different targets/directions)
 * exercise the model's ability to generalize across near-identical UIs.
 */
export type MovementTaskGrammar = {
  name: string;
  build: (random: () => number) => Array<{ action: string; descriptor?: string }>;
};

const SCROLL_DIRECTIONS = ["down", "up"] as const;
const FORM_FIELDS = ["email", "password", "search-field", "address", "note"] as const;
const SUBMIT_TARGETS = ["submit", "save", "confirm", "send"] as const;
const LIST_ITEMS = ["row-1", "row-2", "row-3", "result", "card"] as const;

/**
 * Built-in task families. Deliberately overlapping in vocabulary so that a model
 * trained on some variants can generalize to held-out ones (objective 2d).
 */
export const BUILTIN_MOVEMENT_TASKS: MovementTaskGrammar[] = [
  {
    name: "form-fill",
    build: (random) => {
      const field = pick(random, FORM_FIELDS);
      const secondField = pick(random, FORM_FIELDS);
      return [
        { action: "tap", descriptor: field },
        { action: "type", descriptor: field },
        { action: "tap", descriptor: secondField },
        { action: "type", descriptor: secondField },
        { action: "tap", descriptor: pick(random, SUBMIT_TARGETS) },
      ];
    },
  },
  {
    name: "browse-and-select",
    build: (random) => {
      const steps: Array<{ action: string; descriptor?: string }> = [];
      const scrolls = randomInt(random, 1, 3);
      for (let i = 0; i < scrolls; i += 1) {
        steps.push({ action: "scroll", descriptor: pick(random, SCROLL_DIRECTIONS) });
      }
      steps.push({ action: "tap", descriptor: pick(random, LIST_ITEMS) });
      return steps;
    },
  },
  {
    name: "shortcut-command",
    build: (random) => [
      { action: "shortcut", descriptor: pick(random, ["cmd-k", "cmd-p", "cmd-f"]) },
      { action: "type", descriptor: "command-palette" },
      { action: "tap", descriptor: pick(random, ["result", "confirm"]) },
    ],
  },
];

export type SyntheticStreamOptions = {
  seed: number;
  /** Number of sequences to emit. */
  count: number;
  /** Restrict generation to these task names (defaults to all built-ins). */
  tasks?: MovementTaskGrammar[];
  /** Session id prefix for generated sequences. */
  sessionPrefix?: string;
  /** Base timestamp (ms); events are spaced deterministically from here. */
  startTs?: number;
};

/**
 * Generate a deterministic dataset of synthetic movement sequences. Same options
 * → byte-identical output, so it doubles as a fixture generator for tests.
 */
export function generateSyntheticMovementDataset(options: SyntheticStreamOptions): MovementDataset {
  const random = createSeededRandom(options.seed);
  const tasks = options.tasks && options.tasks.length > 0 ? options.tasks : BUILTIN_MOVEMENT_TASKS;
  const sessionPrefix = options.sessionPrefix ?? "synthetic";
  const startTs = options.startTs ?? 0;

  const sequences: MovementSequence[] = [];
  for (let index = 0; index < options.count; index += 1) {
    const task = pick(random, tasks);
    const steps = task.build(random);
    let ts = startTs + index * 100_000;
    const events: MovementEvent[] = steps.map((step) => {
      ts += randomInt(random, 200, 1200);
      const token = buildMovementToken(step.action, step.descriptor);
      return {
        ts,
        token,
        action: step.action,
        ...(step.descriptor ? { descriptor: step.descriptor } : {}),
      };
    });
    sequences.push({
      id: `${sessionPrefix}-${task.name}-${index}`,
      sessionId: `${sessionPrefix}-session-${index}`,
      events,
      outcome: "success",
    });
  }

  return { sequences };
}
