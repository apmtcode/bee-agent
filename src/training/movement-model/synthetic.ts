import type { MovementDataset, MovementSequence, MovementStep } from "./types.js";

/**
 * Deterministic synthetic movement-stream generator.
 *
 * We run in the cloud with no access to the user's machine, so the
 * capture -> dataset -> train -> replay loop must be validated against
 * simulated input. This produces movement sequences from a small, fixed
 * "app workflow" grammar with a seeded PRNG, so runs are reproducible without
 * `Math.random`. Splitting one generated corpus into train/holdout sets lets the
 * eval harness measure generalization on new-but-related sequences.
 */

export type SyntheticMovementOptions = {
  seed?: number;
  sequenceCount?: number;
  minSteps?: number;
  maxSteps?: number;
};

type WeightedNext = { key: string; weight: number };

type GrammarState = {
  token: string;
  tool: string;
  gesture?: string;
  direction?: string;
  next: WeightedNext[];
};

/**
 * A tiny grammar with *skewed* transitions: each state has a dominant habitual
 * continuation plus rarer deviations. This mirrors real recorded movement — a
 * user mostly follows the same path — and gives a learnable most-likely-next
 * signal, so argmax inference on held-out sequences should track the dominant
 * path well above chance.
 */
const GRAMMAR: Record<string, GrammarState> = {
  focus: {
    token: "os.focus-changed",
    tool: "os",
    gesture: "focus-changed",
    next: [{ key: "tap", weight: 8 }, { key: "type", weight: 1 }, { key: "scroll", weight: 1 }],
  },
  tap: {
    token: "device.tap",
    tool: "device",
    gesture: "tap",
    next: [{ key: "type", weight: 8 }, { key: "shortcut", weight: 1 }, { key: "scroll", weight: 1 }],
  },
  type: {
    token: "device.type",
    tool: "device",
    gesture: "type",
    next: [{ key: "shortcut", weight: 8 }, { key: "tap", weight: 1 }, { key: "done", weight: 1 }],
  },
  scroll: {
    token: "device.scroll.down",
    tool: "device",
    gesture: "scroll",
    direction: "down",
    next: [{ key: "tap", weight: 8 }, { key: "type", weight: 2 }],
  },
  shortcut: {
    token: "device.shortcut",
    tool: "device",
    gesture: "shortcut",
    next: [{ key: "done", weight: 9 }, { key: "type", weight: 1 }],
  },
  done: { token: "os.command-ran", tool: "os", gesture: "command-ran", next: [] },
};

function pickNext(next: WeightedNext[], roll: number): string {
  const total = next.reduce((sum, entry) => sum + entry.weight, 0);
  let threshold = roll * total;
  for (const entry of next) {
    threshold -= entry.weight;
    if (threshold < 0) {
      return entry.key;
    }
  }
  return next[next.length - 1]!.key;
}

/** Mulberry32: small, deterministic PRNG so datasets are reproducible. */
function makePrng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function generateSyntheticMovementDataset(options: SyntheticMovementOptions = {}): MovementDataset {
  const rng = makePrng(options.seed ?? 1);
  const sequenceCount = Math.max(1, options.sequenceCount ?? 24);
  const minSteps = Math.max(2, options.minSteps ?? 3);
  const maxSteps = Math.max(minSteps, options.maxSteps ?? 8);

  const sequences: MovementSequence[] = [];
  for (let index = 0; index < sequenceCount; index += 1) {
    sequences.push(buildSequence(`synthetic-${index}`, rng, minSteps, maxSteps));
  }
  return { version: 1, sequences };
}

function buildSequence(id: string, rng: () => number, minSteps: number, maxSteps: number): MovementSequence {
  const target = minSteps + Math.floor(rng() * (maxSteps - minSteps + 1));
  const steps: MovementStep[] = [];
  let stateKey = "focus";
  let ts = 1_000;
  while (steps.length < target) {
    const node = GRAMMAR[stateKey]!;
    steps.push({
      token: node.token,
      tool: node.tool,
      ts,
      ...(node.gesture ? { gesture: node.gesture } : {}),
      ...(node.direction ? { direction: node.direction } : {}),
    });
    ts += 250;
    if (node.next.length === 0) {
      break;
    }
    stateKey = pickNext(node.next, rng());
  }
  return { id, steps };
}

/** Split a dataset into disjoint train/holdout partitions (deterministic). */
export function splitMovementDataset(
  dataset: MovementDataset,
  holdoutRatio = 0.25,
): { train: MovementDataset; holdout: MovementDataset } {
  const total = dataset.sequences.length;
  const holdoutSize = Math.max(1, Math.round(total * holdoutRatio));
  const trainSize = Math.max(1, total - holdoutSize);
  return {
    train: { version: 1, sequences: dataset.sequences.slice(0, trainSize) },
    holdout: { version: 1, sequences: dataset.sequences.slice(trainSize) },
  };
}
