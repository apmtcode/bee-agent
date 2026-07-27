// Synthetic movement-stream generator.
//
// bee-agent runs in the cloud with no access to a real machine's input devices,
// so the movement-learning pipeline is validated against deterministic synthetic
// streams instead of live capture. This generator emits grammar-driven movement
// sequences from a small library of workflows (each a weighted state machine).
// Sampling is seeded (mulberry32) so a given seed always yields the same corpus,
// which keeps tests reproducible and lets an eval split held-out sequences that
// are *related but new* — drawn from the same grammar, unseen exact paths.

import { MOVEMENT_END, type MovementSequence, type MovementToken } from "./movement-model.js";

/** A weighted transition from one movement to the possible next movements. */
export type MovementTransition = {
  token: MovementToken;
  /** Next tokens with relative weights; empty means this token can terminate. */
  next: Array<{ token: MovementToken; weight: number }>;
};

/** A named workflow grammar: a start token plus a transition table. */
export type MovementWorkflow = {
  name: string;
  start: MovementToken;
  transitions: MovementTransition[];
};

export type SyntheticStreamOptions = {
  seed?: number;
  /** Number of sequences to generate. Default 24. */
  sequenceCount?: number;
  /** Hard cap on tokens per sequence (excludes the appended end marker). Default 24. */
  maxLength?: number;
  /** Append `MOVEMENT_END` to each sequence. Default true. */
  appendEnd?: boolean;
  /** Restrict generation to these workflow names (default: all). */
  workflows?: string[];
};

/**
 * Deterministic PRNG (mulberry32). Seeded so the whole subsystem stays testable
 * without `Math.random`, which also keeps generated corpora reproducible.
 */
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

/** A small default library covering a few distinct desktop workflows. */
export function defaultMovementWorkflows(): MovementWorkflow[] {
  return [
    {
      name: "browser-search",
      start: "focus/window",
      transitions: [
        { token: "focus/window", next: [{ token: "pointer/click/address-bar", weight: 1 }] },
        { token: "pointer/click/address-bar", next: [{ token: "keyboard/type", weight: 1 }] },
        { token: "keyboard/type", next: [{ token: "keyboard/submit", weight: 1 }] },
        { token: "keyboard/submit", next: [{ token: "pointer/scroll/down", weight: 3 }, { token: "pointer/click/result", weight: 2 }] },
        { token: "pointer/scroll/down", next: [{ token: "pointer/click/result", weight: 3 }, { token: "pointer/scroll/down", weight: 1 }] },
        { token: "pointer/click/result", next: [] },
      ],
    },
    {
      name: "file-edit",
      start: "focus/editor",
      transitions: [
        { token: "focus/editor", next: [{ token: "keyboard/open-file", weight: 1 }] },
        { token: "keyboard/open-file", next: [{ token: "pointer/click/line", weight: 2 }, { token: "keyboard/type", weight: 1 }] },
        { token: "pointer/click/line", next: [{ token: "keyboard/type", weight: 1 }] },
        { token: "keyboard/type", next: [{ token: "keyboard/save", weight: 2 }, { token: "keyboard/type", weight: 1 }] },
        { token: "keyboard/save", next: [] },
      ],
    },
    {
      name: "app-switch",
      start: "keyboard/switch-app",
      transitions: [
        { token: "keyboard/switch-app", next: [{ token: "pointer/click/window", weight: 2 }, { token: "keyboard/switch-app", weight: 1 }] },
        { token: "pointer/click/window", next: [{ token: "pointer/scroll/up", weight: 1 }, { token: "pointer/click/menu", weight: 1 }] },
        { token: "pointer/scroll/up", next: [{ token: "pointer/click/menu", weight: 2 }] },
        { token: "pointer/click/menu", next: [] },
      ],
    },
  ];
}

/**
 * Generate a deterministic corpus of movement sequences from the given (or
 * default) workflow library.
 */
export function generateSyntheticMovementStream(options: SyntheticStreamOptions = {}): MovementSequence[] {
  const random = createSeededRandom(options.seed ?? 1);
  const sequenceCount = Math.max(0, Math.floor(options.sequenceCount ?? 24));
  const maxLength = Math.max(1, Math.floor(options.maxLength ?? 24));
  const appendEnd = options.appendEnd !== false;

  const library = defaultMovementWorkflows().filter(
    (workflow) => !options.workflows || options.workflows.includes(workflow.name),
  );
  if (library.length === 0) {
    return [];
  }

  const sequences: MovementSequence[] = [];
  for (let index = 0; index < sequenceCount; index += 1) {
    const workflow = library[Math.floor(random() * library.length) % library.length]!;
    const tokens = walkWorkflow(workflow, random, maxLength);
    if (appendEnd) {
      tokens.push(MOVEMENT_END);
    }
    sequences.push({ id: `${workflow.name}-${index}`, tokens });
  }
  return sequences;
}

function walkWorkflow(workflow: MovementWorkflow, random: () => number, maxLength: number): MovementToken[] {
  const transitions = new Map(workflow.transitions.map((transition) => [transition.token, transition.next]));
  const tokens: MovementToken[] = [workflow.start];
  let current = workflow.start;
  while (tokens.length < maxLength) {
    const next = transitions.get(current);
    if (!next || next.length === 0) {
      break;
    }
    const chosen = weightedPick(next, random);
    tokens.push(chosen);
    current = chosen;
  }
  return tokens;
}

function weightedPick(options: Array<{ token: MovementToken; weight: number }>, random: () => number): MovementToken {
  const total = options.reduce((sum, option) => sum + Math.max(0, option.weight), 0);
  if (total <= 0) {
    return options[0]!.token;
  }
  let threshold = random() * total;
  for (const option of options) {
    threshold -= Math.max(0, option.weight);
    if (threshold < 0) {
      return option.token;
    }
  }
  return options[options.length - 1]!.token;
}

/**
 * Split a corpus into train/held-out subsets. The held-out sequences come from
 * the same grammar, so eval measures generalization to related-but-unseen runs.
 */
export function splitMovementCorpus(
  sequences: MovementSequence[],
  holdOutFraction = 0.25,
): { train: MovementSequence[]; heldOut: MovementSequence[] } {
  const clamped = Math.min(0.9, Math.max(0, holdOutFraction));
  const holdOutCount = Math.floor(sequences.length * clamped);
  const heldOut = sequences.slice(sequences.length - holdOutCount);
  const train = sequences.slice(0, sequences.length - holdOutCount);
  return { train, heldOut };
}
