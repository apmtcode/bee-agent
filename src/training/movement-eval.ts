import type {
  LocalMovementModel,
  LocalMovementModelBackend,
  MovementDataset,
  MovementSequence,
} from "./movement-model.js";

/**
 * Synthetic movement generation + generalization eval (standing objective #2).
 *
 * These utilities let the training pipeline be validated in the cloud/CI with
 * NO real OS input: a seeded, deterministic generator emits movement sequences
 * from a hidden probabilistic grammar, and the eval harness measures how well a
 * trained model predicts held-out (unseen-but-related) sequences drawn from the
 * same grammar. High held-out next-token accuracy is evidence the model
 * generalizes rather than merely memorizing.
 */

/** Deterministic 32-bit PRNG (mulberry32). No `Math.random`, so runs reproduce. */
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

/**
 * A hidden grammar: a set of named "workflows", each a chain of steps where a
 * step may have alternative continuations chosen with weights. Sampling this
 * grammar produces related-but-varied sequences — the substrate for testing
 * generalization.
 */
export type MovementGrammarStep = {
  token: string;
  /** Weighted next-step choices. Empty ends the sequence. */
  next: { step: string; weight: number }[];
};

export type MovementGrammar = {
  start: { step: string; weight: number }[];
  steps: Record<string, MovementGrammarStep>;
};

/**
 * A small default grammar modeling a few overlapping UI "movements" (open app,
 * navigate, edit, save/discard). Branch points create related variants so a
 * back-off model can generalize across them.
 */
export function defaultMovementGrammar(): MovementGrammar {
  return {
    start: [
      { step: "openEditor", weight: 3 },
      { step: "openBrowser", weight: 2 },
    ],
    steps: {
      openEditor: { token: "editor:focus window", next: [{ step: "navigate", weight: 1 }] },
      openBrowser: { token: "browser:focus window", next: [{ step: "navigate", weight: 1 }] },
      navigate: {
        token: "keyboard:navigate to target",
        next: [
          { step: "edit", weight: 3 },
          { step: "select", weight: 1 },
        ],
      },
      select: { token: "mouse:select region", next: [{ step: "edit", weight: 1 }] },
      edit: {
        token: "keyboard:type content",
        next: [
          { step: "save", weight: 3 },
          { step: "discard", weight: 1 },
        ],
      },
      save: { token: "keyboard:save document", next: [] },
      discard: { token: "keyboard:discard changes", next: [] },
    },
  };
}

export type SyntheticMovementOptions = {
  seed: number;
  sequenceCount: number;
  grammar?: MovementGrammar;
  /** Hard cap on steps per sequence to guarantee termination. Defaults to 32. */
  maxSteps?: number;
  /** Prefix for generated sequence ids. Defaults to "synthetic". */
  idPrefix?: string;
};

/** Generate a deterministic synthetic dataset by sampling the grammar. */
export function generateSyntheticMovementDataset(options: SyntheticMovementOptions): MovementDataset {
  const grammar = options.grammar ?? defaultMovementGrammar();
  const maxSteps = Math.max(1, Math.floor(options.maxSteps ?? 32));
  const idPrefix = options.idPrefix ?? "synthetic";
  const rng = createSeededRng(options.seed);
  const sequences: MovementSequence[] = [];

  for (let i = 0; i < options.sequenceCount; i += 1) {
    const tokens: string[] = [];
    let stepName: string | undefined = pickWeighted(grammar.start, rng)?.step;
    for (let step = 0; step < maxSteps && stepName; step += 1) {
      const node = grammar.steps[stepName];
      if (!node) {
        break;
      }
      tokens.push(node.token);
      stepName = pickWeighted(node.next, rng)?.step;
    }
    sequences.push({ id: `${idPrefix}-${i}`, tokens });
  }

  return { version: 1, sequences };
}

export type NextTokenEvalResult = {
  predictions: number;
  correct: number;
  accuracy: number;
  /** Accuracy grouped by the back-off context order that produced each prediction. */
  byContextOrder: Record<number, { predictions: number; correct: number }>;
};

/**
 * Teacher-forced next-token accuracy over held-out sequences: for every
 * position, predict the next token from the true prefix and compare. This is
 * the generalization eval harness — run it on sequences the model was NOT
 * trained on to measure generalization rather than recall.
 */
export function evaluateNextTokenAccuracy(
  backend: LocalMovementModelBackend,
  model: LocalMovementModel,
  sequences: readonly MovementSequence[],
): NextTokenEvalResult {
  let predictions = 0;
  let correct = 0;
  const byContextOrder: Record<number, { predictions: number; correct: number }> = {};

  for (const sequence of sequences) {
    for (let i = 0; i < sequence.tokens.length; i += 1) {
      const context = sequence.tokens.slice(0, i);
      const prediction = backend.predictNext(model, context);
      predictions += 1;
      const bucket = (byContextOrder[prediction.contextOrder] ??= { predictions: 0, correct: 0 });
      bucket.predictions += 1;
      if (prediction.token === sequence.tokens[i]) {
        correct += 1;
        bucket.correct += 1;
      }
    }
  }

  return {
    predictions,
    correct,
    accuracy: predictions > 0 ? correct / predictions : 0,
    byContextOrder,
  };
}

function pickWeighted<T extends { weight: number }>(choices: readonly T[], rng: () => number): T | undefined {
  if (choices.length === 0) {
    return undefined;
  }
  const total = choices.reduce((sum, choice) => sum + Math.max(0, choice.weight), 0);
  if (total <= 0) {
    return choices[0];
  }
  let threshold = rng() * total;
  for (const choice of choices) {
    threshold -= Math.max(0, choice.weight);
    if (threshold < 0) {
      return choice;
    }
  }
  return choices[choices.length - 1];
}
