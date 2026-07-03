import type { MovementDataset, MovementSequence, MovementToken, TrainedMovementModel } from "./model-backend.js";

/**
 * Synthetic movement-stream generator and generalization eval harness.
 *
 * bee-agent runs in the cloud with no access to the user's real mouse/keyboard,
 * so the movement-learning loop is validated against *simulated* event streams.
 * These generators emit deterministic, grammar-structured token sequences (via a
 * seeded PRNG) so tests can (1) train a model on one draw and (2) measure how
 * well it predicts a held-out draw from the same grammar — the operational
 * definition of "generalize to new but related movements" (objective #2d).
 */

/** Deterministic, seedable PRNG (mulberry32). Avoids Math.random for reproducibility. */
export function createSeededRng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state |= 0;
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * A movement grammar: a weighted transition table over primitive tokens plus a
 * set of valid start tokens. Sequences are random walks that terminate on the
 * synthetic END choice.
 */
export type MovementGrammar = {
  start: MovementToken[];
  transitions: Record<MovementToken, Array<{ token: MovementToken; weight: number }>>;
  /** Absorbing token that ends a walk. */
  terminal: MovementToken;
};

/**
 * A small but non-trivial default grammar modeling a "select an item and act on
 * it" desktop interaction: hover/scroll to locate, click to select, then either
 * drag or open, then finish. Related-but-varied sequences share this structure.
 */
export const DEFAULT_MOVEMENT_GRAMMAR: MovementGrammar = {
  start: ["focus-window", "scroll-up", "scroll-down"],
  transitions: {
    "focus-window": [
      { token: "move-right", weight: 3 },
      { token: "move-left", weight: 2 },
    ],
    "scroll-up": [
      { token: "move-right", weight: 2 },
      { token: "scroll-up", weight: 1 },
      { token: "click", weight: 2 },
    ],
    "scroll-down": [
      { token: "move-left", weight: 2 },
      { token: "scroll-down", weight: 1 },
      { token: "click", weight: 2 },
    ],
    "move-right": [
      { token: "click", weight: 4 },
      { token: "move-right", weight: 1 },
    ],
    "move-left": [
      { token: "click", weight: 4 },
      { token: "move-left", weight: 1 },
    ],
    click: [
      { token: "drag", weight: 2 },
      { token: "open", weight: 2 },
      { token: "__end__", weight: 1 },
    ],
    drag: [
      { token: "drop", weight: 5 },
    ],
    drop: [
      { token: "__end__", weight: 4 },
      { token: "click", weight: 1 },
    ],
    open: [
      { token: "type", weight: 3 },
      { token: "__end__", weight: 2 },
    ],
    type: [
      { token: "__end__", weight: 3 },
      { token: "type", weight: 1 },
    ],
  },
  terminal: "__end__",
};

function weightedPick(
  choices: Array<{ token: MovementToken; weight: number }>,
  rng: () => number,
): MovementToken {
  const total = choices.reduce((sum, choice) => sum + choice.weight, 0);
  let threshold = rng() * total;
  for (const choice of choices) {
    threshold -= choice.weight;
    if (threshold <= 0) {
      return choice.token;
    }
  }
  return choices[choices.length - 1]!.token;
}

export type GenerateSyntheticParams = {
  seed: number;
  count: number;
  grammar?: MovementGrammar;
  maxLength?: number;
  idPrefix?: string;
};

/** Draw `count` grammar-structured movement sequences deterministically. */
export function generateSyntheticMovementSequences(params: GenerateSyntheticParams): MovementSequence[] {
  const grammar = params.grammar ?? DEFAULT_MOVEMENT_GRAMMAR;
  const maxLength = params.maxLength ?? 24;
  const prefix = params.idPrefix ?? "synthetic";
  const rng = createSeededRng(params.seed);
  const sequences: MovementSequence[] = [];

  for (let index = 0; index < params.count; index += 1) {
    const tokens: MovementToken[] = [];
    let current = grammar.start[Math.floor(rng() * grammar.start.length)] ?? grammar.start[0]!;
    while (tokens.length < maxLength) {
      tokens.push(current);
      const next = grammar.transitions[current];
      if (!next || next.length === 0) {
        break;
      }
      const chosen = weightedPick(next, rng);
      if (chosen === grammar.terminal) {
        break;
      }
      current = chosen;
    }
    sequences.push({ id: `${prefix}-${index}`, tokens });
  }

  return sequences;
}

export function generateSyntheticDataset(params: GenerateSyntheticParams): MovementDataset {
  return { sequences: generateSyntheticMovementSequences(params) };
}

// --- Generalization eval --------------------------------------------------

export type GeneralizationReport = {
  sequenceCount: number;
  /** Positions scored (each token given its preceding context). */
  tokenCount: number;
  /** Fraction of positions where greedy prediction matched the held-out token. */
  nextTokenAccuracy: number;
  /** Mean per-token log-probability (base e); higher (less negative) is better. */
  meanLogProb: number;
  /** exp(-meanLogProb); lower is better. */
  perplexity: number;
};

/**
 * Measure how well a trained model reproduces / generalizes to a set of
 * sequences. Run on the training draw for fidelity, on a held-out draw for
 * generalization.
 */
export function evaluateGeneralization(
  model: TrainedMovementModel,
  sequences: MovementSequence[],
): GeneralizationReport {
  let correct = 0;
  let scoredPositions = 0;
  let totalLogProb = 0;

  for (const sequence of sequences) {
    if (sequence.tokens.length === 0) {
      continue;
    }
    for (let i = 0; i < sequence.tokens.length; i += 1) {
      const context = sequence.tokens.slice(0, i);
      const actual = sequence.tokens[i]!;
      const prediction = model.predictNext(context);
      if (prediction && prediction.token === actual) {
        correct += 1;
      }
      scoredPositions += 1;
    }
    // Per-token log-prob over the full sequence (includes the END transition).
    totalLogProb += model.sequenceLogProb(sequence.tokens) / (sequence.tokens.length + 1);
  }

  const meanLogProb = sequences.length > 0 ? totalLogProb / sequences.length : 0;
  return {
    sequenceCount: sequences.length,
    tokenCount: scoredPositions,
    nextTokenAccuracy: scoredPositions > 0 ? correct / scoredPositions : 0,
    meanLogProb,
    perplexity: Number.isFinite(meanLogProb) ? Math.exp(-meanLogProb) : Number.POSITIVE_INFINITY,
  };
}
