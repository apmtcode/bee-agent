// Local-movement learning subsystem — pluggable model backend.
//
// Standing objective #2 (c)/(d): once movements are recorded to a replayable
// dataset, bee-agent must be able to *post-train a local model to repeat the
// recorded movements* and *generalize to new but related movements*.
//
// This module is the in-process, deterministic core of that pipeline:
//   1. A tokenizer turning recorded trajectory actions into a discrete
//      movement-token stream (`tokenizeAction`, `buildMovementDataset*`).
//   2. A pluggable `MovementModelBackend` seam so a real on-device small model
//      (mlx/gguf, etc.) can be dropped in later without touching call sites.
//   3. A deterministic reference backend (`MarkovMovementBackend`) — an order-N
//      smoothed n-gram — that actually learns transition statistics, can
//      *repeat* a recorded sequence (greedy arg-max) and *generalize* to novel
//      but related sequences (seeded sampling). It runs fully in the cloud/CI
//      with no OS input and no external process, so tests are deterministic.
//   4. A generalization eval harness (`evaluateGeneralization`) that measures
//      replay fidelity on held-out sequences (next-token top-1 accuracy +
//      perplexity), fulfilling the roadmap's eval item.
//
// Determinism note: this file uses a seeded PRNG (`createSeededRng`) and never
// calls `Math.random`/`Date.now`, so every result is reproducible.

import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

/** A discrete movement token, e.g. `tap:submit`, `swipe:down`, `type:search`. */
export type MovementToken = string;

/** Sentinel token marking the start of a sequence (context padding). */
export const MOVEMENT_START: MovementToken = "START";
/** Sentinel token marking the end of a sequence (generation stop signal). */
export const MOVEMENT_END: MovementToken = "END";

export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
  /** Optional provenance so eval output can point back at a trajectory. */
  sourceId?: string;
};

export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

/** A single next-token decision with its probability and ranked alternatives. */
export type MovementPrediction = {
  token: MovementToken;
  probability: number;
  alternatives: Array<{ token: MovementToken; probability: number }>;
};

/**
 * Pluggable backend seam. The reference implementation is in-process, but a
 * real on-device backend would implement the same shape (train → artifact,
 * load → model) while delegating to mlx/axolotl/gguf under the hood.
 */
export interface MovementModelBackend<Artifact = unknown> {
  readonly id: string;
  train(dataset: MovementDataset): Promise<Artifact>;
  load(artifact: Artifact): MovementModel;
}

export interface MovementModel {
  /** Vocabulary the model was trained on (excludes START, includes END). */
  readonly vocabulary: MovementToken[];
  /** Highest-probability next token for a context (deterministic arg-max). */
  predict(context: MovementToken[]): MovementPrediction;
  /** Probability-weighted next token using a seeded PRNG (generalization). */
  sample(context: MovementToken[], rng: () => number): MovementPrediction;
  /** Smoothed log-probability of a full token sequence. */
  logLikelihood(tokens: MovementToken[]): number;
  /** exp(mean negative log-likelihood) — lower is a better fit. */
  perplexity(tokens: MovementToken[]): number;
}

// ---------------------------------------------------------------------------
// Seeded PRNG (mulberry32) — deterministic, no Math.random.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Tokenizer: recorded trajectory action -> movement token.
// ---------------------------------------------------------------------------

function slug(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 48);
}

/**
 * Derive a stable discrete token from a recorded action. Prefers the structured
 * gesture metadata (kind + target/direction) captured by the device/os/browser
 * adapters, falling back to the tool name and a slug of the human summary.
 */
export function tokenizeAction(action: TrajectoryAction): MovementToken {
  const metadata = action.metadata ?? {};
  const gesture = typeof metadata.gesture === "string" ? metadata.gesture : undefined;
  const target = typeof metadata.target === "string" ? metadata.target : undefined;
  const direction = typeof metadata.direction === "string" ? metadata.direction : undefined;
  const qualifier = target ?? direction;
  if (gesture) {
    return qualifier ? `${gesture}:${slug(qualifier)}` : gesture;
  }
  const tool = slug(action.tool) || "action";
  const summary = slug(action.summary);
  return summary ? `${tool}:${summary}` : tool;
}

export function tokenizeTrajectory(span: TrajectorySpan): MovementSequence {
  const tokens = [...span.actions]
    .sort((a, b) => a.ts - b.ts)
    .map(tokenizeAction);
  return { id: span.id, tokens, sourceId: span.sessionId };
}

export function buildMovementDataset(spans: TrajectorySpan[]): MovementDataset {
  return {
    version: 1,
    sequences: spans.map(tokenizeTrajectory).filter((sequence) => sequence.tokens.length > 0),
  };
}

// ---------------------------------------------------------------------------
// Deterministic reference backend: order-N smoothed n-gram Markov model.
// ---------------------------------------------------------------------------

export type MarkovMovementArtifact = {
  version: 1;
  kind: "markov-movement";
  order: number;
  smoothing: number;
  vocabulary: MovementToken[];
  /** context-key ("a b") -> next-token -> observed count. */
  transitions: Record<string, Record<MovementToken, number>>;
};

export type MarkovMovementBackendOptions = {
  /** Number of prior tokens used as context (default 2). */
  order?: number;
  /** Add-k Laplace smoothing constant (default 0.05). */
  smoothing?: number;
};

const CONTEXT_SEPARATOR = " ";

export class MarkovMovementBackend implements MovementModelBackend<MarkovMovementArtifact> {
  readonly id = "markov-movement";
  private readonly order: number;
  private readonly smoothing: number;

  constructor(options: MarkovMovementBackendOptions = {}) {
    this.order = Math.max(1, Math.floor(options.order ?? 2));
    this.smoothing = options.smoothing ?? 0.05;
    if (!(this.smoothing > 0)) {
      throw new Error("smoothing must be > 0 to keep unseen movements reachable");
    }
  }

  async train(dataset: MovementDataset): Promise<MarkovMovementArtifact> {
    const transitions: Record<string, Record<MovementToken, number>> = {};
    const vocabulary = new Set<MovementToken>([MOVEMENT_END]);

    for (const sequence of dataset.sequences) {
      const padded = [
        ...Array<MovementToken>(this.order).fill(MOVEMENT_START),
        ...sequence.tokens,
        MOVEMENT_END,
      ];
      for (const token of sequence.tokens) {
        vocabulary.add(token);
      }
      for (let i = this.order; i < padded.length; i += 1) {
        const context = padded.slice(i - this.order, i);
        const next = padded[i]!;
        const key = context.join(CONTEXT_SEPARATOR);
        const bucket = (transitions[key] ??= {});
        bucket[next] = (bucket[next] ?? 0) + 1;
      }
    }

    return {
      version: 1,
      kind: "markov-movement",
      order: this.order,
      smoothing: this.smoothing,
      vocabulary: [...vocabulary].sort(),
      transitions,
    };
  }

  load(artifact: MarkovMovementArtifact): MovementModel {
    return new MarkovMovementModel(artifact);
  }
}

class MarkovMovementModel implements MovementModel {
  readonly vocabulary: MovementToken[];
  private readonly order: number;
  private readonly smoothing: number;
  private readonly transitions: Record<string, Record<MovementToken, number>>;
  private readonly vocabSize: number;

  constructor(artifact: MarkovMovementArtifact) {
    this.order = artifact.order;
    this.smoothing = artifact.smoothing;
    this.transitions = artifact.transitions;
    this.vocabulary = artifact.vocabulary;
    this.vocabSize = artifact.vocabulary.length;
  }

  private contextKey(context: MovementToken[]): string {
    const trimmed = context.slice(-this.order);
    const padded =
      trimmed.length < this.order
        ? [...Array<MovementToken>(this.order - trimmed.length).fill(MOVEMENT_START), ...trimmed]
        : trimmed;
    return padded.join(CONTEXT_SEPARATOR);
  }

  /** Smoothed distribution over the full vocabulary for a context. */
  private distribution(context: MovementToken[]): Array<{ token: MovementToken; probability: number }> {
    const counts = this.transitions[this.contextKey(context)] ?? {};
    let total = 0;
    for (const token of this.vocabulary) {
      total += (counts[token] ?? 0) + this.smoothing;
    }
    const distribution = this.vocabulary.map((token) => ({
      token,
      probability: ((counts[token] ?? 0) + this.smoothing) / total,
    }));
    distribution.sort((a, b) =>
      b.probability !== a.probability ? b.probability - a.probability : a.token < b.token ? -1 : 1,
    );
    return distribution;
  }

  predict(context: MovementToken[]): MovementPrediction {
    const distribution = this.distribution(context);
    const [top, ...rest] = distribution;
    if (!top) {
      throw new Error("model has an empty vocabulary; train on at least one movement");
    }
    return { token: top.token, probability: top.probability, alternatives: rest.slice(0, 4) };
  }

  sample(context: MovementToken[], rng: () => number): MovementPrediction {
    const distribution = this.distribution(context);
    const roll = rng();
    let cumulative = 0;
    let chosen = distribution[0]!;
    for (const entry of distribution) {
      cumulative += entry.probability;
      if (roll <= cumulative) {
        chosen = entry;
        break;
      }
    }
    const alternatives = distribution.filter((entry) => entry.token !== chosen.token).slice(0, 4);
    return { token: chosen.token, probability: chosen.probability, alternatives };
  }

  logLikelihood(tokens: MovementToken[]): number {
    const padded = [...Array<MovementToken>(this.order).fill(MOVEMENT_START), ...tokens, MOVEMENT_END];
    let sum = 0;
    for (let i = this.order; i < padded.length; i += 1) {
      const context = padded.slice(i - this.order, i);
      const next = padded[i]!;
      const probability = this.distribution(context).find((entry) => entry.token === next)?.probability;
      // Unseen tokens are not in the vocabulary; charge them the smoothing floor.
      const safe = probability ?? this.smoothing / (this.smoothing * (this.vocabSize + 1));
      sum += Math.log(safe);
    }
    return sum;
  }

  perplexity(tokens: MovementToken[]): number {
    const steps = tokens.length + 1; // +1 for the END transition
    if (steps <= 0) {
      return 1;
    }
    return Math.exp(-this.logLikelihood(tokens) / steps);
  }
}

// ---------------------------------------------------------------------------
// Inference: repeat recorded movements (2c) and generalize (2d).
// ---------------------------------------------------------------------------

export type RepeatOptions = {
  /** Priming tokens; defaults to empty (model starts from START padding). */
  seed?: MovementToken[];
  /** Hard cap on generated length to guarantee termination. */
  maxLength?: number;
};

/**
 * Deterministically replay the dominant learned continuation from a seed —
 * this is the "repeat the recorded movements" capability. Because it always
 * takes the arg-max, a model trained on a single trajectory reproduces it
 * exactly.
 */
export function repeatMovements(model: MovementModel, options: RepeatOptions = {}): MovementToken[] {
  const seed = options.seed ?? [];
  const maxLength = options.maxLength ?? 128;
  const output: MovementToken[] = [...seed];
  const context: MovementToken[] = [...seed];
  while (output.length < maxLength) {
    const next = model.predict(context).token;
    if (next === MOVEMENT_END) {
      break;
    }
    output.push(next);
    context.push(next);
  }
  return output;
}

export type GenerateOptions = RepeatOptions & {
  rng: () => number;
};

/**
 * Sample a novel-but-related movement sequence using a seeded PRNG — this is
 * the "generalize to new but related movements" capability. The same seed
 * produces the same sequence, so it is testable.
 */
export function generateMovements(model: MovementModel, options: GenerateOptions): MovementToken[] {
  const seed = options.seed ?? [];
  const maxLength = options.maxLength ?? 128;
  const output: MovementToken[] = [...seed];
  const context: MovementToken[] = [...seed];
  while (output.length < maxLength) {
    const next = model.sample(context, options.rng).token;
    if (next === MOVEMENT_END) {
      break;
    }
    output.push(next);
    context.push(next);
  }
  return output;
}

// ---------------------------------------------------------------------------
// Generalization eval harness: replay fidelity on held-out sequences.
// ---------------------------------------------------------------------------

export type MovementEvalResult = {
  sequenceCount: number;
  tokenCount: number;
  /** Fraction of next-token decisions where arg-max matched the held-out truth. */
  top1Accuracy: number;
  /** Mean perplexity across held-out sequences (lower is better). */
  meanPerplexity: number;
  perSequence: Array<{ id: string; top1Accuracy: number; perplexity: number }>;
};

/**
 * Measure how well a model predicts held-out (unseen) movement sequences.
 * For each sequence we walk its true tokens, asking the model to predict the
 * next token from the true prefix, and score arg-max top-1 accuracy plus the
 * sequence perplexity. This is the roadmap's "generalization eval harness".
 */
export function evaluateGeneralization(
  model: MovementModel,
  heldOut: MovementSequence[],
): MovementEvalResult {
  let totalTokens = 0;
  let totalCorrect = 0;
  let perplexitySum = 0;
  const perSequence: MovementEvalResult["perSequence"] = [];

  for (const sequence of heldOut) {
    const truth = [...sequence.tokens, MOVEMENT_END];
    let correct = 0;
    for (let i = 0; i < truth.length; i += 1) {
      const context = sequence.tokens.slice(0, i);
      if (model.predict(context).token === truth[i]) {
        correct += 1;
      }
    }
    const perplexity = model.perplexity(sequence.tokens);
    const accuracy = truth.length > 0 ? correct / truth.length : 0;
    perSequence.push({ id: sequence.id, top1Accuracy: accuracy, perplexity });
    totalTokens += truth.length;
    totalCorrect += correct;
    perplexitySum += perplexity;
  }

  return {
    sequenceCount: heldOut.length,
    tokenCount: totalTokens,
    top1Accuracy: totalTokens > 0 ? totalCorrect / totalTokens : 0,
    meanPerplexity: heldOut.length > 0 ? perplexitySum / heldOut.length : 0,
    perSequence,
  };
}

// ---------------------------------------------------------------------------
// Synthetic event-stream generator: validate capture->dataset->replay->train
// round-trips with no real OS input. Deterministic given a seed.
// ---------------------------------------------------------------------------

export type MovementTemplate = {
  name: string;
  /** Ordered motif of movement tokens describing the task's happy path. */
  motif: MovementToken[];
  /**
   * Optional per-step alternatives the generator may substitute to produce
   * "related but new" variants (drives the generalization story).
   */
  variants?: Record<number, MovementToken[]>;
};

export type SynthesizeOptions = {
  templates: MovementTemplate[];
  /** Sequences to emit per template (default 4). */
  perTemplate?: number;
  seed?: number;
  /** Chance [0,1) a step is swapped for a variant when one exists (default 0.35). */
  variantRate?: number;
};

/**
 * Produce a deterministic dataset of related movement sequences from task
 * templates. Each emitted sequence follows its template's motif but may
 * substitute variant tokens, yielding a corpus that is self-similar enough for
 * an n-gram model to learn yet varied enough to exercise generalization.
 */
export function synthesizeMovementDataset(options: SynthesizeOptions): MovementDataset {
  const perTemplate = Math.max(1, Math.floor(options.perTemplate ?? 4));
  const variantRate = options.variantRate ?? 0.35;
  const rng = createSeededRng(options.seed ?? 1);
  const sequences: MovementSequence[] = [];

  options.templates.forEach((template, templateIndex) => {
    for (let copy = 0; copy < perTemplate; copy += 1) {
      const tokens = template.motif.map((token, step) => {
        const variants = template.variants?.[step];
        if (variants && variants.length > 0 && rng() < variantRate) {
          const pick = Math.min(variants.length - 1, Math.floor(rng() * variants.length));
          return variants[pick]!;
        }
        return token;
      });
      sequences.push({
        id: `${template.name}-${templateIndex}-${copy}`,
        tokens,
        sourceId: template.name,
      });
    }
  });

  return { version: 1, sequences };
}

/** Split a dataset into train/holdout by a deterministic stride (no RNG). */
export function splitMovementDataset(
  dataset: MovementDataset,
  holdoutEvery = 4,
): { train: MovementSequence[]; holdout: MovementSequence[] } {
  const train: MovementSequence[] = [];
  const holdout: MovementSequence[] = [];
  dataset.sequences.forEach((sequence, index) => {
    if (holdoutEvery > 0 && (index + 1) % holdoutEvery === 0) {
      holdout.push(sequence);
    } else {
      train.push(sequence);
    }
  });
  return { train, holdout };
}
