import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

/**
 * In-process movement-model backend.
 *
 * The rest of the training subsystem (`runner.ts`, `execution-service.ts`)
 * generates launch plans for external, on-device training runtimes (mlx /
 * axolotl). Those cannot run in the cloud/CI and cannot be unit-tested here.
 *
 * This module provides the missing counterpart: a *pluggable, in-process*
 * movement model that can be trained on recorded movement sequences and used to
 * (a) repeat the recorded movements and (b) generalize to new-but-related
 * movements. The default `MarkovMovementBackend` is a deterministic n-gram
 * sequence model with stupid-backoff smoothing — it genuinely learns from a
 * dataset, is fully deterministic (so it is testable in the cloud), and serves
 * as the reference/mock backend behind the `MovementModelBackend` seam. A real
 * on-device small model can implement the same interface later.
 */

/** A canonical movement token, e.g. `device/tap:save-button` or `browser/scroll:down`. */
export type MovementToken = string;

/** A single recorded (or synthetic) movement sequence used for training/eval. */
export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

/** A ranked next-token candidate. */
export type MovementCandidate = {
  token: MovementToken;
  probability: number;
};

/** The result of predicting the next movement from a context prefix. */
export type MovementPrediction = {
  token: MovementToken;
  probability: number;
  /** How many context tokens were actually used (after backoff). */
  orderUsed: number;
  /** True when the model fell back to a shorter context than requested. */
  backedOff: boolean;
  /** Full ranked distribution for the chosen context (deterministic order). */
  candidates: MovementCandidate[];
};

export type TrainMovementOptions = {
  /** Maximum n-gram context length. Higher = more faithful repetition, less generalization. */
  maxOrder?: number;
};

export type GenerateMovementOptions = {
  /** Maximum number of tokens to emit (excluding the seed). */
  maxSteps?: number;
  /** Stop generating if the same token repeats this many times consecutively. */
  repetitionGuard?: number;
};

/** Serializable snapshot so a trained model can be persisted and reloaded. */
export type MovementModelSnapshot = {
  version: 1;
  backend: string;
  maxOrder: number;
  /** grams[order][contextKey][token] = observed count. order 0 uses "" as the context key. */
  grams: Record<string, Record<string, Record<MovementToken, number>>>;
};

export interface TrainedMovementModel {
  readonly backend: string;
  readonly maxOrder: number;
  /** Predict the single most-likely next movement given a context prefix. */
  predictNext(context: MovementToken[]): MovementPrediction | undefined;
  /** Continue a seed sequence by repeatedly predicting the next movement. */
  generate(seed: MovementToken[], options?: GenerateMovementOptions): MovementToken[];
  /** Serialize for persistence / transport. */
  snapshot(): MovementModelSnapshot;
}

export interface MovementModelBackend {
  readonly name: string;
  train(sequences: MovementSequence[], options?: TrainMovementOptions): TrainedMovementModel;
  load(snapshot: MovementModelSnapshot): TrainedMovementModel;
}

const CONTEXT_SEPARATOR = "";
const DEFAULT_MAX_ORDER = 3;
const DEFAULT_MAX_STEPS = 64;
const DEFAULT_REPETITION_GUARD = 4;

/**
 * Deterministic n-gram movement model with stupid-backoff.
 *
 * - Repetition: when the exact context has been seen, it reproduces the most
 *   frequent continuation, so recorded movements replay faithfully.
 * - Generalization: when the exact context is unseen, it backs off to shorter
 *   contexts (down to the unigram prior), so novel-but-related prefixes still
 *   yield a plausible next movement instead of nothing.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly name = "markov-ngram";

  train(sequences: MovementSequence[], options: TrainMovementOptions = {}): TrainedMovementModel {
    const maxOrder = normalizeMaxOrder(options.maxOrder);
    const grams: MovementModelSnapshot["grams"] = {};

    for (const sequence of sequences) {
      const tokens = sequence.tokens;
      for (let i = 0; i < tokens.length; i += 1) {
        const token = tokens[i]!;
        for (let order = 0; order <= maxOrder; order += 1) {
          if (i - order < 0) {
            break;
          }
          const context = tokens.slice(i - order, i);
          const contextKey = context.join(CONTEXT_SEPARATOR);
          const orderKey = String(order);
          const byContext = (grams[orderKey] ??= {});
          const byToken = (byContext[contextKey] ??= {});
          byToken[token] = (byToken[token] ?? 0) + 1;
        }
      }
    }

    return new MarkovMovementModel(this.name, maxOrder, grams);
  }

  load(snapshot: MovementModelSnapshot): TrainedMovementModel {
    return new MarkovMovementModel(
      snapshot.backend || this.name,
      normalizeMaxOrder(snapshot.maxOrder),
      snapshot.grams ?? {},
    );
  }
}

class MarkovMovementModel implements TrainedMovementModel {
  constructor(
    readonly backend: string,
    readonly maxOrder: number,
    private readonly grams: MovementModelSnapshot["grams"],
  ) {}

  predictNext(context: MovementToken[]): MovementPrediction | undefined {
    const startOrder = Math.min(context.length, this.maxOrder);
    for (let order = startOrder; order >= 0; order -= 1) {
      const contextSlice = order === 0 ? [] : context.slice(context.length - order);
      const contextKey = contextSlice.join(CONTEXT_SEPARATOR);
      const byToken = this.grams[String(order)]?.[contextKey];
      if (!byToken) {
        continue;
      }
      const candidates = rankCandidates(byToken);
      if (candidates.length === 0) {
        continue;
      }
      const best = candidates[0]!;
      return {
        token: best.token,
        probability: best.probability,
        orderUsed: order,
        backedOff: order < startOrder,
        candidates,
      };
    }
    return undefined;
  }

  generate(seed: MovementToken[], options: GenerateMovementOptions = {}): MovementToken[] {
    const maxSteps = options.maxSteps ?? DEFAULT_MAX_STEPS;
    const repetitionGuard = options.repetitionGuard ?? DEFAULT_REPETITION_GUARD;
    const output: MovementToken[] = [];
    const context = [...seed];
    let lastToken: MovementToken | undefined;
    let repeatCount = 0;

    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predictNext(context);
      if (!prediction) {
        break;
      }
      if (prediction.token === lastToken) {
        repeatCount += 1;
        if (repeatCount >= repetitionGuard) {
          break;
        }
      } else {
        repeatCount = 0;
      }
      output.push(prediction.token);
      context.push(prediction.token);
      lastToken = prediction.token;
    }

    return output;
  }

  snapshot(): MovementModelSnapshot {
    return {
      version: 1,
      backend: this.backend,
      maxOrder: this.maxOrder,
      grams: this.grams,
    };
  }
}

function normalizeMaxOrder(value: number | undefined): number {
  if (value === undefined || !Number.isFinite(value)) {
    return DEFAULT_MAX_ORDER;
  }
  return Math.max(0, Math.min(8, Math.floor(value)));
}

/** Rank next-token candidates deterministically: by count desc, then token asc. */
function rankCandidates(byToken: Record<MovementToken, number>): MovementCandidate[] {
  const entries = Object.entries(byToken);
  const total = entries.reduce((sum, [, count]) => sum + count, 0);
  if (total <= 0) {
    return [];
  }
  return entries
    .map(([token, count]) => ({ token, probability: count / total }))
    .sort((a, b) => {
      if (b.probability !== a.probability) {
        return b.probability - a.probability;
      }
      return a.token < b.token ? -1 : a.token > b.token ? 1 : 0;
    });
}

// ---------------------------------------------------------------------------
// Tokenization: turn recorded trajectory actions into movement tokens.
// ---------------------------------------------------------------------------

/**
 * Build a canonical movement token from a recorded trajectory action. Prefers
 * structured metadata (gesture / target / direction, as emitted by the device
 * and browser capture adapters); falls back to a slugified summary so any
 * action still tokenizes stably.
 */
export function movementTokenFromAction(action: TrajectoryAction): MovementToken {
  const tool = slug(action.tool) || "action";
  const metadata = action.metadata ?? {};
  const gesture = pickString(metadata, "gesture");
  const target = pickString(metadata, "target");
  const direction = pickString(metadata, "direction");

  const verb = gesture ? slug(gesture) : slug(action.summary) || "move";
  const qualifier = direction ? slug(direction) : target ? slug(target) : "";

  return qualifier ? `${tool}/${verb}:${qualifier}` : `${tool}/${verb}`;
}

/** Extract one movement sequence per trajectory span (chronologically ordered). */
export function movementSequencesFromTrajectories(trajectories: TrajectorySpan[]): MovementSequence[] {
  return trajectories
    .map((trajectory) => ({
      id: trajectory.id,
      tokens: [...trajectory.actions]
        .sort((a, b) => a.ts - b.ts)
        .map((action) => movementTokenFromAction(action)),
    }))
    .filter((sequence) => sequence.tokens.length > 0);
}

function pickString(metadata: Record<string, unknown>, key: string): string | undefined {
  const value = metadata[key];
  return typeof value === "string" && value.trim().length > 0 ? value : undefined;
}

function slug(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

// ---------------------------------------------------------------------------
// Generalization evaluation harness.
// ---------------------------------------------------------------------------

export type MovementEvalResult = {
  /** Total next-token predictions attempted (one per non-first token per sequence). */
  predictions: number;
  /** Predictions where the model's top-1 token matched the held-out truth. */
  correct: number;
  /** correct / predictions (0 when no predictions were possible). */
  accuracy: number;
  /** Fraction of correct predictions that required backing off to a shorter context. */
  generalizedShare: number;
};

/**
 * Measure next-token top-1 accuracy on held-out sequences. Reports how much of
 * the accuracy came from backoff (i.e. generalization to unseen exact
 * contexts) versus exact-context recall.
 */
export function evaluateNextTokenAccuracy(
  model: TrainedMovementModel,
  sequences: MovementSequence[],
): MovementEvalResult {
  let predictions = 0;
  let correct = 0;
  let generalizedCorrect = 0;

  for (const sequence of sequences) {
    for (let i = 1; i < sequence.tokens.length; i += 1) {
      const context = sequence.tokens.slice(0, i);
      const truth = sequence.tokens[i]!;
      const prediction = model.predictNext(context);
      predictions += 1;
      if (prediction && prediction.token === truth) {
        correct += 1;
        if (prediction.backedOff) {
          generalizedCorrect += 1;
        }
      }
    }
  }

  return {
    predictions,
    correct,
    accuracy: predictions === 0 ? 0 : correct / predictions,
    generalizedShare: correct === 0 ? 0 : generalizedCorrect / correct,
  };
}

// ---------------------------------------------------------------------------
// Synthetic movement-stream generator (deterministic; validates the pipeline
// without any real OS input capture).
// ---------------------------------------------------------------------------

export type SyntheticMovementOptions = {
  /** Seed for deterministic generation (no Math.random). */
  seed: number;
  /** Number of sequences to produce. */
  sequenceCount: number;
  /** Vocabulary of movement tokens to draw from. */
  vocabulary: MovementToken[];
  /** Min/max tokens per sequence. */
  minLength?: number;
  maxLength?: number;
  /**
   * Probability [0,1] of following the dominant "motif" transition rather than
   * a random token. Higher = more learnable structure. Default 0.8.
   */
  motifStrength?: number;
};

/**
 * Generate deterministic synthetic movement sequences with learnable structure:
 * a fixed motif chain (token i tends to be followed by token i+1) plus noise.
 * This lets the capture→dataset→train→replay/generalize pipeline be validated
 * end-to-end in the cloud without any real machine input.
 */
export function generateSyntheticMovementSequences(
  options: SyntheticMovementOptions,
): MovementSequence[] {
  const vocabulary = options.vocabulary;
  if (vocabulary.length < 2) {
    throw new Error("generateSyntheticMovementSequences requires a vocabulary of at least 2 tokens");
  }
  const minLength = Math.max(2, options.minLength ?? 4);
  const maxLength = Math.max(minLength, options.maxLength ?? 8);
  const motifStrength = clamp01(options.motifStrength ?? 0.8);
  const rng = mulberry32(options.seed >>> 0);
  const sequences: MovementSequence[] = [];

  for (let s = 0; s < options.sequenceCount; s += 1) {
    const length = minLength + Math.floor(rng() * (maxLength - minLength + 1));
    let index = Math.floor(rng() * vocabulary.length);
    const tokens: MovementToken[] = [];
    for (let t = 0; t < length; t += 1) {
      tokens.push(vocabulary[index]!);
      if (rng() < motifStrength) {
        index = (index + 1) % vocabulary.length;
      } else {
        index = Math.floor(rng() * vocabulary.length);
      }
    }
    sequences.push({ id: `synthetic-${options.seed}-${s}`, tokens });
  }

  return sequences;
}

function clamp01(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.max(0, Math.min(1, value));
}

/** Small deterministic PRNG (no global state, no Math.random). */
function mulberry32(seed: number): () => number {
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
