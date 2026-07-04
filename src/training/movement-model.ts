import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

/**
 * Pluggable local-movement model subsystem.
 *
 * Objective #2(c)+(d): post-train a *local* model on recorded movements so it can
 * (a) repeat the recorded movements and (b) generalize to new-but-related ones.
 *
 * This module is fully in-process and deterministic so it runs (and is tested) in
 * the cloud with synthetic event streams — the real on-device training backend
 * (mlx/axolotl, see `runner.ts`) plugs into the same {@link MovementModelBackend}
 * seam. No `Math.random`, no wall-clock: identical dataset ⇒ identical snapshot.
 */

/** A discrete, learnable unit of movement derived from a captured action. */
export type MovementToken = string;

/** One (context → next-movement) training example. */
export type MovementSample = {
  context: MovementToken[];
  next: MovementToken;
};

/** An ordered movement dataset plus the per-trajectory token sequences it came from. */
export type MovementDataset = {
  samples: MovementSample[];
  sequences: MovementToken[][];
  contextWindow: number;
};

/** A trained, serializable model artifact. Backends may extend this shape. */
export type MovementModelSnapshot = {
  backend: string;
  version: 1;
};

/** Result of asking a trained model what movement comes next. */
export type MovementPrediction = {
  /** Most-likely next movement, or `null` when the model is empty. */
  token: MovementToken | null;
  /** Probability of `token` at the matched context order, discounted by backoff. */
  confidence: number;
  /** Context length actually matched (0 = unigram fallback). */
  order: number;
  /** How many orders the model had to back off to find a match. */
  backoffSteps: number;
  /** All candidate next-movements, most-likely first. */
  candidates: Array<{ token: MovementToken; probability: number }>;
};

/**
 * The backend seam. A backend turns a {@link MovementSample} dataset into a
 * serializable snapshot and answers next-movement queries from that snapshot.
 * `predict` is pure over `(snapshot, context)` so snapshots can be persisted and
 * reloaded — a real on-device backend returns a handle/artifact path instead.
 */
export interface MovementModelBackend<S extends MovementModelSnapshot = MovementModelSnapshot> {
  readonly name: string;
  train(dataset: MovementSample[]): S;
  predict(snapshot: S, context: MovementToken[]): MovementPrediction;
}

const DEFAULT_CONTEXT_WINDOW = 3;
/** Stupid-backoff discount applied per order dropped, à la Brants et al. 2007. */
const BACKOFF_DISCOUNT = 0.4;

/** Deterministically derive a movement token from a captured trajectory action. */
export function tokenizeAction(action: TrajectoryAction): MovementToken {
  const meta = action.metadata ?? {};
  const gesture = readString(meta.gesture);
  const direction = readString(meta.direction);
  const target = readString(meta.target);
  const descriptor = gesture ?? slug(action.summary) ?? "act";
  const parts = [slug(action.tool) ?? "tool", descriptor];
  if (direction) {
    parts.push(`/${direction}`);
  }
  if (target) {
    parts.push(`@${slug(target)}`);
  }
  return parts.join("");
}

/** Ordered movement tokens for a single trajectory (actions sorted by timestamp). */
export function trajectoryToMovementTokens(trajectory: TrajectorySpan): MovementToken[] {
  return [...trajectory.actions]
    .sort((a, b) => a.ts - b.ts)
    .map((action) => tokenizeAction(action));
}

/** Build a sliding-window (context → next) dataset from recorded trajectories. */
export function buildMovementDataset(
  trajectories: TrajectorySpan[],
  options: { contextWindow?: number } = {},
): MovementDataset {
  const contextWindow = normalizeWindow(options.contextWindow);
  const sequences = trajectories
    .map((trajectory) => trajectoryToMovementTokens(trajectory))
    .filter((sequence) => sequence.length > 0);
  return {
    samples: buildSamplesFromSequences(sequences, contextWindow),
    sequences,
    contextWindow,
  };
}

/** Build samples directly from already-tokenized movement sequences. */
export function buildSamplesFromSequences(
  sequences: MovementToken[][],
  contextWindow: number = DEFAULT_CONTEXT_WINDOW,
): MovementSample[] {
  const window = normalizeWindow(contextWindow);
  const samples: MovementSample[] = [];
  for (const sequence of sequences) {
    for (let index = 0; index < sequence.length; index += 1) {
      const start = Math.max(0, index - window);
      samples.push({
        context: sequence.slice(start, index),
        next: sequence[index]!,
      });
    }
  }
  return samples;
}

type GramEntry = { total: number; counts: Record<MovementToken, number> };

/** Serializable snapshot for {@link MarkovMovementBackend}. */
export type MarkovMovementSnapshot = MovementModelSnapshot & {
  backend: "markov-backoff";
  maxOrder: number;
  vocabulary: MovementToken[];
  sampleCount: number;
  /** context-suffix key → next-movement counts, keyed as `order|t0>t1>…`. */
  grams: Record<string, GramEntry>;
};

/**
 * Variable-order Markov backend with stupid backoff.
 *
 * - **Repeat:** a movement sequence seen in training is continued at its highest
 *   matching order, so recorded flows are reproduced faithfully.
 * - **Generalize:** an unseen full context backs off to shorter suffixes shared
 *   with *related* trajectories, yielding a plausible next movement (with a
 *   discounted confidence) instead of failing.
 *
 * Fully deterministic: argmax with lexicographic tie-break, no randomness.
 */
export class MarkovMovementBackend implements MovementModelBackend<MarkovMovementSnapshot> {
  readonly name = "markov-backoff";

  constructor(private readonly maxOrder: number = DEFAULT_CONTEXT_WINDOW) {}

  train(dataset: MovementSample[]): MarkovMovementSnapshot {
    const grams: Record<string, GramEntry> = {};
    const vocabulary = new Set<MovementToken>();
    for (const sample of dataset) {
      vocabulary.add(sample.next);
      for (const token of sample.context) {
        vocabulary.add(token);
      }
      const maxK = Math.min(this.maxOrder, sample.context.length);
      for (let order = 0; order <= maxK; order += 1) {
        const suffix = sample.context.slice(sample.context.length - order);
        const key = gramKey(order, suffix);
        const entry = (grams[key] ??= { total: 0, counts: {} });
        entry.counts[sample.next] = (entry.counts[sample.next] ?? 0) + 1;
        entry.total += 1;
      }
    }
    return {
      backend: this.name,
      version: 1,
      maxOrder: this.maxOrder,
      vocabulary: [...vocabulary].sort(),
      sampleCount: dataset.length,
      grams,
    };
  }

  predict(snapshot: MarkovMovementSnapshot, context: MovementToken[]): MovementPrediction {
    const startOrder = Math.min(snapshot.maxOrder, context.length);
    for (let order = startOrder; order >= 0; order -= 1) {
      const suffix = context.slice(context.length - order);
      const entry = snapshot.grams[gramKey(order, suffix)];
      if (!entry || entry.total === 0) {
        continue;
      }
      const candidates = Object.entries(entry.counts)
        .map(([token, count]) => ({ token, probability: count / entry.total }))
        .sort((a, b) => (b.probability - a.probability) || (a.token < b.token ? -1 : 1));
      const backoffSteps = startOrder - order;
      const best = candidates[0]!;
      return {
        token: best.token,
        confidence: best.probability * BACKOFF_DISCOUNT ** backoffSteps,
        order,
        backoffSteps,
        candidates,
      };
    }
    return { token: null, confidence: 0, order: 0, backoffSteps: startOrder, candidates: [] };
  }
}

/**
 * Roll the model forward from a seed context to "repeat"/continue a movement flow.
 * Stops early when the model can no longer predict (empty model).
 */
export function rolloutMovements<S extends MovementModelSnapshot>(
  backend: MovementModelBackend<S>,
  snapshot: S,
  seed: MovementToken[],
  steps: number,
): MovementToken[] {
  const generated: MovementToken[] = [];
  const context = [...seed];
  for (let step = 0; step < steps; step += 1) {
    const prediction = backend.predict(snapshot, context);
    if (prediction.token === null) {
      break;
    }
    generated.push(prediction.token);
    context.push(prediction.token);
  }
  return generated;
}

export type MovementEvalResult = {
  /** Held-out next-movement predictions attempted. */
  total: number;
  /** Correct top-1 predictions. */
  correct: number;
  /** `correct / total` (0 when nothing was evaluated). */
  accuracy: number;
  /** Fraction of predictions that matched a context of order ≥ 1 (not pure unigram). */
  informedFraction: number;
  /** Mean confidence over evaluated positions. */
  meanConfidence: number;
};

/**
 * Generalization eval: measure next-movement top-1 accuracy on held-out (but
 * related) synthetic trajectories. `informedFraction` reports how often the model
 * generalized via a real context match versus falling back to the unigram prior.
 */
export function evaluateMovementModel<S extends MovementModelSnapshot>(
  backend: MovementModelBackend<S>,
  snapshot: S,
  heldOut: MovementToken[][],
  options: { contextWindow?: number } = {},
): MovementEvalResult {
  const window = normalizeWindow(options.contextWindow);
  let total = 0;
  let correct = 0;
  let informed = 0;
  let confidenceSum = 0;
  for (const sequence of heldOut) {
    for (let index = 1; index < sequence.length; index += 1) {
      const context = sequence.slice(Math.max(0, index - window), index);
      const prediction = backend.predict(snapshot, context);
      total += 1;
      confidenceSum += prediction.confidence;
      if (prediction.order >= 1) {
        informed += 1;
      }
      if (prediction.token === sequence[index]) {
        correct += 1;
      }
    }
  }
  return {
    total,
    correct,
    accuracy: total === 0 ? 0 : correct / total,
    informedFraction: total === 0 ? 0 : informed / total,
    meanConfidence: total === 0 ? 0 : confidenceSum / total,
  };
}

function gramKey(order: number, suffix: MovementToken[]): string {
  return `${order}|${suffix.join(">")}`;
}

function normalizeWindow(value: number | undefined): number {
  if (value === undefined || !Number.isFinite(value) || value < 1) {
    return DEFAULT_CONTEXT_WINDOW;
  }
  return Math.floor(value);
}

function readString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim().length > 0 ? value.trim() : undefined;
}

function slug(value: string | undefined): string | undefined {
  if (value === undefined) {
    return undefined;
  }
  const normalized = value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return normalized.length > 0 ? normalized : undefined;
}
