/**
 * Local-movement learning model.
 *
 * This module closes the loop on the movement-learning subsystem's objectives
 * (c) "post-train a local model to repeat recorded movements" and (d)
 * "generalize to new but related movements". The runner (`runner.ts`) emits
 * launch plans for real on-device trainers (mlx/axolotl); this module provides
 * a *pluggable backend interface* plus a deterministic, dependency-free backend
 * so the whole pipeline — dataset -> train -> predict/generate -> eval — can be
 * exercised in the cloud/CI without any real OS input or GPU.
 *
 * The default `MarkovMovementBackend` is a variable-order Markov (n-gram) model
 * with stupid-backoff: it reproduces recorded movement sequences exactly when a
 * context has been seen, and generalizes to related movements by backing off to
 * shorter context suffixes when the exact context is novel. Everything is pure
 * and deterministic (no `Math.random`, no clock), so training and inference are
 * fully reproducible and the model state serializes to plain JSON.
 */

/** A discrete movement token — the atomic unit the model learns over. */
export type MovementToken = string;

/** The minimal shape of a recorded action needed to derive a movement token. */
export type MovementActionLike = {
  tool: string;
  summary: string;
  metadata?: Record<string, unknown>;
};

/**
 * Maps a recorded action to a movement token. The default tokenizer prefers the
 * structured gesture metadata emitted by the device adapter (kind/direction/
 * target) so that movements — not prose summaries — drive the vocabulary, and
 * falls back to a slug of `tool:summary` for non-gesture actions.
 */
export type MovementTokenizer = (action: MovementActionLike) => MovementToken;

export const defaultMovementTokenizer: MovementTokenizer = (action) => {
  const metadata = action.metadata ?? {};
  const gesture = typeof metadata.gesture === "string" ? metadata.gesture : undefined;
  if (gesture) {
    const qualifier =
      pickString(metadata.direction) ?? pickString(metadata.target) ?? pickString(metadata.valueSummary);
    return qualifier ? `${gesture}:${slug(qualifier)}` : gesture;
  }
  return `${slug(action.tool)}:${slug(action.summary)}`;
};

/** One training example: the ordered movement tokens of a single trajectory. */
export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

/** A dataset of movement sequences ready to train a movement model. */
export type MovementDataset = {
  version: 1;
  /** Recommended maximum context order (n-gram length) for training. */
  order: number;
  /** Sorted, de-duplicated set of all tokens observed across sequences. */
  vocabulary: MovementToken[];
  sequences: MovementSequence[];
};

export const MOVEMENT_START_TOKEN = "START";
const DEFAULT_ORDER = 3;

/**
 * Builds a `MovementDataset` from arbitrary action sources. Each source becomes
 * one sequence; empty sequences are dropped so held-out splits stay meaningful.
 */
export function buildMovementDataset(params: {
  sources: { id: string; actions: MovementActionLike[] }[];
  order?: number;
  tokenizer?: MovementTokenizer;
}): MovementDataset {
  const order = normalizeOrder(params.order);
  const tokenizer = params.tokenizer ?? defaultMovementTokenizer;
  const sequences: MovementSequence[] = [];
  const vocabulary = new Set<MovementToken>();

  for (const source of params.sources) {
    const tokens = source.actions.map((action) => tokenizer(action));
    if (tokens.length === 0) {
      continue;
    }
    for (const token of tokens) {
      vocabulary.add(token);
    }
    sequences.push({ id: source.id, tokens });
  }

  return {
    version: 1,
    order,
    vocabulary: [...vocabulary].sort(),
    sequences,
  };
}

/**
 * Convenience adapter: build a dataset directly from replay-timeline events
 * (as produced by `buildReplayManifest`). Only `action` events carry movements.
 */
export function buildMovementDatasetFromReplays(params: {
  replays: {
    sessionId: string;
    events: { kind: string; tool?: string; summary?: string; metadata?: Record<string, unknown> }[];
  }[];
  order?: number;
  tokenizer?: MovementTokenizer;
}): MovementDataset {
  return buildMovementDataset({
    order: params.order,
    tokenizer: params.tokenizer,
    sources: params.replays.map((replay) => ({
      id: replay.sessionId,
      actions: replay.events
        .filter((event) => event.kind === "action")
        .map((event) => ({
          tool: event.tool ?? "action",
          summary: event.summary ?? "",
          metadata: event.metadata,
        })),
    })),
  });
}

/** A trained, serializable movement policy. Plain JSON — safe to persist. */
export type MovementPolicy = {
  backend: string;
  version: 1;
  order: number;
  vocabulary: MovementToken[];
  /** contextKey -> nextToken -> observed count, for every backoff order. */
  transitions: Record<string, Record<MovementToken, number>>;
  /** Total tokens the policy was trained on (for smoothing / reporting). */
  totalTokens: number;
};

export type MovementPrediction = {
  /** The most likely next token, or `undefined` if the model is empty. */
  token: MovementToken | undefined;
  /** Probability mass of `token` within the chosen (backed-off) context. */
  confidence: number;
  /** How many prior tokens were matched to make the prediction (backoff depth). */
  contextOrderUsed: number;
  /** Full next-token distribution for the chosen context, sorted by probability. */
  distribution: { token: MovementToken; probability: number }[];
};

/**
 * Pluggable movement-model backend. Real on-device backends (a small local LLM,
 * an MLX/axolotl adapter, etc.) implement the same surface so the training and
 * inference pipelines are backend-agnostic; `MarkovMovementBackend` is the
 * built-in deterministic default used for cloud/CI validation.
 */
export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset): MovementPolicy;
  predict(policy: MovementPolicy, context: MovementToken[]): MovementPrediction;
}

export class MarkovMovementBackend implements MovementModelBackend {
  readonly name = "markov-backoff";

  train(dataset: MovementDataset): MovementPolicy {
    const order = normalizeOrder(dataset.order);
    const transitions: Record<string, Record<MovementToken, number>> = {};
    let totalTokens = 0;

    for (const sequence of dataset.sequences) {
      // Left-pad with START so the first real movement is itself predictable.
      const padded = [MOVEMENT_START_TOKEN, ...sequence.tokens];
      for (let i = 1; i < padded.length; i += 1) {
        totalTokens += 1;
        const next = padded[i];
        // Record the transition at every backoff order 0..order.
        for (let k = 0; k <= order; k += 1) {
          if (i - k < 0) {
            break;
          }
          const context = padded.slice(i - k, i);
          const key = contextKey(context);
          const bucket = (transitions[key] ??= {});
          bucket[next] = (bucket[next] ?? 0) + 1;
        }
      }
    }

    return {
      backend: this.name,
      version: 1,
      order,
      vocabulary: [...dataset.vocabulary],
      transitions,
      totalTokens,
    };
  }

  predict(policy: MovementPolicy, context: MovementToken[]): MovementPrediction {
    const order = normalizeOrder(policy.order);
    // Stupid-backoff: try the longest available context suffix first.
    const maxK = Math.min(order, context.length);
    for (let k = maxK; k >= 0; k -= 1) {
      const suffix = k === 0 ? [] : context.slice(context.length - k);
      const withStart = suffix.length === 0 ? [MOVEMENT_START_TOKEN] : suffix;
      const bucket = policy.transitions[contextKey(withStart)] ?? policy.transitions[contextKey(suffix)];
      if (!bucket) {
        continue;
      }
      const distribution = toDistribution(bucket);
      if (distribution.length === 0) {
        continue;
      }
      return {
        token: distribution[0].token,
        confidence: distribution[0].probability,
        contextOrderUsed: k,
        distribution,
      };
    }
    return { token: undefined, confidence: 0, contextOrderUsed: 0, distribution: [] };
  }
}

/**
 * Greedily generate a movement sequence from a trained policy. Reproduces the
 * dominant recorded path from `seed`, and — because prediction backs off —
 * continues plausibly even after entering unseen context (generalization).
 * Deterministic: ties break by token order, never by chance.
 */
export function generateMovements(params: {
  backend: MovementModelBackend;
  policy: MovementPolicy;
  seed?: MovementToken[];
  maxLength: number;
  /** Stop when the predicted confidence drops below this floor. */
  minConfidence?: number;
}): { tokens: MovementToken[]; steps: MovementPrediction[] } {
  const seed = params.seed ?? [];
  const tokens = [...seed];
  const steps: MovementPrediction[] = [];
  const minConfidence = params.minConfidence ?? 0;

  while (tokens.length < params.maxLength) {
    const prediction = params.backend.predict(params.policy, tokens);
    if (prediction.token === undefined || prediction.confidence < minConfidence) {
      break;
    }
    steps.push(prediction);
    tokens.push(prediction.token);
  }

  return { tokens: tokens.slice(seed.length), steps };
}

export type MovementEvalResult = {
  backend: string;
  /** Total next-token predictions evaluated across held-out sequences. */
  total: number;
  /** How many predictions matched the ground-truth next token exactly. */
  correct: number;
  /** correct / total (0 when total is 0). */
  accuracy: number;
  /** Mean confidence the model assigned to its chosen token. */
  averageConfidence: number;
  /** Mean backoff depth used — higher means more exact-context reproduction. */
  averageContextOrder: number;
  perSequence: { id: string; total: number; correct: number; accuracy: number }[];
};

/**
 * Generalization eval harness: measure next-movement prediction fidelity on
 * held-out sequences (ones the policy was NOT trained on). This is how we score
 * whether the model generalizes to "new but related" movements rather than only
 * memorizing its training set.
 */
export function evaluateNextMovement(params: {
  backend: MovementModelBackend;
  policy: MovementPolicy;
  sequences: MovementSequence[];
}): MovementEvalResult {
  let total = 0;
  let correct = 0;
  let confidenceSum = 0;
  let contextOrderSum = 0;
  const perSequence: MovementEvalResult["perSequence"] = [];

  for (const sequence of params.sequences) {
    let seqTotal = 0;
    let seqCorrect = 0;
    for (let i = 0; i < sequence.tokens.length; i += 1) {
      const context = sequence.tokens.slice(0, i);
      const prediction = params.backend.predict(params.policy, context);
      total += 1;
      seqTotal += 1;
      confidenceSum += prediction.confidence;
      contextOrderSum += prediction.contextOrderUsed;
      if (prediction.token === sequence.tokens[i]) {
        correct += 1;
        seqCorrect += 1;
      }
    }
    perSequence.push({
      id: sequence.id,
      total: seqTotal,
      correct: seqCorrect,
      accuracy: seqTotal === 0 ? 0 : seqCorrect / seqTotal,
    });
  }

  return {
    backend: params.backend.name,
    total,
    correct,
    accuracy: total === 0 ? 0 : correct / total,
    averageConfidence: total === 0 ? 0 : confidenceSum / total,
    averageContextOrder: total === 0 ? 0 : contextOrderSum / total,
    perSequence,
  };
}

/**
 * Deterministic train/held-out split by sequence index (every `holdoutEvery`-th
 * sequence is held out). Deterministic so eval numbers are reproducible across
 * runs and machines.
 */
export function splitMovementDataset(
  dataset: MovementDataset,
  holdoutEvery = 4,
): { train: MovementDataset; holdout: MovementSequence[] } {
  const step = holdoutEvery < 2 ? 2 : Math.floor(holdoutEvery);
  const trainSequences: MovementSequence[] = [];
  const holdout: MovementSequence[] = [];
  dataset.sequences.forEach((sequence, index) => {
    if ((index + 1) % step === 0) {
      holdout.push(sequence);
    } else {
      trainSequences.push(sequence);
    }
  });
  return {
    train: { ...dataset, sequences: trainSequences },
    holdout,
  };
}

function toDistribution(bucket: Record<MovementToken, number>): { token: MovementToken; probability: number }[] {
  const entries = Object.entries(bucket);
  const total = entries.reduce((sum, [, count]) => sum + count, 0);
  if (total === 0) {
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

function contextKey(context: MovementToken[]): string {
  return context.length === 0 ? " " : context.join(" ");
}

function normalizeOrder(order: number | undefined): number {
  if (order === undefined || !Number.isFinite(order) || order < 1) {
    return DEFAULT_ORDER;
  }
  return Math.floor(order);
}

function pickString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim().length > 0 ? value : undefined;
}

function slug(value: string): string {
  const cleaned = value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return cleaned.length === 0 ? "unknown" : cleaned;
}
