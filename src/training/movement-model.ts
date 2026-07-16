import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

/**
 * Movement model: the inference half of the local-movement learning subsystem.
 *
 * The capture pipeline records trajectories of on-device movements/actions and
 * the training runner emits on-device (mlx/axolotl) launch plans. This module
 * closes the loop with a *pluggable model backend* that actually learns a policy
 * from a movement dataset and predicts the next movement — including for
 * new-but-related contexts it never saw verbatim (generalization).
 *
 * The default backend is a deterministic order-N Markov policy with stupid
 * backoff, so it trains and predicts with zero external dependencies and runs in
 * the cloud/CI. Real on-device small models (a distilled transformer, an RNN,
 * etc.) can implement {@link MovementModelBackend} behind the same seam.
 */

/** Canonical, comparable identifier for a single movement/action. */
export type MovementToken = string;

/** Sentinel that marks the start of a movement sequence (for prefix backoff). */
export const MOVEMENT_START_TOKEN = "<start>" as const;

/** One training example: the context tokens leading up to `next`. */
export type MovementSample = {
  /** Ordered context tokens (most recent last), never longer than the dataset order. */
  context: MovementToken[];
  /** The movement performed after `context`. */
  next: MovementToken;
};

export type MovementDataset = {
  version: 1;
  /** Maximum context length the samples were generated with. */
  order: number;
  /** Sorted, de-duplicated set of every token that appears. */
  vocabulary: MovementToken[];
  samples: MovementSample[];
};

export type MovementPrediction = {
  /** The predicted next token, or `undefined` if the model has no signal at all. */
  token?: MovementToken;
  /** Ranked candidates (highest score first), deterministic tie-break by token. */
  candidates: Array<{ token: MovementToken; score: number }>;
  /**
   * Context length actually used to produce the prediction after backoff.
   * Equal to the requested context length on an exact hit; smaller when the
   * model had to generalize by backing off to a shorter context; 0 for the
   * unigram (marginal) fallback.
   */
  backoffOrder: number;
};

export type MovementTrainingConfig = {
  /** Cap on context length used for prediction; defaults to the dataset order. */
  order?: number;
};

/** A serialized model — plain JSON, safe to persist alongside training artifacts. */
export type MovementModelSnapshot = {
  version: 1;
  backend: string;
  order: number;
  vocabulary: MovementToken[];
  /** context-key -> (next-token -> count). Empty key is the unigram table. */
  transitions: Record<string, Record<MovementToken, number>>;
};

export interface MovementModel {
  readonly backend: string;
  readonly order: number;
  readonly vocabulary: readonly MovementToken[];
  /** Predict the next movement given a context (any length; trimmed internally). */
  predict(context: MovementToken[]): MovementPrediction;
  /** Autoregressively roll out `steps` movements starting from `seed`. */
  rollout(seed: MovementToken[], steps: number): MovementToken[];
  serialize(): MovementModelSnapshot;
}

export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, config?: MovementTrainingConfig): MovementModel;
  restore(snapshot: MovementModelSnapshot): MovementModel;
}

const TOKEN_SEPARATOR = "";

function slug(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 48);
}

function metadataString(metadata: Record<string, unknown> | undefined, key: string): string | undefined {
  const value = metadata?.[key];
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

/**
 * Canonicalize a recorded action into a stable movement token. Prefers the
 * structured gesture metadata the device/browser/os adapters attach; falls back
 * to a slug of the human summary so no action is un-tokenizable.
 */
export function movementTokenFromAction(action: TrajectoryAction): MovementToken {
  const metadata = action.metadata;
  const parts: string[] = [action.tool];
  const gesture = metadataString(metadata, "gesture");
  const direction = metadataString(metadata, "direction");
  const target = metadataString(metadata, "target");
  if (gesture) {
    parts.push(gesture);
  }
  if (direction) {
    parts.push(direction);
  }
  if (target) {
    parts.push(slug(target));
  }
  if (parts.length === 1) {
    const summarySlug = slug(action.summary);
    parts.push(summarySlug.length > 0 ? summarySlug : "action");
  }
  return parts.join(":");
}

/** Extract the ordered movement token sequence from one trajectory span. */
export function movementSequenceFromTrajectory(trajectory: TrajectorySpan): MovementToken[] {
  return [...trajectory.actions]
    .sort((a, b) => a.ts - b.ts)
    .map((action) => movementTokenFromAction(action));
}

/**
 * Build a movement dataset from recorded trajectories. Each trajectory yields
 * one sample per action, whose context is the (up to `order`) preceding tokens,
 * padded with {@link MOVEMENT_START_TOKEN} so the model learns how sequences
 * begin. Trajectories with no actions are skipped.
 */
export function buildMovementDataset(
  trajectories: TrajectorySpan[],
  options: { order?: number } = {},
): MovementDataset {
  const order = Math.max(1, Math.floor(options.order ?? 2));
  const samples: MovementSample[] = [];
  const vocabulary = new Set<MovementToken>();

  for (const trajectory of trajectories) {
    const sequence = movementSequenceFromTrajectory(trajectory);
    if (sequence.length === 0) {
      continue;
    }
    const padded = [...Array<MovementToken>(order).fill(MOVEMENT_START_TOKEN), ...sequence];
    for (let index = order; index < padded.length; index += 1) {
      const next = padded[index]!;
      const context = padded.slice(index - order, index);
      samples.push({ context, next });
      vocabulary.add(next);
    }
  }

  return {
    version: 1,
    order,
    vocabulary: [...vocabulary].sort(),
    samples,
  };
}

function contextKey(context: MovementToken[]): string {
  return context.join(TOKEN_SEPARATOR);
}

function trimContext(context: MovementToken[], order: number): MovementToken[] {
  return order <= 0 ? [] : context.slice(-order);
}

class MarkovMovementModel implements MovementModel {
  readonly backend: string;
  readonly order: number;
  readonly vocabulary: readonly MovementToken[];
  private readonly transitions: Map<string, Map<MovementToken, number>>;

  constructor(params: {
    backend: string;
    order: number;
    vocabulary: MovementToken[];
    transitions: Map<string, Map<MovementToken, number>>;
  }) {
    this.backend = params.backend;
    this.order = params.order;
    this.vocabulary = params.vocabulary;
    this.transitions = params.transitions;
  }

  predict(context: MovementToken[]): MovementPrediction {
    // Stupid backoff: try the longest context first, shorten until a table hits.
    for (let used = Math.min(this.order, context.length); used >= 0; used -= 1) {
      const key = contextKey(trimContext(context, used));
      const counts = this.transitions.get(key);
      if (counts && counts.size > 0) {
        return { ...rankCounts(counts), backoffOrder: used };
      }
    }
    return { candidates: [], backoffOrder: 0 };
  }

  rollout(seed: MovementToken[], steps: number): MovementToken[] {
    const produced: MovementToken[] = [];
    let context = [...seed];
    for (let step = 0; step < steps; step += 1) {
      const prediction = this.predict(context);
      if (!prediction.token) {
        break;
      }
      produced.push(prediction.token);
      context = trimContext([...context, prediction.token], this.order);
    }
    return produced;
  }

  serialize(): MovementModelSnapshot {
    const transitions: Record<string, Record<MovementToken, number>> = {};
    for (const [key, counts] of this.transitions) {
      transitions[key] = Object.fromEntries([...counts.entries()].sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)));
    }
    return {
      version: 1,
      backend: this.backend,
      order: this.order,
      vocabulary: [...this.vocabulary],
      transitions,
    };
  }
}

function rankCounts(counts: Map<MovementToken, number>): {
  token: MovementToken | undefined;
  candidates: Array<{ token: MovementToken; score: number }>;
} {
  const total = [...counts.values()].reduce((sum, value) => sum + value, 0);
  const candidates = [...counts.entries()]
    .map(([token, count]) => ({ token, score: total > 0 ? count / total : 0 }))
    // Deterministic order: score desc, then token asc for stable tie-breaks.
    .sort((a, b) => (b.score - a.score) || (a.token < b.token ? -1 : a.token > b.token ? 1 : 0));
  return { token: candidates[0]?.token, candidates };
}

/**
 * Deterministic, dependency-free movement backend: an order-N Markov policy
 * with stupid backoff. Trains by counting context->next transitions at every
 * order from N down to 0 (the unigram marginal), so predicting an unseen
 * context degrades gracefully to shorter contexts instead of failing — this is
 * what lets it generalize to new-but-related movements.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly name = "markov";

  train(dataset: MovementDataset, config: MovementTrainingConfig = {}): MovementModel {
    const order = Math.max(0, Math.min(Math.floor(config.order ?? dataset.order), dataset.order));
    const transitions = new Map<string, Map<MovementToken, number>>();

    const bump = (context: MovementToken[], next: MovementToken): void => {
      const key = contextKey(context);
      let counts = transitions.get(key);
      if (!counts) {
        counts = new Map<MovementToken, number>();
        transitions.set(key, counts);
      }
      counts.set(next, (counts.get(next) ?? 0) + 1);
    };

    for (const sample of dataset.samples) {
      const trimmed = trimContext(sample.context, order);
      // Record the transition at every backoff length so prediction can degrade.
      for (let used = trimmed.length; used >= 0; used -= 1) {
        bump(trimContext(trimmed, used), sample.next);
      }
    }

    return new MarkovMovementModel({
      backend: this.name,
      order,
      vocabulary: [...dataset.vocabulary].sort(),
      transitions,
    });
  }

  restore(snapshot: MovementModelSnapshot): MovementModel {
    const transitions = new Map<string, Map<MovementToken, number>>();
    for (const [key, counts] of Object.entries(snapshot.transitions)) {
      transitions.set(key, new Map(Object.entries(counts)));
    }
    return new MarkovMovementModel({
      backend: snapshot.backend,
      order: snapshot.order,
      vocabulary: [...snapshot.vocabulary],
      transitions,
    });
  }
}

export type MovementEvaluation = {
  total: number;
  correct: number;
  /** Top-1 accuracy in [0, 1]; 0 when there are no samples. */
  accuracy: number;
  /** Fraction of predictions where the true next token was in the top-k candidates. */
  topKAccuracy: number;
  /** Fraction of predictions that required backoff (context not seen verbatim). */
  generalizationRate: number;
};

/**
 * Generalization eval harness: measure how well a trained model reproduces the
 * next movement on held-out samples. `topK` credits a prediction when the true
 * token is among the model's top-k candidates; `generalizationRate` reports how
 * often the model had to back off to a shorter context (i.e. genuinely
 * generalize rather than recall a verbatim context).
 */
export function evaluateMovementModel(
  model: MovementModel,
  samples: MovementSample[],
  options: { topK?: number } = {},
): MovementEvaluation {
  const topK = Math.max(1, Math.floor(options.topK ?? 3));
  let correct = 0;
  let topKCorrect = 0;
  let backedOff = 0;

  for (const sample of samples) {
    const prediction = model.predict(sample.context);
    if (prediction.token === sample.next) {
      correct += 1;
    }
    if (prediction.candidates.slice(0, topK).some((candidate) => candidate.token === sample.next)) {
      topKCorrect += 1;
    }
    if (prediction.backoffOrder < Math.min(model.order, sample.context.length)) {
      backedOff += 1;
    }
  }

  const total = samples.length;
  return {
    total,
    correct,
    accuracy: total > 0 ? correct / total : 0,
    topKAccuracy: total > 0 ? topKCorrect / total : 0,
    generalizationRate: total > 0 ? backedOff / total : 0,
  };
}

/** Convenience default backend instance. */
export const defaultMovementBackend: MovementModelBackend = new MarkovMovementBackend();
