import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Local-movement learning: pluggable in-process model backend.
 *
 * The reviewed-export → training-plan path (see `runner.ts`) emits *launch
 * scripts* for real on-device runtimes (MLX / axolotl on Apple Silicon). Those
 * cannot execute in the cloud/CI. This module provides the complementary piece
 * the objective calls for: an in-process, deterministic backend that can
 * actually **post-train a model on recorded movements and run inference** —
 * (c) repeat the recorded movements and (d) generalize to new-but-related ones.
 *
 * The backend is pluggable via {@link LocalMovementModelBackend}. The bundled
 * {@link NgramMovementBackend} is a deterministic sequence model (suffix
 * back-off + token-overlap similarity) so tests pass without real OS input or a
 * heavyweight ML runtime, while a real small local model can be dropped in
 * behind the same interface.
 */

/** A single movement the model can emit — a tool invocation with a summary. */
export type MovementAction = {
  tool: string;
  summary: string;
};

/**
 * One supervised training example: the ordered context tokens leading up to an
 * action, and the action that followed. `weight` biases frequency counts (e.g.
 * from trajectory reward) and defaults to 1.
 */
export type MovementExample = {
  context: string[];
  action: MovementAction;
  weight?: number;
};

export type MovementDataset = {
  version: 1;
  examples: MovementExample[];
};

export type MovementTrainingConfig = {
  /** Max number of prior events that form an action's context window. */
  contextWindow: number;
  /**
   * Minimum token-overlap (Jaccard) required for the similarity back-off to
   * fire when no exact suffix matches. 0..1; defaults to 0.34.
   */
  similarityThreshold?: number;
};

export type PredictionSource = "exact" | "backoff" | "similar" | "prior" | "none";

export type ActionPrediction = {
  action: MovementAction | undefined;
  /** 0..1 estimate of how strongly the model backs this action. */
  confidence: number;
  /** Length of the context suffix that matched (0 for similarity/prior). */
  matchedContextLength: number;
  source: PredictionSource;
};

export type TrainedMovementModel = {
  readonly backendId: string;
  readonly config: Required<MovementTrainingConfig>;
  /** Predict the next movement given the context tokens observed so far. */
  predict(context: string[]): ActionPrediction;
  /** Serializable summary of what was learned (for inspection / persistence). */
  describe(): MovementModelSummary;
};

export type MovementModelSummary = {
  backendId: string;
  config: Required<MovementTrainingConfig>;
  exampleCount: number;
  distinctActions: number;
  distinctContexts: number;
};

export interface LocalMovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, config: MovementTrainingConfig): Promise<TrainedMovementModel>;
}

const DEFAULT_SIMILARITY_THRESHOLD = 0.34;

/** Stable string key for an action (used for tie-breaking and counting). */
export function movementActionKey(action: MovementAction): string {
  return `${action.tool} ${action.summary}`;
}

function normalizeConfig(config: MovementTrainingConfig): Required<MovementTrainingConfig> {
  return {
    contextWindow: Math.max(1, Math.floor(config.contextWindow)),
    similarityThreshold:
      config.similarityThreshold === undefined
        ? DEFAULT_SIMILARITY_THRESHOLD
        : Math.min(1, Math.max(0, config.similarityThreshold)),
  };
}

/**
 * Flatten trajectory spans into an ordered event token stream and derive one
 * training example per recorded action. Observations become `obs:<source>`
 * tokens and prior actions become `act:<tool>` tokens, so the model conditions
 * on both what was seen and what was done leading up to each movement.
 */
export function buildMovementDataset(
  trajectories: TrajectorySpan[],
  options: { contextWindow?: number } = {},
): MovementDataset {
  const contextWindow = Math.max(1, Math.floor(options.contextWindow ?? 6));
  const examples: MovementExample[] = [];

  for (const trajectory of trajectories) {
    const events = [
      ...trajectory.observations.map((observation) => ({
        ts: observation.ts,
        token: `obs:${observation.source}`,
        action: undefined as MovementAction | undefined,
      })),
      ...trajectory.actions.map((action) => ({
        ts: action.ts,
        token: `act:${action.tool}`,
        action: { tool: action.tool, summary: action.summary } as MovementAction,
      })),
    ].sort((a, b) => a.ts - b.ts);

    const history: string[] = [];
    const rewardWeight = normalizeReward(trajectory.outcome?.reward);
    for (const event of events) {
      if (event.action) {
        examples.push({
          context: history.slice(-contextWindow),
          action: event.action,
          weight: rewardWeight,
        });
      }
      history.push(event.token);
    }
  }

  return { version: 1, examples };
}

function normalizeReward(reward: number | undefined): number {
  if (reward === undefined || !Number.isFinite(reward)) {
    return 1;
  }
  // Map reward into a positive weight; never zero so every example still counts.
  return Math.max(0.1, 1 + reward);
}

type ActionCount = { action: MovementAction; weight: number };

class NgramMovementModel implements TrainedMovementModel {
  readonly backendId: string;
  readonly config: Required<MovementTrainingConfig>;

  /** suffix-key (joined tokens) → action-key → weighted count. */
  private readonly ngrams: Map<string, Map<string, ActionCount>>;
  /** global prior over actions. */
  private readonly prior: Map<string, ActionCount>;
  /** token → action-key → weighted count, for similarity back-off. */
  private readonly tokenActions: Map<string, Map<string, ActionCount>>;
  private readonly exampleCount: number;
  private readonly distinctActions: number;

  constructor(backendId: string, dataset: MovementDataset, config: Required<MovementTrainingConfig>) {
    this.backendId = backendId;
    this.config = config;
    this.ngrams = new Map();
    this.prior = new Map();
    this.tokenActions = new Map();
    const actionKeys = new Set<string>();

    for (const example of dataset.examples) {
      const weight = example.weight === undefined ? 1 : example.weight;
      const context = example.context;
      actionKeys.add(movementActionKey(example.action));
      bump(this.prior, example.action, weight);

      // Record every suffix length up to the context window (n-gram back-off).
      const maxLen = Math.min(context.length, config.contextWindow);
      for (let len = 1; len <= maxLen; len += 1) {
        const suffix = context.slice(context.length - len);
        const key = suffix.join("");
        bump(mapFor(this.ngrams, key), example.action, weight);
      }

      // Token co-occurrence for the generalization path.
      for (const token of new Set(context)) {
        bump(mapFor(this.tokenActions, token), example.action, weight);
      }
    }

    this.exampleCount = dataset.examples.length;
    this.distinctActions = actionKeys.size;
  }

  predict(context: string[]): ActionPrediction {
    if (this.exampleCount === 0) {
      return { action: undefined, confidence: 0, matchedContextLength: 0, source: "none" };
    }

    // 1. Longest exact suffix match (repeat recorded movements).
    const maxLen = Math.min(context.length, this.config.contextWindow);
    for (let len = maxLen; len >= 1; len -= 1) {
      const suffix = context.slice(context.length - len);
      const bucket = this.ngrams.get(suffix.join(""));
      if (bucket) {
        const best = argMax(bucket);
        return {
          action: best.action,
          confidence: best.weight / totalWeight(bucket),
          matchedContextLength: len,
          source: len === maxLen && len === context.length ? "exact" : "backoff",
        };
      }
    }

    // 2. Token-overlap similarity (generalize to new-but-related movements).
    const similar = this.predictBySimilarity(context);
    if (similar) {
      return similar;
    }

    // 3. Global prior fallback.
    const best = argMax(this.prior);
    return {
      action: best.action,
      confidence: best.weight / totalWeight(this.prior),
      matchedContextLength: 0,
      source: "prior",
    };
  }

  private predictBySimilarity(context: string[]): ActionPrediction | undefined {
    const queryTokens = new Set(context);
    if (queryTokens.size === 0) {
      return undefined;
    }

    const scores = new Map<string, ActionCount>();
    for (const token of queryTokens) {
      const bucket = this.tokenActions.get(token);
      if (!bucket) {
        continue;
      }
      for (const [, entry] of bucket) {
        bump(scores, entry.action, entry.weight);
      }
    }
    if (scores.size === 0) {
      return undefined;
    }

    const best = argMax(scores);
    // Approximate overlap: fraction of query tokens that co-occur with the
    // winning action, giving a bounded 0..1 confidence.
    const winnerKey = movementActionKey(best.action);
    let overlappingTokens = 0;
    for (const token of queryTokens) {
      if (this.tokenActions.get(token)?.has(winnerKey)) {
        overlappingTokens += 1;
      }
    }
    const similarity = overlappingTokens / queryTokens.size;
    if (similarity < this.config.similarityThreshold) {
      return undefined;
    }
    return {
      action: best.action,
      confidence: similarity,
      matchedContextLength: 0,
      source: "similar",
    };
  }

  describe(): MovementModelSummary {
    return {
      backendId: this.backendId,
      config: this.config,
      exampleCount: this.exampleCount,
      distinctActions: this.distinctActions,
      distinctContexts: this.ngrams.size,
    };
  }
}

/**
 * Deterministic, dependency-free movement backend. Learns a weighted n-gram
 * distribution over actions plus a token co-occurrence index; inference tries
 * an exact suffix match, then token-overlap similarity, then the global prior.
 * No randomness — identical datasets yield identical models and predictions.
 */
export class NgramMovementBackend implements LocalMovementModelBackend {
  readonly id = "ngram-mock-v1";

  async train(dataset: MovementDataset, config: MovementTrainingConfig): Promise<TrainedMovementModel> {
    return new NgramMovementModel(this.id, dataset, normalizeConfig(config));
  }
}

export type MovementEvaluation = {
  total: number;
  correct: number;
  /** Fraction of held-out actions predicted exactly (tool + summary). */
  accuracy: number;
  /** Fraction where at least the tool matched (movement class). */
  toolAccuracy: number;
  bySource: Record<PredictionSource, number>;
};

/**
 * Replay-fidelity / generalization eval: for each example, predict from its
 * context and compare against the recorded action. Use a held-out dataset to
 * measure generalization to new-but-related movements.
 */
export function evaluateMovementModel(
  model: TrainedMovementModel,
  dataset: MovementDataset,
): MovementEvaluation {
  const bySource: Record<PredictionSource, number> = {
    exact: 0,
    backoff: 0,
    similar: 0,
    prior: 0,
    none: 0,
  };
  let correct = 0;
  let toolCorrect = 0;

  for (const example of dataset.examples) {
    const prediction = model.predict(example.context);
    bySource[prediction.source] += 1;
    if (prediction.action) {
      if (movementActionKey(prediction.action) === movementActionKey(example.action)) {
        correct += 1;
      }
      if (prediction.action.tool === example.action.tool) {
        toolCorrect += 1;
      }
    }
  }

  const total = dataset.examples.length;
  return {
    total,
    correct,
    accuracy: total === 0 ? 0 : correct / total,
    toolAccuracy: total === 0 ? 0 : toolCorrect / total,
    bySource,
  };
}

function mapFor<K>(outer: Map<K, Map<string, ActionCount>>, key: K): Map<string, ActionCount> {
  let inner = outer.get(key);
  if (!inner) {
    inner = new Map();
    outer.set(key, inner);
  }
  return inner;
}

function bump(bucket: Map<string, ActionCount>, action: MovementAction, weight: number): void {
  const key = movementActionKey(action);
  const existing = bucket.get(key);
  if (existing) {
    existing.weight += weight;
  } else {
    bucket.set(key, { action, weight });
  }
}

function totalWeight(bucket: Map<string, ActionCount>): number {
  let sum = 0;
  for (const entry of bucket.values()) {
    sum += entry.weight;
  }
  return sum;
}

/** Highest-weight action; ties broken by action key for determinism. */
function argMax(bucket: Map<string, ActionCount>): ActionCount {
  let best: ActionCount | undefined;
  let bestKey = "";
  for (const [key, entry] of bucket) {
    if (
      best === undefined ||
      entry.weight > best.weight ||
      (entry.weight === best.weight && key < bestKey)
    ) {
      best = entry;
      bestKey = key;
    }
  }
  // bucket is guaranteed non-empty by callers.
  return best as ActionCount;
}
