import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";

/**
 * Local-movement learning: inference layer.
 *
 * The rest of the training subsystem (`exporter`, `runner`, `job-store`) prepares
 * reviewed datasets and launches an external on-device trainer (mlx/axolotl). That
 * pipeline cannot run in the cloud, so the *learn -> repeat -> generalize* loop that
 * standing objective #2 (c/d) describes was never exercised or tested here.
 *
 * This module closes that gap with a fully in-process, deterministic implementation:
 *   - a `MovementModelBackend` seam so a real on-device small model can be dropped in;
 *   - a `DeterministicNearestNeighborBackend` mock that learns from replay events and
 *     can (c) repeat recorded movements exactly and (d) generalize to related-but-new
 *     ones via nearest-neighbour retrieval over featurized context;
 *   - dataset construction from the existing `ReplayManifest` format, so no new schema
 *     is invented; and an eval helper for held-out generalization measurement.
 *
 * It is intentionally free of I/O, randomness, and wall-clock reads, so it is
 * reproducible in CI and safe to unit-test with synthetic event streams.
 */

/** A single observation the agent can condition on (from an observation or transcript event). */
export type MovementObservation = {
  source: string;
  summary: string;
};

/** A prior action, used as short-horizon context for the next prediction. */
export type MovementActionContext = {
  tool: string;
  summary: string;
};

/** The context available to the model at the moment it must choose the next action. */
export type MovementContext = {
  observations: MovementObservation[];
  recentActions: MovementActionContext[];
};

/** A supervised example: given `context`, the recorded operator performed `target`. */
export type MovementExample = {
  context: MovementContext;
  target: MovementActionContext;
};

export type MovementPredictionSource = "recall" | "generalize";

export type MovementPrediction = {
  tool: string;
  summary: string;
  /** 1 for an exact context match (recall); the neighbour similarity otherwise (generalize). */
  confidence: number;
  source: MovementPredictionSource;
};

export type TrainedModelMetadata = {
  backend: string;
  exampleCount: number;
  /** Distinct target tools the model has seen — its action vocabulary. */
  vocabulary: string[];
};

/** A model produced by a backend. Pure/synchronous inference keeps replay deterministic. */
export interface TrainedMovementModel {
  readonly backend: string;
  readonly exampleCount: number;
  predict(context: MovementContext): MovementPrediction | undefined;
  metadata(): TrainedModelMetadata;
}

export type MovementTrainOptions = {
  /**
   * Sliding-window size for context assembled from a replay timeline: how many prior
   * observations/actions precede each target action. Larger windows capture more
   * setup but dilute similarity. Defaults to 4.
   */
  contextWindow?: number;
};

/**
 * The pluggable seam. A real on-device model (e.g. an mlx LoRA adapter) implements the
 * same interface as the deterministic mock, so callers never depend on the backend.
 */
export interface MovementModelBackend {
  readonly name: string;
  train(examples: MovementExample[]): TrainedMovementModel;
}

/**
 * Turn a reviewed `ReplayManifest` into supervised examples. Walks the timeline in order,
 * keeping a bounded window of the most recent observations (observation + transcript
 * events) and actions; every `action` event becomes a training target whose context is
 * the window as it stood *before* that action.
 */
export function buildMovementExamplesFromReplay(
  manifest: ReplayManifest,
  options: MovementTrainOptions = {},
): MovementExample[] {
  const windowSize = normalizeWindow(options.contextWindow);
  const events = [...manifest.events].sort((a, b) => a.ts - b.ts);
  const observations: MovementObservation[] = [];
  const actions: MovementActionContext[] = [];
  const examples: MovementExample[] = [];

  for (const event of events) {
    const observation = toObservation(event);
    if (observation) {
      observations.push(observation);
      continue;
    }
    if (event.kind === "action") {
      examples.push({
        context: {
          observations: observations.slice(-windowSize),
          recentActions: actions.slice(-windowSize),
        },
        target: { tool: event.tool, summary: event.summary },
      });
      actions.push({ tool: event.tool, summary: event.summary });
    }
  }

  return examples;
}

/** Build examples from several replays (a full reviewed export) in one pass. */
export function buildMovementExamplesFromReplays(
  manifests: ReplayManifest[],
  options: MovementTrainOptions = {},
): MovementExample[] {
  return manifests.flatMap((manifest) => buildMovementExamplesFromReplay(manifest, options));
}

type PreparedExample = {
  example: MovementExample;
  tokens: Set<string>;
};

/**
 * Deterministic, dependency-free reference backend. It featurizes each example's context
 * into a token set and, at inference time, returns the target of the most similar
 * training context (Jaccard similarity). Identical context => exact recall (confidence 1);
 * a related context => generalization with confidence = neighbour similarity.
 */
export class DeterministicNearestNeighborBackend implements MovementModelBackend {
  readonly name = "deterministic-nearest-neighbor";

  train(examples: MovementExample[]): TrainedMovementModel {
    const prepared: PreparedExample[] = examples.map((example) => ({
      example,
      tokens: featurize(example.context),
    }));
    return new NearestNeighborModel(this.name, prepared);
  }
}

class NearestNeighborModel implements TrainedMovementModel {
  constructor(
    readonly backend: string,
    private readonly prepared: PreparedExample[],
  ) {}

  get exampleCount(): number {
    return this.prepared.length;
  }

  predict(context: MovementContext): MovementPrediction | undefined {
    const queryTokens = featurize(context);
    let best: { similarity: number; example: MovementExample } | undefined;

    for (const candidate of this.prepared) {
      const similarity = jaccard(queryTokens, candidate.tokens);
      // Strictly-greater keeps the first-seen example on ties -> deterministic.
      if (!best || similarity > best.similarity) {
        best = { similarity, example: candidate.example };
      }
    }

    if (!best || best.similarity <= 0) {
      return undefined;
    }

    const source: MovementPredictionSource = best.similarity >= 1 ? "recall" : "generalize";
    return {
      tool: best.example.target.tool,
      summary: best.example.target.summary,
      confidence: best.similarity,
      source,
    };
  }

  metadata(): TrainedModelMetadata {
    const vocabulary = [...new Set(this.prepared.map((entry) => entry.example.target.tool))].sort();
    return {
      backend: this.backend,
      exampleCount: this.exampleCount,
      vocabulary,
    };
  }
}

export type MovementEvalResult = {
  total: number;
  /** Predictions whose tool matched the held-out target. */
  correct: number;
  /** correct / total, or 0 when there is nothing to evaluate. */
  accuracy: number;
  /** How many predictions came from exact recall vs. generalization. */
  recalled: number;
  generalized: number;
  /** Cases where the model produced no prediction at all. */
  abstained: number;
};

/**
 * Generalization eval harness: measure how often a trained model reproduces the correct
 * tool on held-out examples. Feeding it examples it was trained on tests *repeat*;
 * feeding related-but-unseen examples tests *generalize*.
 */
export function evaluateMovementModel(
  model: TrainedMovementModel,
  examples: MovementExample[],
): MovementEvalResult {
  let correct = 0;
  let recalled = 0;
  let generalized = 0;
  let abstained = 0;

  for (const example of examples) {
    const prediction = model.predict(example.context);
    if (!prediction) {
      abstained += 1;
      continue;
    }
    if (prediction.source === "recall") {
      recalled += 1;
    } else {
      generalized += 1;
    }
    if (prediction.tool === example.target.tool) {
      correct += 1;
    }
  }

  const total = examples.length;
  return {
    total,
    correct,
    accuracy: total === 0 ? 0 : correct / total,
    recalled,
    generalized,
    abstained,
  };
}

function toObservation(event: ReplayTimelineEvent): MovementObservation | undefined {
  if (event.kind === "observation") {
    return { source: event.source, summary: event.summary };
  }
  if (event.kind === "transcript") {
    return { source: `transcript:${event.role}`, summary: event.content };
  }
  return undefined;
}

function normalizeWindow(value: number | undefined): number {
  if (value === undefined || !Number.isFinite(value)) {
    return 4;
  }
  return Math.max(1, Math.floor(value));
}

/** Featurize a context into a set of source/tool-prefixed tokens for similarity. */
export function featurize(context: MovementContext): Set<string> {
  const tokens = new Set<string>();
  for (const observation of context.observations) {
    for (const token of tokenize(observation.summary)) {
      tokens.add(`obs:${observation.source}:${token}`);
    }
  }
  for (const action of context.recentActions) {
    tokens.add(`act:tool:${action.tool.toLowerCase()}`);
    for (const token of tokenize(action.summary)) {
      tokens.add(`act:${token}`);
    }
  }
  return tokens;
}

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((token) => token.length > 0);
}

function jaccard(a: Set<string>, b: Set<string>): number {
  if (a.size === 0 && b.size === 0) {
    return 0;
  }
  let intersection = 0;
  for (const token of a) {
    if (b.has(token)) {
      intersection += 1;
    }
  }
  const union = a.size + b.size - intersection;
  return union === 0 ? 0 : intersection / union;
}
