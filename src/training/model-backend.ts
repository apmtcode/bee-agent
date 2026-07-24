import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Pluggable local-model backend for the movement-learning subsystem.
 *
 * The training runner (`runner.ts`) emits *external* launch commands (mlx /
 * axolotl) that execute on the user's machine. This module provides the
 * in-repo, fully deterministic counterpart: an interface a local model can
 * implement to (a) train on recorded movements and (b) predict the next
 * movement — including for *new but related* movement contexts (objective 2c/2d).
 *
 * Everything here is pure and side-effect free so it can be validated in the
 * cloud against synthetic event streams. A real on-device small model can be
 * dropped in later by implementing {@link MovementModelBackend}; the
 * {@link DeterministicMarkovMovementBackend} is the reference/mock backend.
 */

/** A single recorded (or predicted) movement — a tool invocation with a summary. */
export type MovementActionToken = {
  tool: string;
  summary: string;
};

/** One supervised example: prior movements → the movement that followed. */
export type MovementExample = {
  /** Ordered context of prior movements (oldest → newest). */
  context: MovementActionToken[];
  /** The movement that should be produced next. */
  next: MovementActionToken;
  /** Optional example weight (e.g. derived from trajectory reward). Default 1. */
  weight?: number;
};

/** A replayable, backend-agnostic movement dataset. */
export type MovementDataset = {
  version: 1;
  /** Largest context length any example carries; the model's max order. */
  contextWindow: number;
  examples: MovementExample[];
};

export type MovementTrainOptions = {
  /**
   * Max Markov order / context length the model conditions on. Defaults to the
   * dataset's `contextWindow`. Clamped to `[1, dataset.contextWindow]`.
   */
  contextWindow?: number;
  /** Stamped into the artifact for provenance; injected for determinism. */
  trainedAt?: string;
};

export type MovementPredictionSource = "exact" | "generalized" | "prior" | "none";

export type MovementPredictionCandidate = {
  action: MovementActionToken;
  score: number;
};

export type MovementPrediction = {
  /** Predicted next movement, or undefined when the model has no signal. */
  action: MovementActionToken | undefined;
  /** Confidence in `[0, 1]`. */
  confidence: number;
  /**
   * How the prediction was produced:
   * - `exact`: an observed continuation of the recorded movements (replay).
   * - `generalized`: same tool context, unseen details (new-but-related).
   * - `prior`: global most-common movement fallback.
   * - `none`: no prediction possible.
   */
  source: MovementPredictionSource;
  /** Ranked alternatives at the winning back-off level. */
  candidates: MovementPredictionCandidate[];
};

export type MovementPredictOptions = {
  /** Cap on returned candidates. Default 5. */
  maxCandidates?: number;
};

/** A serializable trained-model artifact. Opaque `parameters` per backend. */
export type MovementModelArtifact = {
  backendId: string;
  version: 1;
  contextWindow: number;
  trainedAt?: string;
  metadata: {
    exampleCount: number;
    vocabularySize: number;
  };
  parameters: unknown;
};

/** The pluggable backend seam. Implement this for a real on-device model. */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainOptions): MovementModelArtifact;
  predict(
    model: MovementModelArtifact,
    context: MovementActionToken[],
    options?: MovementPredictOptions,
  ): MovementPrediction;
}

const TOKEN_FIELD_SEP = "";
const SUFFIX_SEP = "";

export function movementTokenKey(token: MovementActionToken): string {
  return `${token.tool}${TOKEN_FIELD_SEP}${token.summary}`;
}

/**
 * Build a movement dataset from recorded trajectory spans by sliding a window
 * over each trajectory's time-ordered actions. Reviewed/redacted actions take
 * precedence over raw actions when present.
 */
export function buildMovementDataset(
  trajectories: TrajectorySpan[],
  options: { contextWindow?: number } = {},
): MovementDataset {
  const contextWindow = Math.max(1, Math.floor(options.contextWindow ?? 3));
  const examples: MovementExample[] = [];

  for (const trajectory of trajectories) {
    const tokens = movementTokensForTrajectory(trajectory);
    if (tokens.length < 2) {
      continue;
    }
    const weight = normalizeReward(trajectory.outcome?.reward);
    for (let index = 1; index < tokens.length; index += 1) {
      const start = Math.max(0, index - contextWindow);
      examples.push({
        context: tokens.slice(start, index),
        next: tokens[index]!,
        weight,
      });
    }
  }

  return { version: 1, contextWindow, examples };
}

function movementTokensForTrajectory(trajectory: TrajectorySpan): MovementActionToken[] {
  const redacted = trajectory.review?.redactedActions;
  const source = redacted
    ? redacted.map((action) => ({ tool: action.tool, summary: action.summary, ts: action.ts }))
    : trajectory.actions.map((action) => ({ tool: action.tool, summary: action.summary, ts: action.ts }));
  return [...source]
    .sort((a, b) => a.ts - b.ts)
    .map((action) => ({ tool: action.tool, summary: action.summary }));
}

function normalizeReward(reward: number | undefined): number {
  if (reward === undefined || Number.isNaN(reward)) {
    return 1;
  }
  // Keep weights strictly positive so negative-reward examples still inform the
  // model without cancelling counts; map reward → (1 + max(reward, -0.9)).
  return 1 + Math.max(reward, -0.9);
}

type MarkovLevelCounts = Record<string, Record<string, number>>;

type MarkovParameters = {
  contextWindow: number;
  /** suffixKey(order) → nextTokenKey → weighted count. */
  transitions: MarkovLevelCounts;
  /** lastTool → nextTokenKey → weighted count (generalization back-off). */
  toolTransitions: MarkovLevelCounts;
  /** nextTokenKey → weighted count (global prior). */
  prior: Record<string, number>;
  /** tokenKey → token, for reconstructing predictions. */
  vocabulary: Record<string, MovementActionToken>;
};

/**
 * Deterministic variable-order back-off Markov backend.
 *
 * Learns to *repeat* recorded movement sequences via exact n-gram suffix
 * matches, and *generalizes* to new-but-related movements by backing off from
 * the longest matching suffix → the last action's tool → the global prior.
 * Fully deterministic (stable tie-breaks), so cloud tests are reproducible.
 */
export class DeterministicMarkovMovementBackend implements MovementModelBackend {
  readonly id = "deterministic-markov";

  train(dataset: MovementDataset, options: MovementTrainOptions = {}): MovementModelArtifact {
    const contextWindow = clamp(
      Math.floor(options.contextWindow ?? dataset.contextWindow),
      1,
      Math.max(1, dataset.contextWindow),
    );

    const parameters: MarkovParameters = {
      contextWindow,
      transitions: {},
      toolTransitions: {},
      prior: {},
      vocabulary: {},
    };

    for (const example of dataset.examples) {
      const weight = example.weight === undefined ? 1 : Math.max(0, example.weight);
      if (weight === 0) {
        continue;
      }
      const nextKey = movementTokenKey(example.next);
      registerToken(parameters.vocabulary, example.next);
      for (const token of example.context) {
        registerToken(parameters.vocabulary, token);
      }

      // Global prior.
      bump(parameters.prior, nextKey, weight);

      const context = example.context;
      if (context.length === 0) {
        continue;
      }

      // Variable-order suffix transitions.
      const maxOrder = Math.min(contextWindow, context.length);
      for (let order = 1; order <= maxOrder; order += 1) {
        const suffix = context.slice(context.length - order);
        const key = suffixKey(suffix);
        bumpLevel(parameters.transitions, key, nextKey, weight);
      }

      // Tool-level back-off keyed on the most recent action's tool.
      const lastTool = context[context.length - 1]!.tool;
      bumpLevel(parameters.toolTransitions, lastTool, nextKey, weight);
    }

    return {
      backendId: this.id,
      version: 1,
      contextWindow,
      ...(options.trainedAt ? { trainedAt: options.trainedAt } : {}),
      metadata: {
        exampleCount: dataset.examples.length,
        vocabularySize: Object.keys(parameters.vocabulary).length,
      },
      parameters,
    };
  }

  predict(
    model: MovementModelArtifact,
    context: MovementActionToken[],
    options: MovementPredictOptions = {},
  ): MovementPrediction {
    const parameters = model.parameters as MarkovParameters;
    const maxCandidates = Math.max(1, Math.floor(options.maxCandidates ?? 5));

    // 1. Longest exact suffix match wins ("replay" of recorded movements).
    const maxOrder = Math.min(parameters.contextWindow, context.length);
    for (let order = maxOrder; order >= 1; order -= 1) {
      const suffix = context.slice(context.length - order);
      const counts = parameters.transitions[suffixKey(suffix)];
      const ranked = rankCounts(counts, parameters.vocabulary, maxCandidates);
      if (ranked) {
        return { ...ranked, source: "exact" };
      }
    }

    // 2. Tool-level back-off ("new but related" movements).
    if (context.length > 0) {
      const lastTool = context[context.length - 1]!.tool;
      const ranked = rankCounts(parameters.toolTransitions[lastTool], parameters.vocabulary, maxCandidates, 0.6);
      if (ranked) {
        return { ...ranked, source: "generalized" };
      }
    }

    // 3. Global prior.
    const ranked = rankCounts(parameters.prior, parameters.vocabulary, maxCandidates, 0.3);
    if (ranked) {
      return { ...ranked, source: "prior" };
    }

    return { action: undefined, confidence: 0, source: "none", candidates: [] };
  }
}

/**
 * Autoregressively generate a movement sequence from a seed context — i.e.
 * "repeat/continue the recorded movements". Stops on an unconfident/empty
 * prediction, on `stopWhen`, or after `maxSteps`.
 */
export function rolloutMovements(
  backend: MovementModelBackend,
  model: MovementModelArtifact,
  seedContext: MovementActionToken[],
  options: {
    maxSteps?: number;
    minConfidence?: number;
    stopWhen?: (token: MovementActionToken) => boolean;
  } = {},
): Array<MovementActionToken & { source: MovementPredictionSource; confidence: number }> {
  const maxSteps = Math.max(0, Math.floor(options.maxSteps ?? 16));
  const minConfidence = options.minConfidence ?? 0;
  const context = [...seedContext];
  const produced: Array<MovementActionToken & { source: MovementPredictionSource; confidence: number }> = [];

  for (let step = 0; step < maxSteps; step += 1) {
    const prediction = backend.predict(model, context);
    if (!prediction.action || prediction.confidence < minConfidence) {
      break;
    }
    const token = prediction.action;
    produced.push({ ...token, source: prediction.source, confidence: prediction.confidence });
    context.push(token);
    if (options.stopWhen?.(token)) {
      break;
    }
  }

  return produced;
}

export type MovementEvaluation = {
  /** Predictions attempted (one per (prefix → next) held-out example). */
  total: number;
  /** Fraction where tool AND summary matched the held-out next movement. */
  exactAccuracy: number;
  /** Fraction where at least the tool matched (looser, related-movement metric). */
  toolAccuracy: number;
  /** Fraction where the model produced any prediction (source !== "none"). */
  coverage: number;
  /** Breakdown of prediction sources across attempts. */
  sourceCounts: Record<MovementPredictionSource, number>;
};

/**
 * Generalization eval harness: measure next-movement fidelity on held-out (but
 * related) trajectories the model was not trained on.
 */
export function evaluateMovementModel(
  backend: MovementModelBackend,
  model: MovementModelArtifact,
  heldOut: TrajectorySpan[],
  options: { contextWindow?: number } = {},
): MovementEvaluation {
  const dataset = buildMovementDataset(heldOut, {
    contextWindow: options.contextWindow ?? model.contextWindow,
  });

  let exact = 0;
  let toolHits = 0;
  let covered = 0;
  const sourceCounts: Record<MovementPredictionSource, number> = {
    exact: 0,
    generalized: 0,
    prior: 0,
    none: 0,
  };

  for (const example of dataset.examples) {
    const prediction = backend.predict(model, example.context);
    sourceCounts[prediction.source] += 1;
    if (prediction.source !== "none") {
      covered += 1;
    }
    if (!prediction.action) {
      continue;
    }
    if (prediction.action.tool === example.next.tool) {
      toolHits += 1;
      if (prediction.action.summary === example.next.summary) {
        exact += 1;
      }
    }
  }

  const total = dataset.examples.length;
  const safe = (value: number): number => (total === 0 ? 0 : value / total);
  return {
    total,
    exactAccuracy: safe(exact),
    toolAccuracy: safe(toolHits),
    coverage: safe(covered),
    sourceCounts,
  };
}

/** Registry so callers can select a backend by id (real backends plug in here). */
export class MovementModelBackendRegistry {
  private readonly backends = new Map<string, MovementModelBackend>();

  constructor(backends: MovementModelBackend[] = [new DeterministicMarkovMovementBackend()]) {
    for (const backend of backends) {
      this.register(backend);
    }
  }

  register(backend: MovementModelBackend): void {
    this.backends.set(backend.id, backend);
  }

  get(id: string): MovementModelBackend | undefined {
    return this.backends.get(id);
  }

  require(id: string): MovementModelBackend {
    const backend = this.backends.get(id);
    if (!backend) {
      throw new Error(`unknown movement-model backend: ${id}`);
    }
    return backend;
  }

  list(): string[] {
    return [...this.backends.keys()].sort();
  }
}

function registerToken(vocabulary: Record<string, MovementActionToken>, token: MovementActionToken): void {
  const key = movementTokenKey(token);
  if (!vocabulary[key]) {
    vocabulary[key] = { tool: token.tool, summary: token.summary };
  }
}

function suffixKey(tokens: MovementActionToken[]): string {
  return tokens.map(movementTokenKey).join(SUFFIX_SEP);
}

function bump(counts: Record<string, number>, key: string, weight: number): void {
  counts[key] = (counts[key] ?? 0) + weight;
}

function bumpLevel(level: MarkovLevelCounts, outerKey: string, innerKey: string, weight: number): void {
  const inner = (level[outerKey] ??= {});
  inner[innerKey] = (inner[innerKey] ?? 0) + weight;
}

function rankCounts(
  counts: Record<string, number> | undefined,
  vocabulary: Record<string, MovementActionToken>,
  maxCandidates: number,
  confidenceScale = 1,
): { action: MovementActionToken; confidence: number; candidates: MovementPredictionCandidate[] } | undefined {
  if (!counts) {
    return undefined;
  }
  const entries = Object.entries(counts);
  if (entries.length === 0) {
    return undefined;
  }
  const total = entries.reduce((sum, [, value]) => sum + value, 0);
  if (total <= 0) {
    return undefined;
  }
  // Deterministic ordering: score desc, then token key asc.
  entries.sort((a, b) => (b[1] - a[1]) || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));

  const candidates: MovementPredictionCandidate[] = entries.slice(0, maxCandidates).map(([key, value]) => ({
    action: vocabulary[key] ?? decodeTokenKey(key),
    score: value / total,
  }));
  const top = candidates[0]!;
  return {
    action: top.action,
    confidence: clamp(top.score * confidenceScale, 0, 1),
    candidates,
  };
}

function decodeTokenKey(key: string): MovementActionToken {
  const separatorIndex = key.indexOf(TOKEN_FIELD_SEP);
  if (separatorIndex === -1) {
    return { tool: key, summary: "" };
  }
  return { tool: key.slice(0, separatorIndex), summary: key.slice(separatorIndex + 1) };
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}
