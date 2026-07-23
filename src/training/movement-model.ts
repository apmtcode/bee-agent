import type { TrajectorySpan } from "../capture/trajectory.js";
import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";

/**
 * Local-movement learning subsystem — in-process model backend.
 *
 * Standing objective #2 (c)/(d): "post-train a local model on that dataset to
 * repeat the recorded movements" and "generalize to perform new but related
 * movements". The on-device runner (`runner.ts`) only *plans* an external
 * MLX/Axolotl process, so parts (c)/(d) cannot be exercised in the cloud/CI.
 *
 * This module provides a *pluggable* {@link MovementModelBackend} seam plus a
 * deterministic, dependency-free reference implementation
 * ({@link MarkovMovementBackend}) that actually learns movement structure from
 * a dataset and can both replay recorded movements and generate new-but-related
 * ones. It runs entirely in-process so the whole capture→dataset→train→infer
 * loop is testable without real OS input or on-device training. A real
 * on-device small model can be dropped in behind the same interface.
 */

/** A discrete movement token, e.g. `"action:click"` or `"obs:screen"`. */
export type MovementToken = string;

/** Sentinel tokens framing every training sequence. */
export const MOVEMENT_START_TOKEN: MovementToken = "<s>";
export const MOVEMENT_END_TOKEN: MovementToken = "</s>";

export type MovementSequence = {
  /** Source trajectory (or replay session) id, for traceability. */
  id: string;
  /** Ordered movement tokens, *including* the START/END sentinels. */
  tokens: MovementToken[];
  /** Optional scalar reward carried from the trajectory outcome. */
  reward?: number;
};

export type MovementTrainingDataset = {
  version: 1;
  sequences: MovementSequence[];
  /** Sorted, de-duplicated token vocabulary (includes sentinels). */
  vocabulary: MovementToken[];
};

export type MovementTokenizeOptions = {
  /** Include `obs:<source>` tokens alongside `action:<tool>` tokens. */
  includeObservations?: boolean;
  /** Include an `outcome:<status>` token at the end when present. */
  includeOutcome?: boolean;
};

export type MovementTrainingOptions = {
  /** Markov context order (number of preceding tokens). Default 1. */
  order?: number;
  /** Optional ISO timestamp to stamp onto the model (callers supply — the
   * library stays deterministic and never reads the clock). */
  trainedAt?: string;
};

export type MovementModelTransition = {
  /** Joined context key (the preceding `order` tokens). */
  context: string;
  /** Next-token counts, sorted by count desc then token asc (deterministic). */
  distribution: Array<{ token: MovementToken; count: number }>;
};

/** A fully serializable, backend-agnostic trained model artifact. */
export type TrainedMovementModel = {
  version: 1;
  backend: string;
  order: number;
  vocabulary: MovementToken[];
  transitions: MovementModelTransition[];
  sequenceCount: number;
  tokenCount: number;
  trainedAt?: string;
};

export type MovementPredictionCandidate = {
  token: MovementToken;
  probability: number;
};

export type MovementPrediction = {
  context: MovementToken[];
  /** Candidates ranked by probability desc (then token asc). */
  candidates: MovementPredictionCandidate[];
  /** Argmax next token, or undefined when the context is unknown. */
  next?: MovementToken;
};

export type MovementGenerationOptions = {
  /** Hard cap on generated tokens (excluding the prompt). Default 64. */
  maxSteps?: number;
  /** Sample from the distribution using `seed` instead of greedy argmax. */
  sample?: boolean;
  /** Seed for reproducible sampling (required for `sample`). */
  seed?: number;
};

export type MovementGenerationResult = {
  prompt: MovementToken[];
  /** Tokens produced after the prompt, excluding the END sentinel. */
  generated: MovementToken[];
  /** Full token stream: prompt + generated. */
  sequence: MovementToken[];
  stoppedReason: "end" | "max-steps" | "no-transition";
};

/**
 * Pluggable backend seam. `train` may be async (real backends do I/O / spawn
 * on-device training); `predict`/`generate` are pure functions of the model so
 * they stay synchronous and cheap.
 */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementTrainingDataset, options?: MovementTrainingOptions): Promise<TrainedMovementModel>;
  predict(model: TrainedMovementModel, context: MovementToken[]): MovementPrediction;
  generate(
    model: TrainedMovementModel,
    prompt: MovementToken[],
    options?: MovementGenerationOptions,
  ): MovementGenerationResult;
}

// ---------------------------------------------------------------------------
// Tokenization + dataset builders
// ---------------------------------------------------------------------------

function normalizeSegment(value: string): string {
  return value.trim().toLowerCase().replace(/\s+/g, "-");
}

export function actionToken(tool: string): MovementToken {
  return `action:${normalizeSegment(tool)}`;
}

export function observationToken(source: string): MovementToken {
  return `obs:${normalizeSegment(source)}`;
}

/** Turn a single trajectory into an ordered, sentinel-framed token sequence. */
export function tokenizeTrajectory(
  trajectory: TrajectorySpan,
  options: MovementTokenizeOptions = {},
): MovementToken[] {
  const includeObservations = options.includeObservations ?? false;
  type Timed = { ts: number; order: number; token: MovementToken };
  const timed: Timed[] = [];

  if (includeObservations) {
    for (const observation of trajectory.observations) {
      timed.push({ ts: observation.ts, order: 0, token: observationToken(observation.source) });
    }
  }
  for (const action of trajectory.actions) {
    timed.push({ ts: action.ts, order: 1, token: actionToken(action.tool) });
  }

  timed.sort((a, b) => (a.ts !== b.ts ? a.ts - b.ts : a.order - b.order));

  const tokens: MovementToken[] = [MOVEMENT_START_TOKEN, ...timed.map((entry) => entry.token)];
  if ((options.includeOutcome ?? false) && trajectory.outcome) {
    tokens.push(`outcome:${normalizeSegment(trajectory.outcome.status)}`);
  }
  tokens.push(MOVEMENT_END_TOKEN);
  return tokens;
}

function toDataset(sequences: MovementSequence[]): MovementTrainingDataset {
  const vocabulary = new Set<MovementToken>();
  for (const sequence of sequences) {
    for (const token of sequence.tokens) {
      vocabulary.add(token);
    }
  }
  return {
    version: 1,
    sequences,
    vocabulary: [...vocabulary].sort(),
  };
}

/** Build a training dataset from reviewed trajectory spans. */
export function buildMovementDataset(
  trajectories: TrajectorySpan[],
  options: MovementTokenizeOptions = {},
): MovementTrainingDataset {
  const sequences = trajectories
    .map<MovementSequence>((trajectory) => ({
      id: trajectory.id,
      tokens: tokenizeTrajectory(trajectory, options),
      ...(trajectory.outcome?.reward !== undefined ? { reward: trajectory.outcome.reward } : {}),
    }))
    // A sequence with only sentinels carries no movement signal.
    .filter((sequence) => sequence.tokens.length > 2);
  return toDataset(sequences);
}

/** Build a training dataset from replay manifests (timeline events). */
export function buildMovementDatasetFromReplays(
  replays: ReplayManifest[],
  options: MovementTokenizeOptions = {},
): MovementTrainingDataset {
  const includeObservations = options.includeObservations ?? false;
  const sequences = replays
    .map<MovementSequence>((replay) => {
      const ordered = [...replay.events].sort((a, b) => a.ts - b.ts);
      const movementTokens = ordered.flatMap((event) => replayEventToTokens(event, includeObservations));
      return {
        id: replay.sessionId,
        tokens: [MOVEMENT_START_TOKEN, ...movementTokens, MOVEMENT_END_TOKEN],
      };
    })
    .filter((sequence) => sequence.tokens.length > 2);
  return toDataset(sequences);
}

function replayEventToTokens(event: ReplayTimelineEvent, includeObservations: boolean): MovementToken[] {
  switch (event.kind) {
    case "action":
      return [actionToken(event.tool)];
    case "observation":
      return includeObservations ? [observationToken(event.source)] : [];
    case "transcript":
      return [];
  }
}

// ---------------------------------------------------------------------------
// Deterministic reference backend: order-k Markov model
// ---------------------------------------------------------------------------

/** Deterministic PRNG (mulberry32) — reproducible sampling without OS entropy. */
function createSeededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state |= 0;
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function contextKey(tokens: MovementToken[], order: number): string {
  return tokens.slice(-order).join("");
}

/**
 * A dependency-free, fully deterministic movement model backend. It learns an
 * order-k Markov model over movement tokens: replays recorded movements exactly
 * (high-probability paths) and generalizes by recombining learned transitions
 * across trajectories into new-but-related sequences.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov";

  async train(
    dataset: MovementTrainingDataset,
    options: MovementTrainingOptions = {},
  ): Promise<TrainedMovementModel> {
    const order = Math.max(1, Math.floor(options.order ?? 1));
    const counts = new Map<string, Map<MovementToken, number>>();
    let tokenCount = 0;

    for (const sequence of dataset.sequences) {
      const { tokens } = sequence;
      tokenCount += tokens.length;
      for (let index = order; index < tokens.length; index += 1) {
        const key = contextKey(tokens.slice(index - order, index), order);
        const next = tokens[index]!;
        let distribution = counts.get(key);
        if (!distribution) {
          distribution = new Map<MovementToken, number>();
          counts.set(key, distribution);
        }
        distribution.set(next, (distribution.get(next) ?? 0) + 1);
      }
    }

    const transitions: MovementModelTransition[] = [...counts.entries()]
      .map(([context, distribution]) => ({
        context,
        distribution: [...distribution.entries()]
          .map(([token, count]) => ({ token, count }))
          .sort((a, b) => (b.count !== a.count ? b.count - a.count : a.token < b.token ? -1 : 1)),
      }))
      .sort((a, b) => (a.context < b.context ? -1 : a.context > b.context ? 1 : 0));

    return {
      version: 1,
      backend: this.id,
      order,
      vocabulary: dataset.vocabulary,
      transitions,
      sequenceCount: dataset.sequences.length,
      tokenCount,
      ...(options.trainedAt ? { trainedAt: options.trainedAt } : {}),
    };
  }

  predict(model: TrainedMovementModel, context: MovementToken[]): MovementPrediction {
    const key = contextKey(context.length ? context : [MOVEMENT_START_TOKEN], model.order);
    const transition = lookupTransition(model, key);
    if (!transition || transition.distribution.length === 0) {
      return { context, candidates: [] };
    }
    const total = transition.distribution.reduce((sum, entry) => sum + entry.count, 0);
    const candidates = transition.distribution.map((entry) => ({
      token: entry.token,
      probability: entry.count / total,
    }));
    return { context, candidates, next: candidates[0]?.token };
  }

  generate(
    model: TrainedMovementModel,
    prompt: MovementToken[],
    options: MovementGenerationOptions = {},
  ): MovementGenerationResult {
    const maxSteps = Math.max(0, options.maxSteps ?? 64);
    const random = options.sample ? createSeededRandom(options.seed ?? 0) : undefined;
    const sequence: MovementToken[] = prompt.length ? [...prompt] : [MOVEMENT_START_TOKEN];
    const generated: MovementToken[] = [];
    let stoppedReason: MovementGenerationResult["stoppedReason"] = "max-steps";

    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predict(model, sequence);
      if (prediction.candidates.length === 0) {
        stoppedReason = "no-transition";
        break;
      }
      const token = random
        ? sampleCandidate(prediction.candidates, random())
        : prediction.candidates[0]!.token;
      if (token === MOVEMENT_END_TOKEN) {
        stoppedReason = "end";
        break;
      }
      sequence.push(token);
      generated.push(token);
    }

    return { prompt, generated, sequence, stoppedReason };
  }
}

function lookupTransition(
  model: TrainedMovementModel,
  key: string,
): MovementModelTransition | undefined {
  // Transitions are context-sorted; a binary search keeps prediction cheap for
  // large models while staying purely a function of the serialized artifact.
  let low = 0;
  let high = model.transitions.length - 1;
  while (low <= high) {
    const mid = (low + high) >> 1;
    const midKey = model.transitions[mid]!.context;
    if (midKey === key) {
      return model.transitions[mid];
    }
    if (midKey < key) {
      low = mid + 1;
    } else {
      high = mid - 1;
    }
  }
  return undefined;
}

function sampleCandidate(candidates: MovementPredictionCandidate[], roll: number): MovementToken {
  let cumulative = 0;
  for (const candidate of candidates) {
    cumulative += candidate.probability;
    if (roll < cumulative) {
      return candidate.token;
    }
  }
  return candidates[candidates.length - 1]!.token;
}

// ---------------------------------------------------------------------------
// Generalization eval harness
// ---------------------------------------------------------------------------

export type MovementEvalResult = {
  sequenceCount: number;
  /** Number of next-token predictions attempted across all sequences. */
  predictedCount: number;
  /** Predictions whose argmax matched the held-out token. */
  correctCount: number;
  /** correctCount / predictedCount (0 when nothing was predicted). */
  accuracy: number;
};

/**
 * Measure next-token replay fidelity: for every position in each sequence,
 * predict from the preceding context and compare the argmax to the actual
 * token. Use a *held-out* dataset (see {@link splitMovementDataset}) to measure
 * generalization to new-but-related trajectories.
 */
export function evaluateNextTokenAccuracy(
  backend: MovementModelBackend,
  model: TrainedMovementModel,
  dataset: MovementTrainingDataset,
): MovementEvalResult {
  let predictedCount = 0;
  let correctCount = 0;
  for (const sequence of dataset.sequences) {
    const { tokens } = sequence;
    for (let index = model.order; index < tokens.length; index += 1) {
      const context = tokens.slice(index - model.order, index);
      const prediction = backend.predict(model, context);
      if (prediction.next === undefined) {
        continue;
      }
      predictedCount += 1;
      if (prediction.next === tokens[index]) {
        correctCount += 1;
      }
    }
  }
  return {
    sequenceCount: dataset.sequences.length,
    predictedCount,
    correctCount,
    accuracy: predictedCount === 0 ? 0 : correctCount / predictedCount,
  };
}

/**
 * Deterministically split a dataset into train/holdout partitions by a stable
 * hash of each sequence id, so the same dataset always yields the same split
 * (no clock/entropy) — useful for reproducible generalization evals.
 */
export function splitMovementDataset(
  dataset: MovementTrainingDataset,
  holdoutRatio = 0.2,
): { train: MovementTrainingDataset; holdout: MovementTrainingDataset } {
  const ratio = Math.min(1, Math.max(0, holdoutRatio));
  const train: MovementSequence[] = [];
  const holdout: MovementSequence[] = [];
  for (const sequence of dataset.sequences) {
    const bucket = (stableHash(sequence.id) % 1000) / 1000;
    if (bucket < ratio) {
      holdout.push(sequence);
    } else {
      train.push(sequence);
    }
  }
  return { train: toDataset(train), holdout: toDataset(holdout) };
}

function stableHash(value: string): number {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

// ---------------------------------------------------------------------------
// Trainer orchestrator
// ---------------------------------------------------------------------------

/**
 * Ties a pluggable backend to the dataset/predict/generate/evaluate surface so
 * callers depend on one object regardless of which backend is wired in. Swap
 * {@link MarkovMovementBackend} for a real on-device backend without touching
 * call sites.
 */
export class MovementModelTrainer {
  constructor(private readonly backend: MovementModelBackend = new MarkovMovementBackend()) {}

  get backendId(): string {
    return this.backend.id;
  }

  train(dataset: MovementTrainingDataset, options?: MovementTrainingOptions): Promise<TrainedMovementModel> {
    return this.backend.train(dataset, options);
  }

  predict(model: TrainedMovementModel, context: MovementToken[]): MovementPrediction {
    return this.backend.predict(model, context);
  }

  generate(
    model: TrainedMovementModel,
    prompt: MovementToken[],
    options?: MovementGenerationOptions,
  ): MovementGenerationResult {
    return this.backend.generate(model, prompt, options);
  }

  evaluate(model: TrainedMovementModel, dataset: MovementTrainingDataset): MovementEvalResult {
    return evaluateNextTokenAccuracy(this.backend, model, dataset);
  }
}
