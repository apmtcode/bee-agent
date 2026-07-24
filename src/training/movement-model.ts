import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";
import type { ReplayManifest } from "../capture/replay.js";

/**
 * In-process, pluggable movement-model subsystem.
 *
 * This is the "post-train a local model to repeat recorded movements and
 * generalize to new-but-related movements" half of the local-movement learning
 * objective. The heavy on-device runtimes (mlx/axolotl) are still driven by
 * {@link ../training/runner.ts}; this module provides a *pluggable backend seam*
 * and a deterministic, dependency-free reference backend so the whole
 * capture -> dataset -> train -> infer -> replay loop can be exercised and tested
 * in the cloud without any real OS input or GPU.
 *
 * A movement is tokenized into a discrete {@link MovementStep} ("what channel,
 * what verb, optional qualifier"). Sequences of steps are what the model learns
 * from and generates.
 */

export type MovementStep = {
  /** Input channel, e.g. "device", "os", "browser". */
  channel: string;
  /** The salient action, e.g. "tap", "swipe", "type", "focus-changed". */
  verb: string;
  /** Optional generalizable detail, e.g. "down", or a bucketed target class. */
  qualifier?: string;
};

export type MovementSequence = {
  id: string;
  steps: MovementStep[];
};

export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

/** Sentinel tokens marking sequence boundaries in the n-gram model. */
export const MOVEMENT_START_TOKEN = "<s>";
export const MOVEMENT_END_TOKEN = "</s>";

const TOKEN_FIELD_SEPARATOR = "\u0001";

/** Encode a step into a stable, reversible string token. */
export function encodeMovementToken(step: MovementStep): string {
  const parts = [step.channel, step.verb];
  if (step.qualifier !== undefined) {
    parts.push(step.qualifier);
  }
  return parts.map((part) => part.replaceAll(TOKEN_FIELD_SEPARATOR, " ")).join(TOKEN_FIELD_SEPARATOR);
}

/** Decode a token produced by {@link encodeMovementToken}. Boundary tokens decode to undefined. */
export function decodeMovementToken(token: string): MovementStep | undefined {
  if (token === MOVEMENT_START_TOKEN || token === MOVEMENT_END_TOKEN) {
    return undefined;
  }
  const [channel = "", verb = "", qualifier] = token.split(TOKEN_FIELD_SEPARATOR);
  return qualifier !== undefined ? { channel, verb, qualifier } : { channel, verb };
}

export type MovementCandidate = {
  token: string;
  probability: number;
};

export type MovementPrediction = {
  /** Predicted next token; may be {@link MOVEMENT_END_TOKEN}. */
  token: string;
  /** Decoded step, or undefined when the model predicts end-of-sequence. */
  step: MovementStep | undefined;
  probability: number;
  /** n-gram order actually used after backoff (0 = unigram / no context). */
  order: number;
  /** Full ranked distribution used for the prediction. */
  candidates: MovementCandidate[];
};

export type MovementModelSnapshot = {
  backendId: string;
  order: number;
  vocabulary: string[];
  /** transitions[contextKey][token] = observed count. Context keys are order-tagged. */
  transitions: Record<string, Record<string, number>>;
};

export type MovementGenerateParams = {
  /** Optional priming steps; the model continues from here. */
  seed?: MovementStep[];
  /** Hard cap on generated steps (excludes the seed). Defaults to 64. */
  maxSteps?: number;
  /**
   * When set, sampling is stochastic but *deterministic* for a given seed value
   * (seeded LCG), letting the model produce new-but-related variations. When
   * omitted, generation is greedy (argmax) and reproduces the dominant path.
   */
  randomSeed?: number;
};

/** A model produced by a backend: inference + serialization. */
export interface TrainedMovementModel {
  readonly backendId: string;
  /** Predict the next step given a context of prior tokens (most recent last). */
  predictNext(context: string[]): MovementPrediction;
  /** Generate a fresh movement sequence, optionally primed by `seed`. */
  generate(params?: MovementGenerateParams): MovementStep[];
  /** Serialize to a portable snapshot. */
  snapshot(): MovementModelSnapshot;
}

export type MovementTrainOptions = {
  /** Maximum n-gram context length. Higher = more faithful, less general. Default 3. */
  order?: number;
};

/** Pluggable backend seam. Real on-device backends implement the same shape. */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainOptions): Promise<TrainedMovementModel>;
  restore(snapshot: MovementModelSnapshot): TrainedMovementModel;
}

// ---------------------------------------------------------------------------
// Reference backend: a Katz-style backoff n-gram over movement tokens.
// Deterministic, dependency-free, and genuinely generalizing (unseen contexts
// back off to shorter contexts), which is exactly what the eval harness checks.
// ---------------------------------------------------------------------------

const DEFAULT_ORDER = 3;
const DEFAULT_MAX_STEPS = 64;

function contextKey(order: number, tokens: string[]): string {
  return `${order}${TOKEN_FIELD_SEPARATOR}${tokens.join(TOKEN_FIELD_SEPARATOR)}`;
}

class NgramMovementModel implements TrainedMovementModel {
  constructor(
    readonly backendId: string,
    private readonly order: number,
    private readonly vocabulary: string[],
    private readonly transitions: Map<string, Map<string, number>>,
  ) {}

  predictNext(context: string[]): MovementPrediction {
    for (let order = Math.min(this.order, context.length); order >= 1; order -= 1) {
      const suffix = context.slice(context.length - order);
      const counts = this.transitions.get(contextKey(order, suffix));
      if (counts && counts.size > 0) {
        return toPrediction(counts, order);
      }
    }
    const unigram = this.transitions.get(contextKey(0, []));
    if (unigram && unigram.size > 0) {
      return toPrediction(unigram, 0);
    }
    return {
      token: MOVEMENT_END_TOKEN,
      step: undefined,
      probability: 1,
      order: 0,
      candidates: [{ token: MOVEMENT_END_TOKEN, probability: 1 }],
    };
  }

  generate(params: MovementGenerateParams = {}): MovementStep[] {
    const maxSteps = params.maxSteps ?? DEFAULT_MAX_STEPS;
    const seedSteps = params.seed ?? [];
    const context: string[] = [
      ...Array.from({ length: this.order }, () => MOVEMENT_START_TOKEN),
      ...seedSteps.map(encodeMovementToken),
    ];
    const rng = params.randomSeed === undefined ? undefined : createLcg(params.randomSeed);
    const produced: MovementStep[] = [...seedSteps];

    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predictNext(context);
      const token = rng ? sample(prediction.candidates, rng) : prediction.token;
      if (token === MOVEMENT_END_TOKEN) {
        break;
      }
      const decoded = decodeMovementToken(token);
      if (decoded) {
        produced.push(decoded);
      }
      context.push(token);
    }
    return produced;
  }

  snapshot(): MovementModelSnapshot {
    const transitions: Record<string, Record<string, number>> = {};
    for (const [key, counts] of this.transitions) {
      transitions[key] = Object.fromEntries(counts);
    }
    return {
      backendId: this.backendId,
      order: this.order,
      vocabulary: [...this.vocabulary],
      transitions,
    };
  }
}

function toPrediction(counts: Map<string, number>, order: number): MovementPrediction {
  let total = 0;
  for (const value of counts.values()) {
    total += value;
  }
  const candidates: MovementCandidate[] = [...counts.entries()]
    .map(([token, count]) => ({ token, probability: count / total }))
    // Deterministic ordering: higher probability first, then lexical token order.
    .sort((a, b) => (b.probability - a.probability) || (a.token < b.token ? -1 : a.token > b.token ? 1 : 0));
  const top = candidates[0]!;
  return {
    token: top.token,
    step: decodeMovementToken(top.token),
    probability: top.probability,
    order,
    candidates,
  };
}

/** Small linear-congruential RNG so stochastic generation stays reproducible. */
function createLcg(seed: number): () => number {
  let state = (Math.abs(Math.trunc(seed)) % 2147483647) || 1;
  return () => {
    state = (state * 48271) % 2147483647;
    return (state - 1) / 2147483646;
  };
}

function sample(candidates: MovementCandidate[], rng: () => number): string {
  const roll = rng();
  let cumulative = 0;
  for (const candidate of candidates) {
    cumulative += candidate.probability;
    if (roll <= cumulative) {
      return candidate.token;
    }
  }
  return candidates[candidates.length - 1]!.token;
}

export class NgramMovementBackend implements MovementModelBackend {
  readonly id = "ngram-backoff";

  async train(dataset: MovementDataset, options: MovementTrainOptions = {}): Promise<TrainedMovementModel> {
    const order = Math.max(1, Math.trunc(options.order ?? DEFAULT_ORDER));
    const transitions = new Map<string, Map<string, number>>();
    const vocabulary = new Set<string>();

    const bump = (key: string, token: string): void => {
      let counts = transitions.get(key);
      if (!counts) {
        counts = new Map<string, number>();
        transitions.set(key, counts);
      }
      counts.set(token, (counts.get(token) ?? 0) + 1);
    };

    for (const sequence of dataset.sequences) {
      const tokens = sequence.steps.map(encodeMovementToken);
      tokens.forEach((token) => vocabulary.add(token));
      const padded = [
        ...Array.from({ length: order }, () => MOVEMENT_START_TOKEN),
        ...tokens,
        MOVEMENT_END_TOKEN,
      ];
      for (let index = order; index < padded.length; index += 1) {
        const next = padded[index]!;
        bump(contextKey(0, []), next); // unigram
        for (let o = 1; o <= order; o += 1) {
          const suffix = padded.slice(index - o, index);
          bump(contextKey(o, suffix), next);
        }
      }
    }

    return new NgramMovementModel(this.id, order, [...vocabulary].sort(), transitions);
  }

  restore(snapshot: MovementModelSnapshot): TrainedMovementModel {
    const transitions = new Map<string, Map<string, number>>();
    for (const [key, counts] of Object.entries(snapshot.transitions)) {
      transitions.set(key, new Map(Object.entries(counts)));
    }
    return new NgramMovementModel(snapshot.backendId, snapshot.order, [...snapshot.vocabulary], transitions);
  }
}

// ---------------------------------------------------------------------------
// Dataset builders: turn captured trajectories / replay manifests into
// tokenizable movement sequences.
// ---------------------------------------------------------------------------

const MAX_TARGET_BUCKET_LENGTH = 24;

function firstWord(text: string): string {
  const match = text.trim().match(/^[a-z0-9_-]+/i);
  return match ? match[0].toLowerCase() : "act";
}

/** Bucket a free-form target into a stable, low-cardinality class for generalization. */
function bucketTarget(target: string): string {
  const normalized = target.trim().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
  return normalized.slice(0, MAX_TARGET_BUCKET_LENGTH) || "unknown";
}

/** Tokenize a single captured action into a normalized movement step. */
export function movementStepFromAction(action: TrajectoryAction): MovementStep {
  const metadata = action.metadata ?? {};
  const channel = String(action.tool || "action").trim().toLowerCase() || "action";
  const gesture = typeof metadata.gesture === "string" ? metadata.gesture : undefined;
  const event = typeof metadata.event === "string" ? metadata.event : undefined;
  const verb = (gesture ?? event ?? firstWord(action.summary)).toLowerCase();
  const direction = typeof metadata.direction === "string" ? metadata.direction : undefined;
  const target = typeof metadata.target === "string" ? metadata.target : undefined;
  const qualifier = direction ?? (target ? bucketTarget(target) : undefined);
  return qualifier ? { channel, verb, qualifier } : { channel, verb };
}

/** Build a movement dataset from captured trajectory spans (one sequence per span). */
export function buildMovementDatasetFromTrajectories(trajectories: TrajectorySpan[]): MovementDataset {
  const sequences: MovementSequence[] = [];
  for (const trajectory of trajectories) {
    const orderedActions = [...trajectory.actions].sort((a, b) => a.ts - b.ts);
    const steps = orderedActions.map(movementStepFromAction);
    if (steps.length > 0) {
      sequences.push({ id: trajectory.id, steps });
    }
  }
  return { version: 1, sequences };
}

/** Build a movement dataset from replay manifests (one sequence per manifest). */
export function buildMovementDatasetFromReplays(replays: ReplayManifest[]): MovementDataset {
  const sequences: MovementSequence[] = [];
  for (const replay of replays) {
    const steps: MovementStep[] = [];
    for (const event of replay.events) {
      if (event.kind === "action") {
        steps.push({ channel: event.tool.trim().toLowerCase() || "action", verb: firstWord(event.summary) });
      }
    }
    if (steps.length > 0) {
      sequences.push({ id: replay.trajectoryIds.join("+") || replay.sessionId, steps });
    }
  }
  return { version: 1, sequences };
}

// ---------------------------------------------------------------------------
// Generalization eval harness: next-step prediction accuracy on held-out data.
// ---------------------------------------------------------------------------

export type MovementEvalResult = {
  sequenceCount: number;
  predictionCount: number;
  correct: number;
  /** Fraction of held-out next-step predictions the model got exactly right. */
  accuracy: number;
  /** Mean n-gram order used across predictions (lower = relied more on backoff). */
  averageBackoffOrder: number;
  /** Fraction of predictions that required backing off below the model's max order. */
  backoffRate: number;
};

/**
 * Walk each held-out sequence and, at every position, ask the model to predict
 * the next step from the true prefix. Measures how well a model trained on one
 * set generalizes to related-but-unseen sequences.
 */
export function evaluateNextStepPrediction(
  model: TrainedMovementModel,
  heldOut: MovementDataset,
  options: { order?: number } = {},
): MovementEvalResult {
  const maxOrder = options.order ?? Infinity;
  let predictionCount = 0;
  let correct = 0;
  let orderSum = 0;
  let backoffCount = 0;

  for (const sequence of heldOut.sequences) {
    const tokens = sequence.steps.map(encodeMovementToken);
    const context: string[] = [];
    for (const expected of tokens) {
      const prediction = model.predictNext(context);
      predictionCount += 1;
      orderSum += prediction.order;
      if (prediction.order < Math.min(maxOrder, context.length + 1) && context.length > 0) {
        backoffCount += 1;
      }
      if (prediction.token === expected) {
        correct += 1;
      }
      context.push(expected);
    }
  }

  return {
    sequenceCount: heldOut.sequences.length,
    predictionCount,
    correct,
    accuracy: predictionCount === 0 ? 0 : correct / predictionCount,
    averageBackoffOrder: predictionCount === 0 ? 0 : orderSum / predictionCount,
    backoffRate: predictionCount === 0 ? 0 : backoffCount / predictionCount,
  };
}
