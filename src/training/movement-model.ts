import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Pluggable local movement-model subsystem.
 *
 * This is the in-process, deterministic core of standing objective #2's
 * train + generalize pieces. The real on-device runner
 * ({@link ../training/runner.ts LocalAppleSiliconTrainingRunner}) emits
 * external mlx/axolotl launch plans that only execute on the user's machine;
 * nothing there is runnable — or testable — in the cloud.
 *
 * This module closes that gap with a {@link MovementModelBackend} interface and
 * a deterministic {@link MarkovMovementBackend} default that actually learns
 * from a recorded {@link MovementDataset} and can:
 *  - repeat the recorded movements (high-probability transitions replay the
 *    training sequences), and
 *  - generalize to new-but-related movements (feature backoff predicts a
 *    plausible action for contexts never seen verbatim in training).
 *
 * The backend is pluggable: a future real small on-device model implements the
 * same interface, so call sites and tests are backend-agnostic.
 */

/** Context features a movement is conditioned on (all optional; sparse). */
export type MovementContextFeatures = {
  /** App identifier the movement occurred in (from observation metadata). */
  app?: string;
  /** Device/OS platform, when known. */
  platform?: string;
  /** Tool of the immediately preceding action, if any. */
  lastTool?: string;
  /** Gesture kind of the immediately preceding action, if any. */
  lastGesture?: string;
  /** Target of the immediately preceding action, if any (sequences the chain). */
  lastTarget?: string;
  /** Source of the most recent observation, if any. */
  observationSource?: string;
};

/** A single learned/predicted movement, decoupled from raw metadata. */
export type MovementActionToken = {
  tool: string;
  gesture?: string;
  target?: string;
  direction?: string;
  summary: string;
};

/** One (context → action) transition observed in a trajectory. */
export type MovementExample = {
  context: MovementContextFeatures;
  action: MovementActionToken;
};

/** All transitions of a single trajectory, in temporal order. */
export type MovementSequence = {
  trajectoryId: string;
  sessionId: string;
  outcome?: "success" | "failure" | "aborted";
  reward?: number;
  examples: MovementExample[];
};

/** A replayable, model-ready dataset built from recorded trajectories. */
export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

/**
 * Sentinel `lastTarget` marking the start of a movement chain, so the first
 * step of a sequence is a distinct, matchable context rather than colliding
 * with the app-only aggregate of every step.
 */
export const MOVEMENT_START = "__start__";

/**
 * Reserved tool marking end-of-chain. Each recorded sequence contributes one
 * terminal example so the model learns *when to stop*, not just what to do
 * next; {@link rolloutMovements} halts (without emitting) when it predicts this.
 */
export const MOVEMENT_END_TOOL = "__end__";

const END_ACTION: MovementActionToken = { tool: MOVEMENT_END_TOOL, summary: "end of movement chain" };

export type MovementPredictionCandidate = {
  action: MovementActionToken;
  probability: number;
};

export type MovementPrediction = {
  /** Most likely next action, or undefined when the model has no signal. */
  action: MovementActionToken | undefined;
  /** Probability mass of {@link action} at the matched backoff level. */
  confidence: number;
  /**
   * Generalization distance: 0 = full-context match (a movement the model saw
   * in this exact context), higher = predicted via progressively coarser
   * context. -1 means no prediction was possible.
   */
  backoffLevel: number;
  /** Full ranked distribution at the matched backoff level. */
  distribution: MovementPredictionCandidate[];
};

/**
 * A trained model plus the backend that produced it. Backends are pluggable;
 * the model payload is opaque to callers and round-trips via serialize.
 */
export interface MovementModelBackend<TModel> {
  readonly name: string;
  train(dataset: MovementDataset, options?: MovementTrainOptions): TModel;
  predict(model: TModel, context: MovementContextFeatures): MovementPrediction;
  serialize(model: TModel): string;
  deserialize(data: string): TModel;
}

export type MovementTrainOptions = {
  /**
   * Additive (Laplace) smoothing applied to every observed action count.
   * Keeps rare-but-real movements from collapsing to zero probability.
   */
  smoothing?: number;
  /**
   * Weight applied to a sequence's contribution by outcome. Successful
   * trajectories teach more strongly than aborted/failed ones. Defaults favour
   * success (1.0) over failure (0.25) and aborted (0.1).
   */
  outcomeWeights?: Partial<Record<NonNullable<MovementSequence["outcome"]>, number>>;
};

const DEFAULT_OUTCOME_WEIGHTS: Record<NonNullable<MovementSequence["outcome"]>, number> = {
  success: 1,
  failure: 0.25,
  aborted: 0.1,
};

// --- Dataset construction -------------------------------------------------

/**
 * Build a {@link MovementDataset} from recorded trajectory spans. Actions are
 * walked in timestamp order; each action's context is derived from the app it
 * ran in (from the nearest preceding observation) and the action before it, so
 * the model learns ordered movement chains rather than a bag of clicks.
 */
export function buildMovementDataset(trajectories: TrajectorySpan[]): MovementDataset {
  const sequences = trajectories.map<MovementSequence>((trajectory) => {
    const orderedActions = [...trajectory.actions].sort((a, b) => a.ts - b.ts);
    const orderedObservations = [...trajectory.observations].sort((a, b) => a.ts - b.ts);

    let previous: MovementActionToken | undefined;
    const examples = orderedActions.map<MovementExample>((action) => {
      const observation = latestObservationBefore(orderedObservations, action.ts);
      const token = toActionToken(action);
      const context: MovementContextFeatures = {
        ...(observationApp(observation) ? { app: observationApp(observation) } : {}),
        ...(observationPlatform(observation) ? { platform: observationPlatform(observation) } : {}),
        ...(previous ? { lastTool: previous.tool } : {}),
        ...(previous?.gesture ? { lastGesture: previous.gesture } : {}),
        lastTarget: sequencingSignal(previous),
        ...(observation ? { observationSource: observation.source } : {}),
      };
      previous = token;
      return { context, action: token };
    });

    // Terminal example: after the last action, the correct move is to stop.
    if (previous) {
      const lastObservation = orderedObservations[orderedObservations.length - 1];
      examples.push({
        context: {
          ...(observationApp(lastObservation) ? { app: observationApp(lastObservation) } : {}),
          ...(observationPlatform(lastObservation) ? { platform: observationPlatform(lastObservation) } : {}),
          lastTool: previous.tool,
          ...(previous.gesture ? { lastGesture: previous.gesture } : {}),
          lastTarget: sequencingSignal(previous),
        },
        action: END_ACTION,
      });
    }

    return {
      trajectoryId: trajectory.id,
      sessionId: trajectory.sessionId,
      ...(trajectory.outcome ? { outcome: trajectory.outcome.status } : {}),
      ...(trajectory.outcome?.reward !== undefined ? { reward: trajectory.outcome.reward } : {}),
      examples,
    };
  });

  return { version: 1, sequences };
}

/**
 * The signal that sequences a chain: the previous action's target, falling back
 * to its gesture/tool, or {@link MOVEMENT_START} at the head of a sequence.
 */
function sequencingSignal(previous: MovementActionToken | undefined): string {
  if (!previous) {
    return MOVEMENT_START;
  }
  return previous.target ?? previous.gesture ?? previous.tool;
}

function latestObservationBefore(
  observations: TrajectorySpan["observations"],
  ts: number,
): TrajectorySpan["observations"][number] | undefined {
  let match: TrajectorySpan["observations"][number] | undefined;
  for (const observation of observations) {
    if (observation.ts <= ts) {
      match = observation;
    } else {
      break;
    }
  }
  return match ?? observations[0];
}

function observationApp(observation: TrajectorySpan["observations"][number] | undefined): string | undefined {
  const value = observation?.metadata?.["appName"] ?? observation?.metadata?.["appId"];
  return typeof value === "string" ? value : undefined;
}

function observationPlatform(observation: TrajectorySpan["observations"][number] | undefined): string | undefined {
  const value = observation?.metadata?.["platform"];
  return typeof value === "string" ? value : undefined;
}

function toActionToken(action: TrajectorySpan["actions"][number]): MovementActionToken {
  const metadata = action.metadata ?? {};
  const gesture = typeof metadata["gesture"] === "string" ? metadata["gesture"] : undefined;
  const target = typeof metadata["target"] === "string" ? metadata["target"] : undefined;
  const direction = typeof metadata["direction"] === "string" ? metadata["direction"] : undefined;
  return {
    tool: action.tool,
    ...(gesture ? { gesture } : {}),
    ...(target ? { target } : {}),
    ...(direction ? { direction } : {}),
    summary: action.summary,
  };
}

/** Canonical, order-independent key for an action token (used for counting). */
export function movementActionKey(action: MovementActionToken): string {
  return JSON.stringify([action.tool, action.gesture ?? "", action.target ?? "", action.direction ?? ""]);
}

// --- Deterministic Markov backend ----------------------------------------

type ContextBucket = {
  total: number;
  counts: Map<string, number>;
  tokens: Map<string, MovementActionToken>;
};

/**
 * Serializable model produced by {@link MarkovMovementBackend}. It stores, at
 * several backoff levels (most specific → global), how often each action
 * followed each context, so prediction can fall back from an exact context to a
 * coarser one it has seen.
 */
export type MarkovMovementModel = {
  version: 1;
  backend: "markov";
  smoothing: number;
  /** backoff level → context-key → bucket of action counts. */
  levels: Array<Record<string, ContextBucketData>>;
  exampleCount: number;
};

type ContextBucketData = {
  total: number;
  actions: Array<{ key: string; count: number; token: MovementActionToken }>;
};

/**
 * Backoff hierarchy, most specific first. Each entry projects a context down to
 * a coarser key so an unseen full context can still match a related one.
 */
const BACKOFF_PROJECTIONS: Array<(context: MovementContextFeatures) => string | undefined> = [
  (c) => keyOf({ app: c.app, lastTool: c.lastTool, lastGesture: c.lastGesture, lastTarget: c.lastTarget }),
  (c) => keyOf({ app: c.app, lastTarget: c.lastTarget }),
  (c) => keyOf({ app: c.app, lastTool: c.lastTool }),
  (c) => keyOf({ app: c.app }),
  (c) => keyOf({ lastTool: c.lastTool }),
  (c) => keyOf({ platform: c.platform }),
  () => "__global__",
];

function keyOf(parts: Record<string, string | undefined>): string | undefined {
  const entries = Object.entries(parts).filter(([, value]) => value !== undefined && value !== "");
  if (entries.length === 0) {
    return undefined;
  }
  entries.sort(([a], [b]) => a.localeCompare(b));
  return entries.map(([field, value]) => `${field}=${value}`).join("|");
}

export class MarkovMovementBackend implements MovementModelBackend<MarkovMovementModel> {
  readonly name = "markov";

  train(dataset: MovementDataset, options: MovementTrainOptions = {}): MarkovMovementModel {
    const smoothing = options.smoothing ?? 0.5;
    const outcomeWeights = { ...DEFAULT_OUTCOME_WEIGHTS, ...(options.outcomeWeights ?? {}) };
    const levels: Array<Map<string, ContextBucket>> = BACKOFF_PROJECTIONS.map(() => new Map());
    let exampleCount = 0;

    for (const sequence of dataset.sequences) {
      const weight = sequence.outcome ? outcomeWeights[sequence.outcome] ?? 1 : 1;
      if (weight <= 0) {
        continue;
      }
      for (const example of sequence.examples) {
        exampleCount += 1;
        const actionKey = movementActionKey(example.action);
        BACKOFF_PROJECTIONS.forEach((project, level) => {
          const contextKey = project(example.context);
          if (contextKey === undefined) {
            return;
          }
          const bucket = getOrCreateBucket(levels[level]!, contextKey);
          bucket.total += weight;
          bucket.counts.set(actionKey, (bucket.counts.get(actionKey) ?? 0) + weight);
          if (!bucket.tokens.has(actionKey)) {
            bucket.tokens.set(actionKey, example.action);
          }
        });
      }
    }

    return {
      version: 1,
      backend: "markov",
      smoothing,
      levels: levels.map((level) => serializeLevel(level)),
      exampleCount,
    };
  }

  predict(model: MarkovMovementModel, context: MovementContextFeatures): MovementPrediction {
    for (let level = 0; level < BACKOFF_PROJECTIONS.length; level += 1) {
      const project = BACKOFF_PROJECTIONS[level]!;
      const contextKey = project(context);
      if (contextKey === undefined) {
        continue;
      }
      const bucket = model.levels[level]?.[contextKey];
      if (!bucket || bucket.actions.length === 0) {
        continue;
      }
      return distributionFrom(bucket, model.smoothing, level);
    }
    return { action: undefined, confidence: 0, backoffLevel: -1, distribution: [] };
  }

  serialize(model: MarkovMovementModel): string {
    return JSON.stringify(model);
  }

  deserialize(data: string): MarkovMovementModel {
    const parsed = JSON.parse(data) as MarkovMovementModel;
    if (parsed.backend !== "markov" || parsed.version !== 1) {
      throw new Error("unsupported movement model payload");
    }
    return parsed;
  }
}

function getOrCreateBucket(level: Map<string, ContextBucket>, key: string): ContextBucket {
  let bucket = level.get(key);
  if (!bucket) {
    bucket = { total: 0, counts: new Map(), tokens: new Map() };
    level.set(key, bucket);
  }
  return bucket;
}

function serializeLevel(level: Map<string, ContextBucket>): Record<string, ContextBucketData> {
  const record: Record<string, ContextBucketData> = {};
  for (const [key, bucket] of level) {
    record[key] = {
      total: bucket.total,
      actions: [...bucket.counts.entries()]
        .map(([actionKey, count]) => ({ key: actionKey, count, token: bucket.tokens.get(actionKey)! }))
        // Deterministic order: by count desc, then key asc — no reliance on Map insertion order across (de)serialization.
        .sort((a, b) => (b.count - a.count) || a.key.localeCompare(b.key)),
    };
  }
  return record;
}

function distributionFrom(bucket: ContextBucketData, smoothing: number, level: number): MovementPrediction {
  const denominator = bucket.total + smoothing * bucket.actions.length;
  const distribution = bucket.actions
    .map<MovementPredictionCandidate>((entry) => ({
      action: entry.token,
      probability: (entry.count + smoothing) / denominator,
    }))
    .sort((a, b) => (b.probability - a.probability) || movementActionKey(a.action).localeCompare(movementActionKey(b.action)));
  const best = distribution[0];
  return {
    action: best?.action,
    confidence: best?.probability ?? 0,
    backoffLevel: level,
    distribution,
  };
}

// --- Rollout (repeat / generalize a movement chain) -----------------------

export type MovementRolloutOptions = {
  maxSteps?: number;
  /** Stop early once a step's confidence drops below this floor. */
  minConfidence?: number;
  /** Stop early if the same action would repeat this many times in a row. */
  maxRepeats?: number;
};

export type MovementRolloutStep = {
  action: MovementActionToken;
  confidence: number;
  backoffLevel: number;
};

/**
 * Greedily generate a movement chain from a starting context. This is how the
 * subsystem "repeats the recorded movements": from a start context seen in
 * training, the highest-probability transitions reconstruct the recorded chain;
 * from a related-but-unseen start, backoff produces a plausible chain instead.
 */
export function rolloutMovements<TModel>(
  backend: MovementModelBackend<TModel>,
  model: TModel,
  start: MovementContextFeatures,
  options: MovementRolloutOptions = {},
): MovementRolloutStep[] {
  const maxSteps = options.maxSteps ?? 16;
  const minConfidence = options.minConfidence ?? 0;
  const maxRepeats = options.maxRepeats ?? 3;

  const steps: MovementRolloutStep[] = [];
  // Seed the chain head: an unprimed start context begins at MOVEMENT_START so
  // the first prediction matches the recorded first step, not the app aggregate.
  let context: MovementContextFeatures = {
    ...start,
    lastTarget: start.lastTarget ?? (start.lastTool ? undefined : MOVEMENT_START),
  };
  let repeatKey: string | undefined;
  let repeatCount = 0;

  for (let i = 0; i < maxSteps; i += 1) {
    const prediction = backend.predict(model, context);
    if (!prediction.action || prediction.action.tool === MOVEMENT_END_TOOL || prediction.confidence < minConfidence) {
      break;
    }
    const key = movementActionKey(prediction.action);
    if (key === repeatKey) {
      repeatCount += 1;
      if (repeatCount >= maxRepeats) {
        break;
      }
    } else {
      repeatKey = key;
      repeatCount = 1;
    }
    steps.push({
      action: prediction.action,
      confidence: prediction.confidence,
      backoffLevel: prediction.backoffLevel,
    });
    context = {
      ...context,
      lastTool: prediction.action.tool,
      lastGesture: prediction.action.gesture,
      lastTarget: sequencingSignal(prediction.action),
    };
  }

  return steps;
}

// --- Generalization eval harness ------------------------------------------

export type MovementEvalReport = {
  total: number;
  /** Fraction where the predicted tool matched the held-out action's tool. */
  toolAccuracy: number;
  /** Fraction where tool + gesture + direction all matched exactly. */
  exactAccuracy: number;
  /** Fraction of predictions that required backoff (level > 0) to fire. */
  generalizationRate: number;
  /** Fraction where the model produced no prediction at all. */
  abstentionRate: number;
  meanConfidence: number;
};

/**
 * Measure replay fidelity on held-out but related trajectories — the roadmap's
 * generalization eval harness. For each held-out transition, predict from its
 * context and score against the recorded action.
 */
export function evaluateGeneralization<TModel>(
  backend: MovementModelBackend<TModel>,
  model: TModel,
  heldOut: MovementSequence[],
): MovementEvalReport {
  let total = 0;
  let toolHits = 0;
  let exactHits = 0;
  let generalized = 0;
  let abstained = 0;
  let confidenceSum = 0;

  for (const sequence of heldOut) {
    for (const example of sequence.examples) {
      total += 1;
      const prediction = backend.predict(model, example.context);
      if (!prediction.action) {
        abstained += 1;
        continue;
      }
      confidenceSum += prediction.confidence;
      if (prediction.backoffLevel > 0) {
        generalized += 1;
      }
      if (prediction.action.tool === example.action.tool) {
        toolHits += 1;
      }
      if (
        prediction.action.tool === example.action.tool &&
        (prediction.action.gesture ?? "") === (example.action.gesture ?? "") &&
        (prediction.action.direction ?? "") === (example.action.direction ?? "")
      ) {
        exactHits += 1;
      }
    }
  }

  const predicted = total - abstained;
  return {
    total,
    toolAccuracy: total === 0 ? 0 : toolHits / total,
    exactAccuracy: total === 0 ? 0 : exactHits / total,
    generalizationRate: predicted === 0 ? 0 : generalized / predicted,
    abstentionRate: total === 0 ? 0 : abstained / total,
    meanConfidence: predicted === 0 ? 0 : confidenceSum / predicted,
  };
}

/** Split a dataset deterministically into train/held-out by trajectory index. */
export function splitMovementDataset(
  dataset: MovementDataset,
  holdOutEvery = 4,
): { train: MovementDataset; heldOut: MovementSequence[] } {
  const train: MovementSequence[] = [];
  const heldOut: MovementSequence[] = [];
  dataset.sequences.forEach((sequence, index) => {
    if (holdOutEvery > 0 && (index + 1) % holdOutEvery === 0) {
      heldOut.push(sequence);
    } else {
      train.push(sequence);
    }
  });
  return { train: { version: 1, sequences: train }, heldOut };
}
