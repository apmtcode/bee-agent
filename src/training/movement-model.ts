// Local-movement learning subsystem: pluggable, in-process training + inference.
//
// This module closes the loop for standing objective #2 (c)/(d): given recorded
// movement trajectories it (c) post-trains a small *local* model that repeats the
// recorded movements and (d) generalizes to new-but-related movements — all
// runnable in the cloud with no real OS access.
//
// The heavy backend (a real on-device small model via MLX/axolotl) is described
// by `LocalAppleSiliconTrainingRunner`; that runner emits an external launch
// script. This module provides the *pluggable interface* those backends share
// plus a **deterministic n-gram backend** that actually learns and infers
// in-process, so capture -> dataset -> train -> replay/generalize can be
// validated without a GPU or a user's machine. Swap `NgramMovementBackend` for a
// neural backend behind the same `MovementModelBackend` seam.

import type { ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

/** Sentinel tokens marking sequence boundaries in the learned model. */
export const MOVEMENT_START_TOKEN = "<start>";
export const MOVEMENT_END_TOKEN = "<end>";

const GRAM_SEP = "␟"; // unit separator; never appears in a normalized token

/** A single normalized movement (one mouse/keyboard/gesture action). */
export type MovementStep = {
  tool: string;
  action: string;
  target?: string;
  direction?: string;
  value?: string;
  /** Canonical, vocabulary-stable encoding used by the sequence model. */
  token: string;
};

/** An ordered movement performed within one trajectory. */
export type MovementSequence = {
  id: string;
  steps: MovementStep[];
};

/** The replayable dataset a movement backend trains on. */
export type MovementDataset = {
  sequences: MovementSequence[];
  vocabulary: string[];
};

export type MovementTrainingConfig = {
  /** Max n-gram context length (>= 1). Higher = more literal recall, less generalization. */
  order: number;
};

/** One learned context table: context-key -> (next-token -> count). */
export type MovementGramTable = {
  order: number;
  entries: Record<string, Record<string, number>>;
};

/** Serializable trained model. Backend-tagged so the loader can reject mismatches. */
export type MovementModel = {
  backend: string;
  order: number;
  vocabulary: string[];
  grams: MovementGramTable[];
  sequenceCount: number;
  stepCount: number;
};

export type MovementPrediction = {
  token: string;
  /** Estimated probability at the context length that produced the pick, in [0,1]. */
  confidence: number;
  /** Context length actually used after stupid-backoff (<= requested order). */
  backoffOrder: number;
  /** Normalized next-token distribution at that context length, sorted desc. */
  distribution: Array<{ token: string; probability: number }>;
};

/**
 * The pluggable backend seam. A neural on-device backend implements the same
 * three members and can be dropped in without touching the pipeline/eval code.
 */
export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, config: MovementTrainingConfig): Promise<MovementModel>;
  predict(model: MovementModel, context: readonly string[]): MovementPrediction;
}

// ---------------------------------------------------------------------------
// Normalization: capture actions -> canonical movement steps
// ---------------------------------------------------------------------------

type MovementActionLike = {
  tool: string;
  summary: string;
  metadata?: Record<string, unknown>;
};

function readString(metadata: Record<string, unknown> | undefined, key: string): string | undefined {
  const value = metadata?.[key];
  return typeof value === "string" && value.trim().length > 0 ? value.trim() : undefined;
}

function slug(value: string): string {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 48);
}

/**
 * Derive a canonical `MovementStep` from any capture action. Handles the shapes
 * emitted by the device/browser/os adapters (`gesture`/`action` metadata) and
 * falls back to the human summary when metadata is absent.
 */
export function deriveMovementStep(action: MovementActionLike): MovementStep {
  const tool = slug(action.tool) || "tool";
  const rawAction =
    readString(action.metadata, "gesture") ??
    readString(action.metadata, "action") ??
    firstWord(action.summary);
  const step: MovementStep = {
    tool,
    action: slug(rawAction) || "act",
    token: "",
  };
  const target = readString(action.metadata, "target");
  const direction = readString(action.metadata, "direction");
  const value = readString(action.metadata, "valueSummary");
  if (target) {
    step.target = slug(target);
  }
  if (direction) {
    step.direction = slug(direction);
  }
  if (value) {
    step.value = slug(value);
  }
  step.token = encodeMovementToken(step);
  return step;
}

function firstWord(summary: string): string {
  const match = summary.trim().split(/\s+/)[0];
  return match ?? "act";
}

/** Stable token encoding. Target/direction discriminate otherwise-identical actions. */
export function encodeMovementToken(step: Pick<MovementStep, "tool" | "action" | "target" | "direction">): string {
  const parts = [`${step.tool}.${step.action}`];
  if (step.target) {
    parts.push(`@${step.target}`);
  }
  if (step.direction) {
    parts.push(`^${step.direction}`);
  }
  return parts.join("");
}

// ---------------------------------------------------------------------------
// Dataset construction
// ---------------------------------------------------------------------------

/** Build one movement sequence per trajectory from its ordered actions. */
export function buildMovementSequences(
  trajectories: ReadonlyArray<Pick<TrajectorySpan, "id" | "actions">>,
): MovementSequence[] {
  return trajectories.map((trajectory) => ({
    id: trajectory.id,
    steps: [...trajectory.actions]
      .sort((a: TrajectoryAction, b: TrajectoryAction) => a.ts - b.ts)
      .map((action) => deriveMovementStep(action)),
  }));
}

/** Build movement sequences from replay-manifest action events, grouped per trajectory. */
export function buildMovementSequencesFromReplay(events: ReadonlyArray<ReplayTimelineEvent>): MovementSequence[] {
  const byTrajectory = new Map<string, ReplayTimelineEvent[]>();
  for (const event of events) {
    if (event.kind !== "action") {
      continue;
    }
    const bucket = byTrajectory.get(event.trajectoryId) ?? [];
    bucket.push(event);
    byTrajectory.set(event.trajectoryId, bucket);
  }
  return [...byTrajectory.entries()].map(([id, actionEvents]) => ({
    id,
    steps: actionEvents
      .sort((a, b) => a.ts - b.ts)
      .map((event) => deriveMovementStep({ tool: event.kind === "action" ? event.tool : "tool", summary: event.kind === "action" ? event.summary : "" })),
  }));
}

/** Assemble a dataset with a stable, sorted vocabulary. */
export function buildMovementDataset(sequences: ReadonlyArray<MovementSequence>): MovementDataset {
  const vocabulary = new Set<string>();
  for (const sequence of sequences) {
    for (const step of sequence.steps) {
      vocabulary.add(step.token);
    }
  }
  return {
    sequences: sequences.map((sequence) => ({ id: sequence.id, steps: [...sequence.steps] })),
    vocabulary: [...vocabulary].sort(),
  };
}

// ---------------------------------------------------------------------------
// Deterministic n-gram backend (stupid-backoff) — trains + generalizes locally
// ---------------------------------------------------------------------------

/** Discount applied per level of backoff, mirroring the stupid-backoff heuristic. */
const BACKOFF_DISCOUNT = 0.4;

export class NgramMovementBackend implements MovementModelBackend {
  readonly name = "ngram-movement";

  async train(dataset: MovementDataset, config: MovementTrainingConfig): Promise<MovementModel> {
    const order = Math.max(1, Math.floor(config.order));
    const grams: MovementGramTable[] = Array.from({ length: order + 1 }, (_unused, level) => ({
      order: level,
      entries: {},
    }));
    let stepCount = 0;

    for (const sequence of dataset.sequences) {
      const tokens = [MOVEMENT_START_TOKEN, ...sequence.steps.map((step) => step.token), MOVEMENT_END_TOKEN];
      stepCount += sequence.steps.length;
      for (let index = 1; index < tokens.length; index += 1) {
        const next = tokens[index];
        for (let level = 0; level <= order; level += 1) {
          if (index - level < 0) {
            break;
          }
          const context = tokens.slice(index - level, index);
          const key = context.join(GRAM_SEP);
          const table = grams[level].entries;
          const row = (table[key] ??= {});
          row[next] = (row[next] ?? 0) + 1;
        }
      }
    }

    return {
      backend: this.name,
      order,
      vocabulary: [...dataset.vocabulary],
      grams,
      sequenceCount: dataset.sequences.length,
      stepCount,
    };
  }

  predict(model: MovementModel, context: readonly string[]): MovementPrediction {
    const maxLevel = Math.min(model.order, context.length);
    for (let level = maxLevel; level >= 0; level -= 1) {
      const key = context.slice(context.length - level).join(GRAM_SEP);
      const table = model.grams[level]?.entries[key];
      if (!table) {
        continue;
      }
      const distribution = normalizeDistribution(table);
      if (distribution.length === 0) {
        continue;
      }
      const backoffSteps = maxLevel - level;
      const top = distribution[0];
      return {
        token: top.token,
        confidence: top.probability * BACKOFF_DISCOUNT ** backoffSteps,
        backoffOrder: level,
        distribution,
      };
    }
    return { token: MOVEMENT_END_TOKEN, confidence: 0, backoffOrder: 0, distribution: [] };
  }
}

/** Deterministic normalization: probability desc, then token asc for stable ties. */
function normalizeDistribution(counts: Record<string, number>): Array<{ token: string; probability: number }> {
  const total = Object.values(counts).reduce((sum, value) => sum + value, 0);
  if (total === 0) {
    return [];
  }
  return Object.entries(counts)
    .map(([token, count]) => ({ token, probability: count / total }))
    .sort((a, b) => (b.probability !== a.probability ? b.probability - a.probability : a.token < b.token ? -1 : 1));
}

// ---------------------------------------------------------------------------
// Inference: rollout + generalization eval
// ---------------------------------------------------------------------------

export type MovementRolloutStep = {
  token: string;
  confidence: number;
  backoffOrder: number;
};

/**
 * Roll out a predicted movement from a seed prefix. Seeding with a recorded
 * prefix *repeats* the learned movement; seeding with a related-but-unseen
 * prefix exercises backoff to *generalize*. Stops at `<end>` or `maxSteps`.
 */
export function generateMovement(
  backend: MovementModelBackend,
  model: MovementModel,
  seed: readonly string[] = [],
  maxSteps = 32,
): MovementRolloutStep[] {
  const history = [MOVEMENT_START_TOKEN, ...seed];
  const rollout: MovementRolloutStep[] = seed.map((token) => ({ token, confidence: 1, backoffOrder: 0 }));
  for (let index = 0; index < maxSteps; index += 1) {
    const prediction = backend.predict(model, history);
    if (prediction.token === MOVEMENT_END_TOKEN || prediction.distribution.length === 0) {
      break;
    }
    rollout.push({
      token: prediction.token,
      confidence: prediction.confidence,
      backoffOrder: prediction.backoffOrder,
    });
    history.push(prediction.token);
  }
  return rollout;
}

export type MovementEvalResult = {
  /** Positions scored (one per non-first token across all held-out sequences). */
  predictions: number;
  correct: number;
  /** Fraction of next-step predictions matching the held-out ground truth. */
  accuracy: number;
  /** Share of predictions that required backing off below the full order. */
  backoffRate: number;
};

/**
 * Teacher-forced next-step accuracy on held-out sequences — the generalization
 * metric. Feed sequences the model never trained on to measure whether it
 * predicts related movements, not just memorized ones.
 */
export function evaluateMovementModel(
  backend: MovementModelBackend,
  model: MovementModel,
  heldOut: ReadonlyArray<MovementSequence>,
): MovementEvalResult {
  let predictions = 0;
  let correct = 0;
  let backedOff = 0;
  for (const sequence of heldOut) {
    const tokens = [MOVEMENT_START_TOKEN, ...sequence.steps.map((step) => step.token), MOVEMENT_END_TOKEN];
    for (let index = 1; index < tokens.length; index += 1) {
      const context = tokens.slice(0, index);
      const prediction = backend.predict(model, context);
      predictions += 1;
      if (prediction.token === tokens[index]) {
        correct += 1;
      }
      if (prediction.backoffOrder < Math.min(model.order, context.length)) {
        backedOff += 1;
      }
    }
  }
  return {
    predictions,
    correct,
    accuracy: predictions === 0 ? 0 : correct / predictions,
    backoffRate: predictions === 0 ? 0 : backedOff / predictions,
  };
}

// ---------------------------------------------------------------------------
// Backend registry + one-shot training pipeline
// ---------------------------------------------------------------------------

export type MovementBackendName = "ngram-movement";

/** Resolve a backend by name. Extend as neural backends land behind the same seam. */
export function createMovementBackend(name: MovementBackendName = "ngram-movement"): MovementModelBackend {
  switch (name) {
    case "ngram-movement":
      return new NgramMovementBackend();
    default: {
      const exhaustive: never = name;
      throw new Error(`unknown movement backend: ${String(exhaustive)}`);
    }
  }
}

export type MovementTrainingResult = {
  model: MovementModel;
  dataset: MovementDataset;
  /** In-sample fit — how well the model reproduces its own training movements. */
  trainEval: MovementEvalResult;
};

/**
 * End-to-end: normalize trajectories -> dataset -> train -> self-eval. Pure and
 * deterministic, so cloud/CI runs validate the full learning loop.
 */
export async function trainMovementModel(
  trajectories: ReadonlyArray<Pick<TrajectorySpan, "id" | "actions">>,
  options: { backend?: MovementModelBackend; config?: MovementTrainingConfig } = {},
): Promise<MovementTrainingResult> {
  const backend = options.backend ?? createMovementBackend();
  const config = options.config ?? { order: 2 };
  const dataset = buildMovementDataset(buildMovementSequences(trajectories));
  const model = await backend.train(dataset, config);
  const trainEval = evaluateMovementModel(backend, model, dataset.sequences);
  return { model, dataset, trainEval };
}

// ---------------------------------------------------------------------------
// Synthetic event-stream generator (deterministic, no OS / no RNG globals)
// ---------------------------------------------------------------------------

/** Tiny deterministic LCG so synthetic data is reproducible without Math.random. */
function lcg(seed: number): () => number {
  let state = (seed >>> 0) || 1;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

export type SyntheticMovementOptions = {
  seed?: number;
  sequenceCount?: number;
  /** Ordered pools of action tokens; each sequence walks one pool with light noise. */
  patterns?: string[][];
  minSteps?: number;
  maxSteps?: number;
};

const DEFAULT_PATTERNS: string[][] = [
  ["os.focus@editor", "browser.click@search", "browser.type@search", "browser.submit@search"],
  ["device.tap@compose", "device.type@body", "device.tap@send"],
  ["os.focus@terminal", "os.command@build", "os.command@test"],
];

/**
 * Generate reproducible synthetic movement sequences that mimic captured
 * trajectories — used to validate capture->dataset->train->replay round-trips
 * and to build held-out generalization sets without any real OS input.
 */
export function generateSyntheticMovementSequences(options: SyntheticMovementOptions = {}): MovementSequence[] {
  const random = lcg(options.seed ?? 1);
  const patterns = options.patterns ?? DEFAULT_PATTERNS;
  const sequenceCount = Math.max(1, options.sequenceCount ?? 12);
  const minSteps = Math.max(1, options.minSteps ?? 2);
  const maxSteps = Math.max(minSteps, options.maxSteps ?? 6);
  const sequences: MovementSequence[] = [];

  for (let index = 0; index < sequenceCount; index += 1) {
    const pattern = patterns[Math.floor(random() * patterns.length)] ?? patterns[0];
    const length = minSteps + Math.floor(random() * (maxSteps - minSteps + 1));
    const steps: MovementStep[] = [];
    for (let position = 0; position < length; position += 1) {
      // Walk the pattern in order, occasionally repeating a step (dwell noise).
      const patternIndex = Math.min(pattern.length - 1, position);
      const token = pattern[patternIndex] ?? pattern[0];
      steps.push(decodeSyntheticToken(token));
    }
    sequences.push({ id: `synthetic-${options.seed ?? 1}-${index}`, steps });
  }
  return sequences;
}

function decodeSyntheticToken(token: string): MovementStep {
  const match = /^([^.]+)\.([^@^]+)(?:@([^^]+))?(?:\^(.+))?$/.exec(token);
  if (!match) {
    return { tool: "tool", action: "act", token };
  }
  const [, tool, action, target, direction] = match;
  const step: MovementStep = { tool, action, token };
  if (target) {
    step.target = target;
  }
  if (direction) {
    step.direction = direction;
  }
  step.token = encodeMovementToken(step);
  return step;
}
