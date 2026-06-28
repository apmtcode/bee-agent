/**
 * In-process movement-policy model for the local-movement learning subsystem.
 *
 * Objective #2(c)/(d): post-train a *local* model on a recorded movement
 * dataset so it can (a) repeat the recorded movements and (b) generalize to new
 * but related movements. The reference codebases delegate training to external
 * MLX/axolotl launch scripts that cannot run in the cloud, so none of that logic
 * is testable here. This module provides a small, fully deterministic,
 * dependency-free learner that runs in-process — both as a working capability
 * and as a CI-safe stand-in that validates the capture -> dataset -> train ->
 * infer pipeline end to end. The backend is pluggable so a real on-device small
 * model can be slotted in behind the same interface.
 */

import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

/** A canonical, comparable token describing a single movement/action. */
export type MovementToken = string;

/** Sentinel tokens used to model sequence boundaries during training. */
export const MOVEMENT_START_TOKEN: MovementToken = "<start>";
export const MOVEMENT_END_TOKEN: MovementToken = "<end>";

/** A tokenized movement dataset ready for training. */
export type MovementDataset = {
  version: 1;
  /** Sorted, de-duplicated vocabulary (excludes the start sentinel). */
  vocabulary: MovementToken[];
  /** One token sequence per source trajectory, in capture order. */
  sequences: MovementToken[][];
  sequenceCount: number;
  tokenCount: number;
};

export type MovementTrainingConfig = {
  /** Maximum history length (context tokens) the model conditions on. */
  order: number;
  /** Add-k smoothing applied uniformly over the vocabulary. */
  smoothing: number;
};

export const DEFAULT_MOVEMENT_TRAINING_CONFIG: MovementTrainingConfig = {
  order: 3,
  smoothing: 0,
};

export type MovementPredictionCandidate = {
  token: MovementToken;
  /** Normalized probability among the candidates at the chosen backoff order. */
  probability: number;
};

export type MovementPrediction = {
  token: MovementToken;
  probability: number;
  /** How many context tokens were actually usable (after backoff). */
  backoffOrder: number;
  candidates: MovementPredictionCandidate[];
};

/** Serializable form of a trained model — replayable and persistable as JSON. */
export type MovementModelSnapshot = {
  version: 1;
  backendId: string;
  order: number;
  smoothing: number;
  vocabulary: MovementToken[];
  /** context-string -> { token -> count }. Empty context key is "". */
  counts: Record<string, Record<MovementToken, number>>;
};

export interface MovementModel {
  readonly backendId: string;
  readonly order: number;
  /** Rank the next-token candidates given the prior context. */
  predictNext(context: MovementToken[]): MovementPrediction;
  /** Roll out up to `steps` tokens from a seed, stopping at the end sentinel. */
  generate(seed: MovementToken[], steps: number): MovementToken[];
  serialize(): MovementModelSnapshot;
}

export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, config?: Partial<MovementTrainingConfig>): MovementModel;
}

// ---------------------------------------------------------------------------
// Tokenization: trajectory actions / replay events -> movement tokens
// ---------------------------------------------------------------------------

function slug(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 48);
}

/**
 * Derive a stable token for an action. Metadata fields that describe the
 * *movement* (gesture/target/direction/key) are preferred over the free-text
 * summary so semantically identical movements collapse to the same token,
 * which is what lets the model generalize.
 */
export function tokenizeMovementAction(action: Pick<TrajectoryAction, "tool" | "summary" | "metadata">): MovementToken {
  const tool = slug(action.tool) || "tool";
  const meta = action.metadata ?? {};
  const parts: string[] = [];
  for (const key of ["gesture", "key", "button", "target", "direction"]) {
    const raw = meta[key];
    if (typeof raw === "string" && raw.trim().length > 0) {
      parts.push(slug(raw));
    }
  }
  const detail = parts.length > 0 ? parts.join(".") : slug(action.summary) || "action";
  return `${tool}:${detail}`;
}

function tokenizeReplayActionEvent(event: Extract<ReplayTimelineEvent, { kind: "action" }>): MovementToken {
  return tokenizeMovementAction({ tool: event.tool, summary: event.summary });
}

// ---------------------------------------------------------------------------
// Dataset construction
// ---------------------------------------------------------------------------

function buildDatasetFromSequences(sequences: MovementToken[][]): MovementDataset {
  const nonEmpty = sequences.filter((sequence) => sequence.length > 0);
  const vocabulary = [...new Set(nonEmpty.flat())].sort();
  return {
    version: 1,
    vocabulary,
    sequences: nonEmpty,
    sequenceCount: nonEmpty.length,
    tokenCount: nonEmpty.reduce((total, sequence) => total + sequence.length, 0),
  };
}

/** Build a dataset from trajectory spans (actions in capture order per span). */
export function buildMovementDatasetFromTrajectories(trajectories: TrajectorySpan[]): MovementDataset {
  const sequences = trajectories.map((trajectory) =>
    [...trajectory.actions]
      .sort((a, b) => a.ts - b.ts)
      .map((action) => tokenizeMovementAction(action)),
  );
  return buildDatasetFromSequences(sequences);
}

/** Build a dataset from one or more replay manifests (one sequence per trajectory). */
export function buildMovementDatasetFromReplays(replays: ReplayManifest[]): MovementDataset {
  const byTrajectory = new Map<string, { ts: number; token: MovementToken }[]>();
  for (const replay of replays) {
    for (const event of replay.events) {
      if (event.kind !== "action") {
        continue;
      }
      const key = `${replay.sessionId}:${event.trajectoryId}`;
      const bucket = byTrajectory.get(key) ?? [];
      bucket.push({ ts: event.ts, token: tokenizeReplayActionEvent(event) });
      byTrajectory.set(key, bucket);
    }
  }
  const sequences = [...byTrajectory.values()].map((bucket) =>
    bucket.sort((a, b) => a.ts - b.ts).map((entry) => entry.token),
  );
  return buildDatasetFromSequences(sequences);
}

// ---------------------------------------------------------------------------
// N-gram backoff backend — deterministic, dependency-free
// ---------------------------------------------------------------------------

const CONTEXT_DELIMITER = "";

function contextKey(context: MovementToken[]): string {
  return context.join(CONTEXT_DELIMITER);
}

class NgramMovementModel implements MovementModel {
  readonly backendId: string;
  readonly order: number;
  private readonly smoothing: number;
  private readonly vocabulary: MovementToken[];
  private readonly counts: Map<string, Map<MovementToken, number>>;

  constructor(params: {
    backendId: string;
    order: number;
    smoothing: number;
    vocabulary: MovementToken[];
    counts: Map<string, Map<MovementToken, number>>;
  }) {
    this.backendId = params.backendId;
    this.order = params.order;
    this.smoothing = params.smoothing;
    this.vocabulary = params.vocabulary;
    this.counts = params.counts;
  }

  predictNext(context: MovementToken[]): MovementPrediction {
    for (let length = Math.min(this.order, context.length); length >= 0; length -= 1) {
      const key = contextKey(context.slice(context.length - length));
      const bucket = this.counts.get(key);
      if (!bucket || bucket.size === 0) {
        continue;
      }
      return this.rank(bucket, length);
    }
    // No observed context at all: fall back to the end sentinel, which keeps
    // generation terminating rather than emitting an arbitrary token.
    return {
      token: MOVEMENT_END_TOKEN,
      probability: 1,
      backoffOrder: 0,
      candidates: [{ token: MOVEMENT_END_TOKEN, probability: 1 }],
    };
  }

  private rank(bucket: Map<MovementToken, number>, backoffOrder: number): MovementPrediction {
    // Candidate set: observed tokens, plus the full vocabulary when smoothing
    // is enabled (so unseen-but-related tokens get non-zero mass).
    const candidateTokens = new Set<MovementToken>(bucket.keys());
    candidateTokens.add(MOVEMENT_END_TOKEN);
    if (this.smoothing > 0) {
      for (const token of this.vocabulary) {
        candidateTokens.add(token);
      }
    }
    const denominator =
      [...bucket.values()].reduce((total, count) => total + count, 0) + this.smoothing * candidateTokens.size;
    const candidates = [...candidateTokens]
      .map((token) => ({
        token,
        probability: ((bucket.get(token) ?? 0) + this.smoothing) / denominator,
      }))
      .filter((candidate) => candidate.probability > 0)
      .sort((a, b) => (b.probability !== a.probability ? b.probability - a.probability : a.token < b.token ? -1 : 1));
    const best = candidates[0] ?? { token: MOVEMENT_END_TOKEN, probability: 1 };
    return { token: best.token, probability: best.probability, backoffOrder, candidates };
  }

  generate(seed: MovementToken[], steps: number): MovementToken[] {
    const history = [...seed];
    const produced: MovementToken[] = [];
    for (let step = 0; step < steps; step += 1) {
      const prediction = this.predictNext(history);
      if (prediction.token === MOVEMENT_END_TOKEN) {
        break;
      }
      produced.push(prediction.token);
      history.push(prediction.token);
    }
    return produced;
  }

  serialize(): MovementModelSnapshot {
    const counts: Record<string, Record<MovementToken, number>> = {};
    for (const [key, bucket] of this.counts) {
      counts[key] = Object.fromEntries(bucket);
    }
    return {
      version: 1,
      backendId: this.backendId,
      order: this.order,
      smoothing: this.smoothing,
      vocabulary: [...this.vocabulary],
      counts,
    };
  }
}

export class NgramMovementBackend implements MovementModelBackend {
  readonly id = "ngram-backoff";

  train(dataset: MovementDataset, config: Partial<MovementTrainingConfig> = {}): MovementModel {
    const order = Math.max(0, Math.trunc(config.order ?? DEFAULT_MOVEMENT_TRAINING_CONFIG.order));
    const smoothing = Math.max(0, config.smoothing ?? DEFAULT_MOVEMENT_TRAINING_CONFIG.smoothing);
    const counts = new Map<string, Map<MovementToken, number>>();

    const bump = (context: MovementToken[], next: MovementToken): void => {
      const key = contextKey(context);
      const bucket = counts.get(key) ?? new Map<MovementToken, number>();
      bucket.set(next, (bucket.get(next) ?? 0) + 1);
      counts.set(key, bucket);
    };

    for (const sequence of dataset.sequences) {
      const padded = [
        ...Array.from({ length: order }, () => MOVEMENT_START_TOKEN),
        ...sequence,
        MOVEMENT_END_TOKEN,
      ];
      for (let index = order; index < padded.length; index += 1) {
        const next = padded[index] as MovementToken;
        for (let length = 0; length <= order; length += 1) {
          bump(padded.slice(index - length, index), next);
        }
      }
    }

    return new NgramMovementModel({ backendId: this.id, order, smoothing, vocabulary: dataset.vocabulary, counts });
  }
}

export function deserializeMovementModel(snapshot: MovementModelSnapshot): MovementModel {
  const counts = new Map<string, Map<MovementToken, number>>();
  for (const [key, bucket] of Object.entries(snapshot.counts)) {
    counts.set(key, new Map(Object.entries(bucket)));
  }
  return new NgramMovementModel({
    backendId: snapshot.backendId,
    order: snapshot.order,
    smoothing: snapshot.smoothing,
    vocabulary: [...snapshot.vocabulary],
    counts,
  });
}

/** Convenience: train the default backend in one call. */
export function trainMovementModel(
  dataset: MovementDataset,
  config?: Partial<MovementTrainingConfig>,
): MovementModel {
  return new NgramMovementBackend().train(dataset, config);
}

// ---------------------------------------------------------------------------
// Generalization / fidelity eval harness
// ---------------------------------------------------------------------------

export type MovementEvalResult = {
  sequenceCount: number;
  /** Teacher-forced next-token accuracy across all positions. */
  nextTokenAccuracy: number;
  /** Mean longest-common-prefix ratio of free rollouts vs. ground truth. */
  rolloutFidelity: number;
  /** Fraction of held-out sequences reproduced exactly by free rollout. */
  exactRolloutRate: number;
};

function longestCommonPrefix(a: MovementToken[], b: MovementToken[]): number {
  const limit = Math.min(a.length, b.length);
  let index = 0;
  while (index < limit && a[index] === b[index]) {
    index += 1;
  }
  return index;
}

/**
 * Measure how well a trained model reproduces and generalizes to held-out
 * trajectories. `seedLength` controls how much of each sequence is given before
 * the model must free-roll the remainder (the generalization test).
 */
export function evaluateMovementModel(
  model: MovementModel,
  heldOut: MovementToken[][],
  options: { seedLength?: number } = {},
): MovementEvalResult {
  const sequences = heldOut.filter((sequence) => sequence.length > 0);
  if (sequences.length === 0) {
    return { sequenceCount: 0, nextTokenAccuracy: 0, rolloutFidelity: 0, exactRolloutRate: 0 };
  }
  const seedLength = Math.max(0, options.seedLength ?? 1);

  let correctTokens = 0;
  let totalTokens = 0;
  let fidelitySum = 0;
  let exactCount = 0;

  for (const sequence of sequences) {
    const padded = [...Array.from({ length: model.order }, () => MOVEMENT_START_TOKEN), ...sequence];
    for (let index = model.order; index < padded.length; index += 1) {
      const context = padded.slice(0, index);
      const prediction = model.predictNext(context);
      if (prediction.token === padded[index]) {
        correctTokens += 1;
      }
      totalTokens += 1;
    }

    const seed = sequence.slice(0, Math.min(seedLength, sequence.length));
    const expectedTail = sequence.slice(seed.length);
    const rollout = model.generate(seed, expectedTail.length + model.order + 1);
    if (expectedTail.length === 0) {
      fidelitySum += rollout.length === 0 ? 1 : 0;
    } else {
      fidelitySum += longestCommonPrefix(rollout, expectedTail) / expectedTail.length;
    }
    if (rollout.length === expectedTail.length && longestCommonPrefix(rollout, expectedTail) === expectedTail.length) {
      exactCount += 1;
    }
  }

  return {
    sequenceCount: sequences.length,
    nextTokenAccuracy: totalTokens === 0 ? 0 : correctTokens / totalTokens,
    rolloutFidelity: fidelitySum / sequences.length,
    exactRolloutRate: exactCount / sequences.length,
  };
}
