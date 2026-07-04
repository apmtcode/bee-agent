/**
 * Pluggable local-movement model backend.
 *
 * This module closes objective #2 parts (c) "post-train a local model on the
 * dataset to repeat recorded movements" and (d) "generalize to new but related
 * movements" with code that runs fully in-process — no OS input, no external
 * training runtime — so it can be validated in the cloud with synthetic event
 * streams.
 *
 * The design is intentionally backend-agnostic:
 *   - `MovementModelBackend` is the training seam. A real on-device backend
 *     (MLX / llama.cpp / a small transformer) implements the same interface;
 *     `NgramMovementBackend` is a deterministic, dependency-free mock that
 *     actually learns and generalizes so tests and CI have something real to
 *     assert against.
 *   - Movements are tokenized from the existing capture schema
 *     (`TrajectoryAction` / `ReplayTimelineEvent`) into a canonical token
 *     stream, so the model consumes exactly what the recorder produces.
 *
 * Everything here is deterministic (no `Date.now`/`Math.random`) so training,
 * inference, and the generalization eval reproduce byte-for-byte.
 */

import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";
import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";

/** Canonical movement token, e.g. `device/tap/save-button`. */
export type MovementToken = string;

/** Sentinels wrapping every training sequence so the model can learn how
 * sequences begin and end (and so `generate` knows when to stop). */
export const MOVEMENT_START: MovementToken = "<mov:start>";
export const MOVEMENT_END: MovementToken = "<mov:end>";

/** One recorded (or synthetic) movement trajectory as an ordered token list. */
export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

export type MovementTrainingDataset = {
  sequences: MovementSequence[];
};

// ---------------------------------------------------------------------------
// Tokenization — turn the capture schema into a canonical token stream.
// ---------------------------------------------------------------------------

function slug(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 48) || "none";
}

function firstString(...values: unknown[]): string | undefined {
  for (const value of values) {
    if (typeof value === "string" && value.trim().length > 0) {
      return value;
    }
  }
  return undefined;
}

/**
 * Derive a canonical token from a recorded action. Prefers structured
 * metadata (gesture/target/direction) and falls back to a slug of the summary,
 * so tokens stay stable across recordings of the same movement.
 */
export function movementTokenFromAction(action: TrajectoryAction): MovementToken {
  const metadata = action.metadata ?? {};
  const verb = firstString(metadata.gesture, metadata.event, metadata.action) ?? "act";
  const object = firstString(
    metadata.target,
    metadata.direction,
    metadata.filePath,
    metadata.windowTitle,
    metadata.valueSummary,
  );
  const descriptor = object ? slug(object) : slug(action.summary);
  return `${slug(action.tool)}/${slug(verb)}/${descriptor}`;
}

/** Token for a replay-timeline action event (which carries no metadata). */
export function movementTokenFromReplayEvent(event: ReplayTimelineEvent): MovementToken | undefined {
  if (event.kind !== "action") {
    return undefined;
  }
  return `${slug(event.tool)}/act/${slug(event.summary)}`;
}

/** Build a movement sequence from a single trajectory's actions (time-ordered). */
export function sequenceFromTrajectory(trajectory: TrajectorySpan): MovementSequence {
  const tokens = [...trajectory.actions]
    .sort((a, b) => a.ts - b.ts)
    .map((action) => movementTokenFromAction(action));
  return { id: trajectory.id, tokens };
}

export function datasetFromTrajectories(trajectories: TrajectorySpan[]): MovementTrainingDataset {
  return {
    sequences: trajectories
      .map((trajectory) => sequenceFromTrajectory(trajectory))
      .filter((sequence) => sequence.tokens.length > 0),
  };
}

export function datasetFromReplayManifests(manifests: ReplayManifest[]): MovementTrainingDataset {
  const sequences: MovementSequence[] = [];
  for (const manifest of manifests) {
    const byTrajectory = new Map<string, { ts: number; token: MovementToken }[]>();
    for (const event of manifest.events) {
      if (event.kind !== "action") {
        continue;
      }
      const token = movementTokenFromReplayEvent(event);
      if (!token) {
        continue;
      }
      const bucket = byTrajectory.get(event.trajectoryId) ?? [];
      bucket.push({ ts: event.ts, token });
      byTrajectory.set(event.trajectoryId, bucket);
    }
    for (const [trajectoryId, entries] of byTrajectory) {
      const tokens = entries.sort((a, b) => a.ts - b.ts).map((entry) => entry.token);
      if (tokens.length > 0) {
        sequences.push({ id: `${manifest.sessionId}:${trajectoryId}`, tokens });
      }
    }
  }
  return { sequences };
}

// ---------------------------------------------------------------------------
// Model interface — the pluggable seam.
// ---------------------------------------------------------------------------

export type MovementCandidate = {
  token: MovementToken;
  probability: number;
};

export type MovementPrediction = {
  token: MovementToken;
  probability: number;
  /** Context length (in tokens) that actually produced the prediction — lower
   * values mean the model backed off to a shorter, more general context. */
  order: number;
  candidates: MovementCandidate[];
};

export interface TrainedMovementModel {
  readonly backend: string;
  readonly order: number;
  /** Most-likely next token given a context, or undefined if the model is empty. */
  predictNext(context: MovementToken[]): MovementPrediction | undefined;
  /** Roll the policy forward from a seed until `<mov:end>` or `maxSteps`. */
  generate(seed: MovementToken[], maxSteps: number): MovementToken[];
  /** Mean per-token log-probability of a sequence under the model (higher =
   * more plausible). Used by the generalization eval. */
  scoreSequence(tokens: MovementToken[]): number;
  snapshot(): MovementModelSnapshot;
}

export type MovementTrainOptions = {
  /** Maximum context length used for prediction. Default 2 (a trigram model). */
  order?: number;
};

export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementTrainingDataset, options?: MovementTrainOptions): Promise<TrainedMovementModel>;
}

// ---------------------------------------------------------------------------
// Serialization — persist/reload a trained model without the training data.
// ---------------------------------------------------------------------------

export type MovementModelSnapshot = {
  version: 1;
  backend: string;
  order: number;
  /** counts[o] maps a joined context of length `o` to token→count. */
  counts: Record<number, Record<string, Record<MovementToken, number>>>;
};

const CONTEXT_SEP = "";

function contextKey(tokens: MovementToken[]): string {
  return tokens.join(CONTEXT_SEP);
}

// ---------------------------------------------------------------------------
// NgramMovementBackend — deterministic Markov model with stupid backoff.
// ---------------------------------------------------------------------------

const BACKOFF_ALPHA = 0.4;
const UNSEEN_FLOOR = 1e-6;

class NgramMovementModel implements TrainedMovementModel {
  readonly backend = "ngram";

  constructor(
    readonly order: number,
    /** counts[o]: contextKey -> (token -> count) for contexts of length o. */
    private readonly counts: Map<number, Map<string, Map<MovementToken, number>>>,
  ) {}

  private distributionAt(order: number, context: MovementToken[]): Map<MovementToken, number> | undefined {
    const level = this.counts.get(order);
    if (!level) {
      return undefined;
    }
    const suffix = order === 0 ? [] : context.slice(context.length - order);
    if (suffix.length < order) {
      return undefined;
    }
    return level.get(contextKey(suffix));
  }

  predictNext(context: MovementToken[]): MovementPrediction | undefined {
    const maxOrder = Math.min(this.order, context.length);
    for (let order = maxOrder; order >= 0; order--) {
      const distribution = this.distributionAt(order, context);
      if (!distribution || distribution.size === 0) {
        continue;
      }
      const total = [...distribution.values()].reduce((sum, count) => sum + count, 0);
      const candidates: MovementCandidate[] = [...distribution.entries()]
        .map(([token, count]) => ({ token, probability: count / total }))
        // deterministic ordering: higher probability first, then lexicographic.
        .sort((a, b) => (b.probability - a.probability) || (a.token < b.token ? -1 : a.token > b.token ? 1 : 0));
      const best = candidates[0];
      if (best) {
        return { token: best.token, probability: best.probability, order, candidates };
      }
    }
    return undefined;
  }

  generate(seed: MovementToken[], maxSteps: number): MovementToken[] {
    const context = [MOVEMENT_START, ...seed];
    const produced: MovementToken[] = [...seed];
    for (let step = 0; step < maxSteps; step++) {
      const prediction = this.predictNext(context);
      if (!prediction || prediction.token === MOVEMENT_END) {
        break;
      }
      produced.push(prediction.token);
      context.push(prediction.token);
    }
    return produced;
  }

  /** P(token | context) via stupid backoff, floored so unseen tokens don't -Inf. */
  private probability(context: MovementToken[], token: MovementToken): number {
    const maxOrder = Math.min(this.order, context.length);
    for (let order = maxOrder; order >= 0; order--) {
      const distribution = this.distributionAt(order, context);
      if (!distribution || distribution.size === 0) {
        continue;
      }
      const count = distribution.get(token);
      const total = [...distribution.values()].reduce((sum, value) => sum + value, 0);
      const weight = BACKOFF_ALPHA ** (maxOrder - order);
      if (count && count > 0) {
        return weight * (count / total);
      }
      // Token unseen at this order: keep backing off to a shorter context.
    }
    return UNSEEN_FLOOR;
  }

  scoreSequence(tokens: MovementToken[]): number {
    const stream = [MOVEMENT_START, ...tokens, MOVEMENT_END];
    let logProb = 0;
    let count = 0;
    for (let i = 1; i < stream.length; i++) {
      const probability = this.probability(stream.slice(0, i), stream[i]);
      logProb += Math.log(Math.max(probability, UNSEEN_FLOOR));
      count += 1;
    }
    return count === 0 ? 0 : logProb / count;
  }

  snapshot(): MovementModelSnapshot {
    const counts: MovementModelSnapshot["counts"] = {};
    for (const [order, contexts] of this.counts) {
      const levelRecord: Record<string, Record<MovementToken, number>> = {};
      for (const [key, tokens] of contexts) {
        const tokenRecord: Record<MovementToken, number> = {};
        for (const [token, value] of tokens) {
          tokenRecord[token] = value;
        }
        levelRecord[key] = tokenRecord;
      }
      counts[order] = levelRecord;
    }
    return { version: 1, backend: this.backend, order: this.order, counts };
  }
}

export class NgramMovementBackend implements MovementModelBackend {
  readonly name = "ngram";

  async train(
    dataset: MovementTrainingDataset,
    options?: MovementTrainOptions,
  ): Promise<TrainedMovementModel> {
    const order = Math.max(0, options?.order ?? 2);
    const counts = new Map<number, Map<string, Map<MovementToken, number>>>();
    for (let o = 0; o <= order; o++) {
      counts.set(o, new Map());
    }

    for (const sequence of dataset.sequences) {
      const stream = [MOVEMENT_START, ...sequence.tokens, MOVEMENT_END];
      for (let i = 1; i < stream.length; i++) {
        const token = stream[i];
        for (let o = 0; o <= order; o++) {
          if (i - o < 0) {
            continue;
          }
          const context = o === 0 ? [] : stream.slice(i - o, i);
          const level = counts.get(o)!;
          const key = contextKey(context);
          const distribution = level.get(key) ?? new Map<MovementToken, number>();
          distribution.set(token, (distribution.get(token) ?? 0) + 1);
          level.set(key, distribution);
        }
      }
    }

    return new NgramMovementModel(order, counts);
  }
}

/** Reconstruct a trained model from a snapshot (no training data required). */
export function loadMovementModel(snapshot: MovementModelSnapshot): TrainedMovementModel {
  const counts = new Map<number, Map<string, Map<MovementToken, number>>>();
  for (const [orderKey, contexts] of Object.entries(snapshot.counts)) {
    const order = Number(orderKey);
    const level = new Map<string, Map<MovementToken, number>>();
    for (const [key, tokens] of Object.entries(contexts)) {
      const distribution = new Map<MovementToken, number>();
      for (const [token, value] of Object.entries(tokens)) {
        distribution.set(token, value);
      }
      level.set(key, distribution);
    }
    counts.set(order, level);
  }
  return new NgramMovementModel(snapshot.order, counts);
}

// ---------------------------------------------------------------------------
// Backend registry — makes the model backend swappable at runtime.
// ---------------------------------------------------------------------------

export class MovementBackendRegistry {
  private readonly backends = new Map<string, MovementModelBackend>();

  constructor(backends: MovementModelBackend[] = [new NgramMovementBackend()]) {
    for (const backend of backends) {
      this.register(backend);
    }
  }

  register(backend: MovementModelBackend): void {
    this.backends.set(backend.name, backend);
  }

  get(name: string): MovementModelBackend | undefined {
    return this.backends.get(name);
  }

  list(): string[] {
    return [...this.backends.keys()].sort();
  }
}

// ---------------------------------------------------------------------------
// Generalization eval harness — measure replay fidelity on held-out data.
// ---------------------------------------------------------------------------

export type GeneralizationEvalResult = {
  sequenceCount: number;
  tokenCount: number;
  /** Fraction of next-token predictions (given the true prefix) that matched. */
  nextTokenAccuracy: number;
  /** Mean per-token log-probability the model assigns to the held-out data. */
  meanLogProb: number;
  perSequence: {
    id: string;
    tokenCount: number;
    nextTokenAccuracy: number;
    meanLogProb: number;
  }[];
};

/**
 * Evaluate how well a trained model generalizes to held-out (but related)
 * movement sequences. For each position the model predicts the next token from
 * the *true* prefix; accuracy is the match rate. This is the payoff for
 * objective #2(d): a model that only memorized would score near-zero on
 * novel-but-related sequences, while one that generalized scores high.
 */
export function evaluateGeneralization(
  model: TrainedMovementModel,
  heldOut: MovementSequence[],
): GeneralizationEvalResult {
  const perSequence: GeneralizationEvalResult["perSequence"] = [];
  let totalTokens = 0;
  let totalHits = 0;

  for (const sequence of heldOut) {
    const stream = [MOVEMENT_START, ...sequence.tokens, MOVEMENT_END];
    let hits = 0;
    let predictions = 0;
    for (let i = 1; i < stream.length; i++) {
      const prediction = model.predictNext(stream.slice(0, i));
      predictions += 1;
      if (prediction && prediction.token === stream[i]) {
        hits += 1;
      }
    }
    const accuracy = predictions === 0 ? 0 : hits / predictions;
    perSequence.push({
      id: sequence.id,
      tokenCount: sequence.tokens.length,
      nextTokenAccuracy: accuracy,
      meanLogProb: model.scoreSequence(sequence.tokens),
    });
    totalTokens += predictions;
    totalHits += hits;
  }

  const meanLogProb =
    perSequence.length === 0
      ? 0
      : perSequence.reduce((sum, entry) => sum + entry.meanLogProb, 0) / perSequence.length;

  return {
    sequenceCount: heldOut.length,
    tokenCount: totalTokens,
    nextTokenAccuracy: totalTokens === 0 ? 0 : totalHits / totalTokens,
    meanLogProb,
    perSequence,
  };
}

// ---------------------------------------------------------------------------
// Synthetic event-stream generator — deterministic dataset for cloud/CI tests.
// ---------------------------------------------------------------------------

export type MovementMotif = {
  id: string;
  tokens: MovementToken[];
};

/**
 * Deterministically expand a set of motifs into a movement dataset. Each motif
 * is repeated `repeats` times, and (when `interleave` is set) adjacent motifs
 * are concatenated to synthesize novel-but-related compound sequences — exactly
 * the distribution shift a generalizing model must handle. No RNG: identical
 * inputs always yield an identical dataset.
 */
export function generateSyntheticMovementDataset(params: {
  motifs: MovementMotif[];
  repeats?: number;
  interleave?: boolean;
}): MovementTrainingDataset {
  const repeats = Math.max(1, params.repeats ?? 1);
  const sequences: MovementSequence[] = [];

  for (const motif of params.motifs) {
    for (let r = 0; r < repeats; r++) {
      sequences.push({ id: `${motif.id}#${r}`, tokens: [...motif.tokens] });
    }
  }

  if (params.interleave) {
    for (let i = 0; i < params.motifs.length; i++) {
      const a = params.motifs[i];
      const b = params.motifs[(i + 1) % params.motifs.length];
      sequences.push({ id: `${a.id}+${b.id}`, tokens: [...a.tokens, ...b.tokens] });
    }
  }

  return { sequences };
}
