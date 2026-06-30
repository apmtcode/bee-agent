// Pluggable local-model backend for the movement-learning subsystem.
//
// Objective #2(d): "post-train a local model on the recorded movement dataset to
// repeat the recorded movements, and generalize to perform new but related
// movements."
//
// The runner in `runner.ts` emits Apple-Silicon shell plans (mlx/axolotl) that
// only execute on a real on-device GPU. That seam is correct for production, but
// nothing could train or run inference *in the cloud/CI*, so the end-to-end
// pipeline (capture -> dataset -> train -> infer) was untestable.
//
// This module adds a backend-agnostic interface plus a deterministic, in-process
// reference backend (a variable-order Markov model with interpolated backoff).
// It is a real — if small — local model: it memorizes recorded movement
// sequences and generalizes to related-but-unseen prefixes via lower-order
// context. Because it is pure (no OS input, no Python, no randomness) the whole
// pipeline is validated with synthetic event streams in tests. Swap in an
// on-device backend (e.g. an mlx-backed adapter) by implementing the same
// `LocalMovementModelBackend` interface.

import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/** A canonical, order-stable token describing a single movement/action. */
export type MovementToken = string;

/** Sentinel tokens marking sequence boundaries (used for context + stop). */
export const MOVEMENT_BOS: MovementToken = "BOS";
export const MOVEMENT_EOS: MovementToken = "EOS";

/** One ordered run of movements, typically derived from a single trajectory. */
export type MovementSequence = {
  trajectoryId: string;
  sessionId?: string;
  tokens: MovementToken[];
};

export type MovementTrainingDataset = {
  sequences: MovementSequence[];
};

export type MovementTrainingOptions = {
  /** Maximum Markov context length (n-gram order). Default 3. */
  order?: number;
  /** Backoff weight applied per dropped context token. Default 0.4. */
  backoff?: number;
};

export type MovementPrediction = {
  token: MovementToken;
  /** Normalized score in [0,1] across the returned + withheld candidates. */
  probability: number;
};

export type MovementGenerationOptions = {
  /** Hard cap on generated tokens (excluding the stop sentinel). Default 64. */
  maxSteps?: number;
  /** Stop as soon as EOS is the top prediction. Default true. */
  stopAtEos?: boolean;
};

/** JSON-serializable model produced by a backend's `train`. */
export type SerializableMovementModel = {
  version: 1;
  backendId: string;
  order: number;
  backoff: number;
  /** context-key -> nextToken -> count. context-key joins tokens with . */
  counts: Record<string, Record<MovementToken, number>>;
  vocabulary: MovementToken[];
};

/** A loaded model: predict the next movement, or autoregress a full sequence. */
export interface MovementModelHandle {
  readonly backendId: string;
  predictNext(prefix: MovementToken[], topK?: number): MovementPrediction[];
  generate(prefix: MovementToken[], options?: MovementGenerationOptions): MovementToken[];
  serialize(): SerializableMovementModel;
}

/** Pluggable backend seam. Implement this to wire a real on-device model. */
export interface LocalMovementModelBackend {
  readonly id: string;
  train(dataset: MovementTrainingDataset, options?: MovementTrainingOptions): MovementModelHandle;
  load(model: SerializableMovementModel): MovementModelHandle;
}

const CONTEXT_SEPARATOR = "";

// ---------------------------------------------------------------------------
// Tokenization: ReplayTimelineEvent / TrajectorySpan -> MovementToken
// ---------------------------------------------------------------------------

/**
 * Canonical token for an action movement. Stable across runs: `tool:slug`, where
 * the slug is a lowercased, punctuation-normalized form of the action summary so
 * that "tapped Login" and "tapped login!" collapse to the same movement.
 */
export function tokenizeMovement(event: {
  tool: string;
  summary: string;
  metadata?: Record<string, unknown>;
}): MovementToken {
  const gesture = typeof event.metadata?.gesture === "string" ? event.metadata.gesture : undefined;
  const direction = typeof event.metadata?.direction === "string" ? event.metadata.direction : undefined;
  const target = typeof event.metadata?.target === "string" ? event.metadata.target : undefined;
  const facets = [event.tool, gesture, direction ?? target, slug(event.summary)].filter(
    (facet): facet is string => typeof facet === "string" && facet.length > 0,
  );
  // De-duplicate adjacent identical facets (e.g. tool already in summary slug).
  const compact = facets.filter((facet, index) => facet !== facets[index - 1]);
  return compact.join(":");
}

function slug(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

/** Build movement sequences from trajectories: one sequence per trajectory. */
export function extractMovementSequences(trajectories: TrajectorySpan[]): MovementSequence[] {
  return trajectories
    .map((trajectory) => ({
      trajectoryId: trajectory.id,
      sessionId: trajectory.sessionId,
      tokens: [...trajectory.actions]
        .sort((a, b) => a.ts - b.ts)
        .map((action) => tokenizeMovement(action)),
    }))
    .filter((sequence) => sequence.tokens.length > 0);
}

/** Build movement sequences from replay manifests, grouping actions by trajectory. */
export function extractMovementSequencesFromReplays(manifests: ReplayManifest[]): MovementSequence[] {
  const sequences: MovementSequence[] = [];
  for (const manifest of manifests) {
    const byTrajectory = new Map<string, { ts: number; token: MovementToken }[]>();
    for (const event of manifest.events) {
      if (event.kind !== "action") {
        continue;
      }
      const action = event as Extract<ReplayTimelineEvent, { kind: "action" }>;
      const bucket = byTrajectory.get(action.trajectoryId) ?? [];
      bucket.push({ ts: action.ts, token: tokenizeMovement({ tool: action.tool, summary: action.summary }) });
      byTrajectory.set(action.trajectoryId, bucket);
    }
    for (const trajectoryId of [...byTrajectory.keys()].sort()) {
      const entries = byTrajectory.get(trajectoryId) ?? [];
      sequences.push({
        trajectoryId,
        sessionId: manifest.sessionId,
        tokens: entries.sort((a, b) => a.ts - b.ts).map((entry) => entry.token),
      });
    }
  }
  return sequences.filter((sequence) => sequence.tokens.length > 0);
}

// ---------------------------------------------------------------------------
// Reference backend: variable-order Markov model with interpolated backoff
// ---------------------------------------------------------------------------

class MarkovMovementModel implements MovementModelHandle {
  readonly backendId: string;
  private readonly order: number;
  private readonly backoff: number;
  private readonly counts: Map<string, Map<MovementToken, number>>;
  private readonly vocabulary: MovementToken[];

  constructor(model: SerializableMovementModel) {
    this.backendId = model.backendId;
    this.order = model.order;
    this.backoff = model.backoff;
    this.vocabulary = [...model.vocabulary];
    this.counts = new Map();
    for (const [context, nextCounts] of Object.entries(model.counts)) {
      this.counts.set(context, new Map(Object.entries(nextCounts)));
    }
  }

  predictNext(prefix: MovementToken[], topK = 5): MovementPrediction[] {
    const padded = [MOVEMENT_BOS, ...prefix];
    const scores = new Map<MovementToken, number>();
    // Interpolate over every context length 1..order, weighting longer (more
    // specific) contexts higher and shorter ones as a generalization fallback.
    for (let k = Math.min(this.order, padded.length); k >= 1; k--) {
      const context = padded.slice(padded.length - k).join(CONTEXT_SEPARATOR);
      const nextCounts = this.counts.get(context);
      if (!nextCounts) {
        continue;
      }
      let total = 0;
      for (const count of nextCounts.values()) {
        total += count;
      }
      if (total === 0) {
        continue;
      }
      const weight = Math.pow(this.backoff, this.order - k);
      for (const [token, count] of nextCounts) {
        scores.set(token, (scores.get(token) ?? 0) + weight * (count / total));
      }
    }
    let totalScore = 0;
    for (const score of scores.values()) {
      totalScore += score;
    }
    const ranked = [...scores.entries()]
      .map(([token, score]) => ({ token, probability: totalScore > 0 ? score / totalScore : 0 }))
      // Deterministic ordering: score desc, then token asc as a stable tiebreak.
      .sort((a, b) => (b.probability - a.probability) || (a.token < b.token ? -1 : a.token > b.token ? 1 : 0));
    return topK > 0 ? ranked.slice(0, topK) : ranked;
  }

  generate(prefix: MovementToken[], options: MovementGenerationOptions = {}): MovementToken[] {
    const maxSteps = options.maxSteps ?? 64;
    const stopAtEos = options.stopAtEos ?? true;
    const generated: MovementToken[] = [];
    let context = [...prefix];
    for (let step = 0; step < maxSteps; step++) {
      const [best] = this.predictNext(context, 1);
      if (!best) {
        break;
      }
      if (best.token === MOVEMENT_EOS) {
        if (stopAtEos) {
          break;
        }
        continue;
      }
      generated.push(best.token);
      context = [...context, best.token];
    }
    return generated;
  }

  serialize(): SerializableMovementModel {
    const counts: Record<string, Record<MovementToken, number>> = {};
    for (const context of [...this.counts.keys()].sort()) {
      const nextCounts = this.counts.get(context);
      if (!nextCounts) {
        continue;
      }
      const serializedNext: Record<MovementToken, number> = {};
      for (const token of [...nextCounts.keys()].sort()) {
        serializedNext[token] = nextCounts.get(token) ?? 0;
      }
      counts[context] = serializedNext;
    }
    return {
      version: 1,
      backendId: this.backendId,
      order: this.order,
      backoff: this.backoff,
      counts,
      vocabulary: [...this.vocabulary].sort(),
    };
  }
}

/**
 * Deterministic, in-process reference backend. Trains a variable-order Markov
 * model over movement tokens with interpolated backoff — it reproduces recorded
 * movement runs exactly and generalizes to related-but-unseen prefixes through
 * shorter shared context.
 */
export class MarkovMovementBackend implements LocalMovementModelBackend {
  readonly id = "markov-local";

  train(dataset: MovementTrainingDataset, options: MovementTrainingOptions = {}): MovementModelHandle {
    const order = Math.max(1, Math.floor(options.order ?? 3));
    const backoff = options.backoff ?? 0.4;
    const counts: Record<string, Record<MovementToken, number>> = {};
    const vocabulary = new Set<MovementToken>();

    for (const sequence of dataset.sequences) {
      const tokens = [MOVEMENT_BOS, ...sequence.tokens, MOVEMENT_EOS];
      for (const token of sequence.tokens) {
        vocabulary.add(token);
      }
      for (let i = 1; i < tokens.length; i++) {
        const next = tokens[i] as MovementToken;
        for (let k = 1; k <= order; k++) {
          if (i - k < 0) {
            break;
          }
          const context = tokens.slice(i - k, i).join(CONTEXT_SEPARATOR);
          const bucket = (counts[context] ??= {});
          bucket[next] = (bucket[next] ?? 0) + 1;
        }
      }
    }

    return new MarkovMovementModel({
      version: 1,
      backendId: this.id,
      order,
      backoff,
      counts,
      vocabulary: [...vocabulary],
    });
  }

  load(model: SerializableMovementModel): MovementModelHandle {
    return new MarkovMovementModel(model);
  }
}

// ---------------------------------------------------------------------------
// Generalization eval harness
// ---------------------------------------------------------------------------

export type MovementEvalResult = {
  /** Number of next-token predictions scored. */
  predictions: number;
  /** Fraction where the held-out next token was the top-1 prediction. */
  top1Accuracy: number;
  /** Fraction where the held-out next token was within the top-K predictions. */
  topKAccuracy: number;
  k: number;
};

/**
 * Measure how well a trained model predicts movements in held-out (but related)
 * sequences. For every position in each held-out sequence we feed the true
 * prefix and check whether the model's prediction(s) include the actual next
 * movement — the generalization signal the objective calls for.
 */
export function evaluateMovementModel(
  model: MovementModelHandle,
  heldOut: MovementSequence[],
  k = 3,
): MovementEvalResult {
  let predictions = 0;
  let top1 = 0;
  let topK = 0;
  for (const sequence of heldOut) {
    const target = [...sequence.tokens, MOVEMENT_EOS];
    for (let i = 0; i < target.length; i++) {
      const prefix = sequence.tokens.slice(0, i);
      const expected = target[i];
      const ranked = model.predictNext(prefix, k);
      predictions += 1;
      if (ranked[0]?.token === expected) {
        top1 += 1;
      }
      if (ranked.some((prediction) => prediction.token === expected)) {
        topK += 1;
      }
    }
  }
  return {
    predictions,
    top1Accuracy: predictions > 0 ? top1 / predictions : 0,
    topKAccuracy: predictions > 0 ? topK / predictions : 0,
    k,
  };
}
