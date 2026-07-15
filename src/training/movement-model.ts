import type { ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";
import type { ReviewedExportManifest } from "./export-manifest.js";

/**
 * Local-movement learning: pluggable model backend.
 *
 * The capture/replay pipeline produces trajectories and replay manifests of
 * recorded on-device movements (mouse/keyboard/gesture actions). This module
 * turns those into a training dataset of movement-token sequences and defines a
 * pluggable {@link MovementModelBackend} that can (c) post-train a model on the
 * dataset to *repeat* recorded movements and (d) *generalize* to new-but-related
 * movements.
 *
 * The shipped {@link NGramMovementBackend} is a deterministic, dependency-free
 * n-gram learner with stupid-backoff. It is a genuine learning algorithm (it
 * fits transition statistics from data and generalizes to unseen contexts via
 * suffix backoff), yet is fully reproducible so it validates the whole pipeline
 * in the cloud/CI with no real OS input. A real on-device small model (e.g. an
 * MLX-trained policy) can implement the same interface later — the training
 * runner and callers stay unchanged.
 */

export type MovementToken = string;

export type MovementSequence = {
  sequenceId: string;
  tokens: MovementToken[];
};

export type MovementTrainingDataset = {
  version: 1;
  sequences: MovementSequence[];
};

export type MovementBackendKind = "mock" | "local" | "remote";

export type MovementTrainOptions = {
  /** Maximum context length (previous tokens) the model conditions on. */
  order?: number;
};

export type MovementPredictOptions = {
  /** Number of ranked candidates to return. */
  topK?: number;
};

export type MovementPredictionCandidate = {
  token: MovementToken;
  probability: number;
  count: number;
};

export type MovementPrediction = {
  /** Highest-ranked next token, or undefined when the model has no data. */
  token: MovementToken | undefined;
  candidates: MovementPredictionCandidate[];
  /** Context length actually used to produce the prediction. */
  matchedOrder: number;
  /** True when the model had to shorten the context to find a match. */
  backedOff: boolean;
};

export type MovementGramEntry = {
  context: MovementToken[];
  continuations: Array<{ token: MovementToken; count: number }>;
};

export type MovementModelArtifact = {
  version: 1;
  backend: string;
  order: number;
  vocabulary: MovementToken[];
  grams: MovementGramEntry[];
  sequenceCount: number;
  tokenCount: number;
};

export interface MovementModel {
  readonly order: number;
  predict(context: MovementToken[], options?: MovementPredictOptions): MovementPrediction;
  serialize(): MovementModelArtifact;
}

export interface MovementModelBackend {
  readonly id: string;
  readonly kind: MovementBackendKind;
  train(dataset: MovementTrainingDataset, options?: MovementTrainOptions): Promise<MovementModel>;
  load(artifact: MovementModelArtifact): MovementModel;
}

export const DEFAULT_MOVEMENT_ORDER = 3;

// --- Tokenization -----------------------------------------------------------

export type MovementTokenizeOptions = {
  /** "actions" (default) keeps only movement actions; "all" also emits observation-context tokens. */
  include?: "actions" | "all";
};

/** Normalize a free-text label into a stable, generalization-friendly token part. */
export function normalizeMovementLabel(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/\s+/g, "-")
    .replace(/[^a-z0-9:_-]/g, "");
}

export function movementActionToken(tool: string, summary: string): MovementToken {
  return `action:${normalizeMovementLabel(tool)}:${normalizeMovementLabel(summary)}`;
}

export function movementObservationToken(source: string, summary: string): MovementToken {
  return `obs:${normalizeMovementLabel(source)}:${normalizeMovementLabel(summary)}`;
}

/** Turn an ordered replay timeline into a movement-token sequence. */
export function tokenizeMovementEvents(
  events: ReplayTimelineEvent[],
  options: MovementTokenizeOptions = {},
): MovementToken[] {
  const include = options.include ?? "actions";
  const tokens: MovementToken[] = [];
  for (const event of events) {
    if (event.kind === "action") {
      tokens.push(movementActionToken(event.tool, event.summary));
    } else if (include === "all" && event.kind === "observation") {
      tokens.push(movementObservationToken(event.source, event.summary));
    }
  }
  return tokens;
}

/** Build a movement sequence from a single trajectory's recorded actions. */
export function buildMovementSequenceFromTrajectory(
  trajectory: TrajectorySpan,
  options: MovementTokenizeOptions = {},
): MovementSequence {
  const include = options.include ?? "actions";
  const merged: Array<{ ts: number; token: MovementToken }> = [
    ...trajectory.actions.map((action) => ({
      ts: action.ts,
      token: movementActionToken(action.tool, action.summary),
    })),
    ...(include === "all"
      ? trajectory.observations.map((observation) => ({
          ts: observation.ts,
          token: movementObservationToken(observation.source, observation.summary),
        }))
      : []),
  ].sort((a, b) => a.ts - b.ts);
  return { sequenceId: trajectory.id, tokens: merged.map((entry) => entry.token) };
}

/** Build a training dataset from the reviewed export manifest's replay bundles. */
export function buildMovementDatasetFromManifest(
  manifest: ReviewedExportManifest,
  options: MovementTokenizeOptions = {},
): MovementTrainingDataset {
  const sequences: MovementSequence[] = manifest.replays.map((replay, index) => ({
    sequenceId: replay.trajectoryIds[0] ?? `${replay.sessionId}-${index}`,
    tokens: tokenizeMovementEvents(replay.events as ReplayTimelineEvent[], options),
  }));
  return { version: 1, sequences };
}

export function buildMovementDataset(sequences: MovementSequence[]): MovementTrainingDataset {
  return { version: 1, sequences };
}

// --- n-gram backend ---------------------------------------------------------

type ContinuationCounts = Map<MovementToken, number>;

class NGramMovementModel implements MovementModel {
  constructor(
    readonly order: number,
    private readonly grams: Map<string, ContinuationCounts>,
    private readonly vocabulary: MovementToken[],
    private readonly sequenceCount: number,
    private readonly tokenCount: number,
    private readonly backendId: string,
  ) {}

  predict(context: MovementToken[], options: MovementPredictOptions = {}): MovementPrediction {
    const topK = Math.max(1, options.topK ?? 5);
    const requested = Math.min(context.length, this.order);
    for (let length = requested; length >= 0; length--) {
      const suffix = context.slice(context.length - length);
      const counts = this.grams.get(contextKey(suffix));
      if (!counts || counts.size === 0) {
        continue;
      }
      const total = sumCounts(counts);
      const candidates = [...counts.entries()]
        .map(([token, count]) => ({ token, count, probability: count / total }))
        // Deterministic ordering: highest count first, ties broken lexically.
        .sort((a, b) => (b.count - a.count) || compareToken(a.token, b.token))
        .slice(0, topK);
      return {
        token: candidates[0]?.token,
        candidates,
        matchedOrder: length,
        backedOff: length < requested,
      };
    }
    return { token: undefined, candidates: [], matchedOrder: 0, backedOff: requested > 0 };
  }

  serialize(): MovementModelArtifact {
    const grams: MovementGramEntry[] = [...this.grams.entries()]
      .map(([key, counts]) => ({
        context: parseContextKey(key),
        continuations: [...counts.entries()]
          .map(([token, count]) => ({ token, count }))
          .sort((a, b) => compareToken(a.token, b.token)),
      }))
      .sort((a, b) => a.context.length - b.context.length || compareToken(contextKey(a.context), contextKey(b.context)));
    return {
      version: 1,
      backend: this.backendId,
      order: this.order,
      vocabulary: [...this.vocabulary].sort(compareToken),
      grams,
      sequenceCount: this.sequenceCount,
      tokenCount: this.tokenCount,
    };
  }
}

export class NGramMovementBackend implements MovementModelBackend {
  readonly id = "ngram-mock";
  readonly kind: MovementBackendKind = "mock";

  async train(
    dataset: MovementTrainingDataset,
    options: MovementTrainOptions = {},
  ): Promise<MovementModel> {
    const order = Math.max(1, options.order ?? DEFAULT_MOVEMENT_ORDER);
    const grams = new Map<string, ContinuationCounts>();
    const vocabulary = new Set<MovementToken>();
    let tokenCount = 0;

    for (const sequence of dataset.sequences) {
      const tokens = sequence.tokens;
      for (let index = 0; index < tokens.length; index++) {
        const next = tokens[index]!;
        vocabulary.add(next);
        tokenCount++;
        // Condition `next` on every context length 0..order.
        for (let length = 0; length <= order; length++) {
          if (index - length < 0) {
            break;
          }
          const context = tokens.slice(index - length, index);
          increment(grams, contextKey(context), next);
        }
      }
    }

    return new NGramMovementModel(
      order,
      grams,
      [...vocabulary],
      dataset.sequences.length,
      tokenCount,
      this.id,
    );
  }

  load(artifact: MovementModelArtifact): MovementModel {
    const grams = new Map<string, ContinuationCounts>();
    for (const entry of artifact.grams) {
      const counts: ContinuationCounts = new Map();
      for (const continuation of entry.continuations) {
        counts.set(continuation.token, continuation.count);
      }
      grams.set(contextKey(entry.context), counts);
    }
    return new NGramMovementModel(
      artifact.order,
      grams,
      [...artifact.vocabulary],
      artifact.sequenceCount,
      artifact.tokenCount,
      artifact.backend,
    );
  }
}

// --- Generalization eval harness -------------------------------------------

export type MovementEvalOptions = {
  topK?: number;
  /** Skip predicting the very first token of each sequence (no real context). */
  skipFirstToken?: boolean;
};

export type MovementEvalResult = {
  sequenceCount: number;
  predictionCount: number;
  top1Hits: number;
  topKHits: number;
  backoffCount: number;
  top1Accuracy: number;
  topKAccuracy: number;
  backoffRate: number;
  topK: number;
};

/**
 * Measure next-movement prediction fidelity on held-out sequences — the
 * generalization signal for "repeat + generalize related movements".
 */
export function evaluateMovementModel(
  model: MovementModel,
  heldOut: MovementSequence[],
  options: MovementEvalOptions = {},
): MovementEvalResult {
  const topK = Math.max(1, options.topK ?? 5);
  const start = options.skipFirstToken ? 1 : 0;
  let predictionCount = 0;
  let top1Hits = 0;
  let topKHits = 0;
  let backoffCount = 0;

  for (const sequence of heldOut) {
    for (let index = start; index < sequence.tokens.length; index++) {
      const context = sequence.tokens.slice(0, index);
      const actual = sequence.tokens[index]!;
      const prediction = model.predict(context, { topK });
      predictionCount++;
      if (prediction.backedOff) {
        backoffCount++;
      }
      if (prediction.token === actual) {
        top1Hits++;
      }
      if (prediction.candidates.some((candidate) => candidate.token === actual)) {
        topKHits++;
      }
    }
  }

  return {
    sequenceCount: heldOut.length,
    predictionCount,
    top1Hits,
    topKHits,
    backoffCount,
    top1Accuracy: predictionCount === 0 ? 0 : top1Hits / predictionCount,
    topKAccuracy: predictionCount === 0 ? 0 : topKHits / predictionCount,
    backoffRate: predictionCount === 0 ? 0 : backoffCount / predictionCount,
    topK,
  };
}

// --- helpers ----------------------------------------------------------------

function increment(grams: Map<string, ContinuationCounts>, key: string, token: MovementToken): void {
  let counts = grams.get(key);
  if (!counts) {
    counts = new Map();
    grams.set(key, counts);
  }
  counts.set(token, (counts.get(token) ?? 0) + 1);
}

function sumCounts(counts: ContinuationCounts): number {
  let total = 0;
  for (const value of counts.values()) {
    total += value;
  }
  return total;
}

function contextKey(context: MovementToken[]): string {
  return JSON.stringify(context);
}

function parseContextKey(key: string): MovementToken[] {
  return JSON.parse(key) as MovementToken[];
}

function compareToken(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0;
}
