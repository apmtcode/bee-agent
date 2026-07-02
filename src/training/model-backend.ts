import type { ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Pluggable local-model backend for the movement-learning subsystem.
 *
 * The training *plan/launch* pipeline (`runner.ts` / `execution-service.ts`)
 * emits Apple-Silicon shell scripts that drive `mlx`/`axolotl` on the user's
 * machine. Those cannot run in the cloud/CI, and they do not give bee-agent an
 * in-process way to (c) reproduce recorded movements or (d) generalize to
 * related ones.
 *
 * This module fills that gap with a small, backend-agnostic seam:
 *   - `MovementModelBackend.train(dataset)` learns a policy from movement
 *     sequences and returns a `TrainedMovementModel`.
 *   - `TrainedMovementModel.predictNext(context)` predicts the next movement
 *     token; `generate(prefix)` rolls out a full continuation.
 *   - `serialize()` / `restore()` round-trip the model as plain JSON so it can
 *     be persisted next to the reviewed export.
 *
 * The default backend (`NgramMovementBackend`) is fully deterministic and needs
 * no native deps, so it validates the whole capture→dataset→train→infer loop in
 * the cloud. A real on-device small model can implement the same interface
 * behind the `createMovementModelBackend` registry without touching callers.
 */

/** A single recorded movement, normalized from a replay/trajectory event. */
export type MovementEvent =
  | { kind: "observation"; source: string; summary: string }
  | { kind: "action"; tool: string; summary: string };

/** Ordered movements for one trajectory, plus optional outcome weighting. */
export type MovementSequence = {
  trajectoryId?: string;
  tokens: string[];
  /** Relative weight (e.g. reward-scaled); defaults to 1 during counting. */
  weight?: number;
};

export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
  /** Sorted unique token vocabulary (includes boundary markers). */
  vocabulary: string[];
};

export type MovementTrainOptions = {
  /** Max n-gram order (context length + 1). Default 3. */
  order?: number;
};

export type MovementCandidate = { token: string; probability: number };

export type MovementPrediction = {
  /** Predicted next token, or `undefined` when the sequence should end. */
  token: string | undefined;
  probability: number;
  /** Which n-gram order matched after backoff (0 = unigram fallback). */
  matchedOrder: number;
  /** All candidates at the matched order, highest probability first. */
  candidates: MovementCandidate[];
};

export type MovementGenerateOptions = {
  maxSteps?: number;
};

export type MovementModelSnapshot = {
  backendId: string;
  version: 1;
  order: number;
  /** context-token-joined -> { nextToken -> weightedCount }. */
  counts: Record<string, Record<string, number>>;
  vocabulary: string[];
};

export interface TrainedMovementModel {
  readonly backendId: string;
  readonly order: number;
  /** Predict the next token given the trailing context (unbounded length). */
  predictNext(context: string[]): MovementPrediction;
  /** Roll out a continuation from `prefix` until BOUNDARY_END or maxSteps. */
  generate(prefix: string[], options?: MovementGenerateOptions): string[];
  serialize(): MovementModelSnapshot;
}

export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainOptions): Promise<TrainedMovementModel>;
}

/** Sequence boundary sentinels. Kept out of the emitted movement stream. */
export const BOUNDARY_START = "bos";
export const BOUNDARY_END = "eos";
const CONTEXT_DELIMITER = "";

// ---------------------------------------------------------------------------
// Tokenization + dataset construction
// ---------------------------------------------------------------------------

const FIELD_DELIMITER = "";

/** Deterministic, reversible token for a normalized movement event. */
export function encodeMovementToken(event: MovementEvent): string {
  return event.kind === "action"
    ? `act${FIELD_DELIMITER}${event.tool}${FIELD_DELIMITER}${event.summary}`
    : `obs${FIELD_DELIMITER}${event.source}${FIELD_DELIMITER}${event.summary}`;
}

/** Inverse of {@link encodeMovementToken}; `undefined` for boundary tokens. */
export function decodeMovementToken(token: string): MovementEvent | undefined {
  const [kind, first, ...rest] = token.split(FIELD_DELIMITER);
  const summary = rest.join(FIELD_DELIMITER);
  if (kind === "act" && first !== undefined) {
    return { kind: "action", tool: first, summary };
  }
  if (kind === "obs" && first !== undefined) {
    return { kind: "observation", source: first, summary };
  }
  return undefined;
}

export type ReplaySource = { trajectoryIds: string[]; events: ReplayTimelineEvent[] };

export type BuildMovementDatasetOptions = {
  /** Include observation events as context tokens (default true). */
  includeObservations?: boolean;
  order?: number;
};

/** Convert reviewed replay manifests into a trainable movement dataset. */
export function buildMovementDatasetFromReplays(
  replays: ReplaySource[],
  options: BuildMovementDatasetOptions = {},
): MovementDataset {
  const includeObservations = options.includeObservations ?? true;
  const sequences: MovementSequence[] = replays.map((replay) => {
    const tokens: string[] = [];
    for (const event of replay.events) {
      if (event.kind === "action") {
        tokens.push(encodeMovementToken({ kind: "action", tool: event.tool, summary: event.summary }));
      } else if (event.kind === "observation" && includeObservations) {
        tokens.push(encodeMovementToken({ kind: "observation", source: event.source, summary: event.summary }));
      }
    }
    return { trajectoryId: replay.trajectoryIds[0], tokens };
  });
  return finalizeDataset(sequences);
}

/** Convert raw trajectory spans into a movement dataset (reward-weighted). */
export function buildMovementDatasetFromTrajectories(
  trajectories: TrajectorySpan[],
  options: BuildMovementDatasetOptions = {},
): MovementDataset {
  const includeObservations = options.includeObservations ?? true;
  const sequences: MovementSequence[] = trajectories.map((trajectory) => {
    const timeline = [
      ...(includeObservations ? trajectory.observations : []),
      ...trajectory.actions,
    ].sort((a, b) => a.ts - b.ts);
    const tokens = timeline.map((event) =>
      event.kind === "action"
        ? encodeMovementToken({ kind: "action", tool: event.tool, summary: event.summary })
        : encodeMovementToken({ kind: "observation", source: event.source, summary: event.summary }),
    );
    const reward = trajectory.outcome?.reward;
    return {
      trajectoryId: trajectory.id,
      tokens,
      ...(typeof reward === "number" && reward > 0 ? { weight: reward } : {}),
    };
  });
  return finalizeDataset(sequences);
}

function finalizeDataset(sequences: MovementSequence[]): MovementDataset {
  const vocabulary = new Set<string>([BOUNDARY_START, BOUNDARY_END]);
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

// ---------------------------------------------------------------------------
// Deterministic n-gram backend (default, cloud/CI-safe)
// ---------------------------------------------------------------------------

/**
 * Learns a weighted n-gram model over movement sequences with stupid-backoff
 * decoding. Reproduces recorded trajectories exactly (argmax on the seen
 * context) and generalizes to new-but-related sequences by backing off to the
 * longest matching suffix. No randomness: ties break by descending
 * weighted-count then ascending token, so training is fully reproducible.
 */
export class NgramMovementBackend implements MovementModelBackend {
  readonly id = "ngram";

  async train(dataset: MovementDataset, options: MovementTrainOptions = {}): Promise<TrainedMovementModel> {
    const order = normalizeOrder(options.order ?? 3);
    const counts = new Map<string, Map<string, number>>();

    for (const sequence of dataset.sequences) {
      const weight = sequence.weight ?? 1;
      const padded = [
        ...Array.from({ length: order - 1 }, () => BOUNDARY_START),
        ...sequence.tokens,
        BOUNDARY_END,
      ];
      // For every order 1..order, count context -> next transitions.
      for (let contextLength = 0; contextLength < order; contextLength += 1) {
        for (let i = contextLength; i < padded.length; i += 1) {
          const context = padded.slice(i - contextLength, i);
          const next = padded[i]!;
          addCount(counts, contextKey(context), next, weight);
        }
      }
    }

    return new NgramMovementModel(this.id, order, counts, dataset.vocabulary);
  }
}

class NgramMovementModel implements TrainedMovementModel {
  readonly backendId: string;
  readonly order: number;
  private readonly counts: Map<string, Map<string, number>>;
  private readonly vocabulary: string[];

  constructor(
    backendId: string,
    order: number,
    counts: Map<string, Map<string, number>>,
    vocabulary: string[],
  ) {
    this.backendId = backendId;
    this.order = order;
    this.counts = counts;
    this.vocabulary = vocabulary;
  }

  predictNext(context: string[]): MovementPrediction {
    // Stupid-backoff: try the longest context (up to order-1), then shorten.
    for (let contextLength = Math.min(context.length, this.order - 1); contextLength >= 0; contextLength -= 1) {
      const slice = context.slice(context.length - contextLength);
      const distribution = this.counts.get(contextKey(slice));
      if (!distribution || distribution.size === 0) {
        continue;
      }
      const total = [...distribution.values()].reduce((sum, value) => sum + value, 0);
      const candidates = [...distribution.entries()]
        .map(([token, count]) => ({ token, probability: count / total }))
        .sort(compareCandidates);
      const best = candidates[0]!;
      return {
        token: best.token === BOUNDARY_END ? undefined : best.token,
        probability: best.probability,
        matchedOrder: contextLength,
        candidates,
      };
    }
    return { token: undefined, probability: 0, matchedOrder: 0, candidates: [] };
  }

  generate(prefix: string[], options: MovementGenerateOptions = {}): string[] {
    const maxSteps = options.maxSteps ?? 256;
    const emitted: string[] = [];
    const context = [...prefix];
    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predictNext(context);
      const raw = prediction.candidates[0]?.token;
      if (raw === undefined || raw === BOUNDARY_END) {
        break;
      }
      emitted.push(raw);
      context.push(raw);
    }
    return emitted;
  }

  serialize(): MovementModelSnapshot {
    const counts: Record<string, Record<string, number>> = {};
    for (const [key, distribution] of this.counts) {
      counts[key] = Object.fromEntries(distribution);
    }
    return {
      backendId: this.backendId,
      version: 1,
      order: this.order,
      counts,
      vocabulary: [...this.vocabulary],
    };
  }

  static restore(snapshot: MovementModelSnapshot): NgramMovementModel {
    const counts = new Map<string, Map<string, number>>();
    for (const [key, distribution] of Object.entries(snapshot.counts)) {
      counts.set(key, new Map(Object.entries(distribution)));
    }
    return new NgramMovementModel(snapshot.backendId, snapshot.order, counts, snapshot.vocabulary);
  }
}

/** Rehydrate a persisted snapshot into a runnable model (backend-dispatched). */
export function restoreMovementModel(snapshot: MovementModelSnapshot): TrainedMovementModel {
  switch (snapshot.backendId) {
    case "ngram":
      return NgramMovementModel.restore(snapshot);
    default:
      throw new Error(`Unknown movement model backend: ${snapshot.backendId}`);
  }
}

export type MovementBackendKind = "ngram";

/**
 * Registry seam for pluggable backends. Cloud/CI uses the deterministic
 * `ngram` backend; a real on-device small-model backend registers here.
 */
export function createMovementModelBackend(kind: MovementBackendKind = "ngram"): MovementModelBackend {
  switch (kind) {
    case "ngram":
      return new NgramMovementBackend();
    default:
      throw new Error(`Unknown movement model backend kind: ${kind satisfies never}`);
  }
}

function normalizeOrder(order: number): number {
  if (!Number.isInteger(order) || order < 1) {
    throw new Error(`n-gram order must be a positive integer, got ${order}`);
  }
  return order;
}

function contextKey(context: string[]): string {
  return context.join(CONTEXT_DELIMITER);
}

function addCount(
  counts: Map<string, Map<string, number>>,
  key: string,
  token: string,
  weight: number,
): void {
  let distribution = counts.get(key);
  if (!distribution) {
    distribution = new Map<string, number>();
    counts.set(key, distribution);
  }
  distribution.set(token, (distribution.get(token) ?? 0) + weight);
}

function compareCandidates(a: MovementCandidate, b: MovementCandidate): number {
  if (a.probability !== b.probability) {
    return b.probability - a.probability;
  }
  return a.token < b.token ? -1 : a.token > b.token ? 1 : 0;
}
