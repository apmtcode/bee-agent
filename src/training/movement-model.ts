import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Pluggable local-movement model backend.
 *
 * This is the on-device learning seam for standing objective #2(d): post-train a
 * local model on recorded movements so bee-agent can (a) repeat the recorded
 * movements and (b) generalize to new-but-related movements.
 *
 * The heavy on-device runtimes (mlx / axolotl, wired in `runner.ts`) execute only
 * when the user runs bee-agent locally. To keep the *pipeline* testable in the
 * cloud we model the backend as an interface and ship a deterministic, dependency
 * free reference backend (`NGramMovementBackend`) that trains and infers fully in
 * process. A real neural backend implements the same interface and drops in.
 */

/** Canonical, replayable representation of a single movement/action. */
export type MovementToken = string;

export const MOVEMENT_START_TOKEN: MovementToken = "start";
export const MOVEMENT_END_TOKEN: MovementToken = "end";

/** One recorded (or synthetic) sequence of movements. */
export type MovementSequence = {
  id: string;
  /** Ordered movement tokens, earliest first. Boundary tokens are added at train time. */
  tokens: MovementToken[];
};

/** The replayable dataset a backend trains on. */
export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

export type MovementTrainingOptions = {
  /** Context window (n-gram order). Higher = more literal replay, less generalization. */
  order?: number;
};

export type MovementPrediction = {
  token: MovementToken;
  probability: number;
  /** Context length actually matched after stupid-backoff (order - 1 down to 0). */
  matchedContext: number;
};

export type GenerateOptions = {
  /** Hard cap on generated tokens (excludes the boundary END token). */
  maxSteps?: number;
};

export type SerializedMovementModel = {
  backendId: string;
  order: number;
  /** contextKey (JSON-encoded context array) -> (token -> count). */
  counts: Record<string, Record<MovementToken, number>>;
};

/** A trained model: repeat what was recorded, generalize to related movements. */
export interface TrainedMovementModel {
  readonly backendId: string;
  readonly order: number;
  /** Ranked next-movement candidates for the given recent context (most recent last). */
  predictNext(context: MovementToken[]): MovementPrediction[];
  /** Roll a full movement sequence forward from an optional prompt until END/maxSteps. */
  generate(prompt?: MovementToken[], options?: GenerateOptions): MovementToken[];
  serialize(): SerializedMovementModel;
}

/** The pluggable seam: swap this for a real on-device runtime. */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainingOptions): TrainedMovementModel;
}

const DEFAULT_ORDER = 3;

/**
 * Deterministic n-gram backend with stupid-backoff.
 *
 * - Repeats recorded movements: seen contexts resolve to their observed next move.
 * - Generalizes: an unseen context backs off to shorter suffixes, so a novel prefix
 *   that shares a tail with training data still yields a sensible next movement.
 * - Fully deterministic (argmax with lexicographic tie-break) so cloud/CI tests are
 *   reproducible without a RNG.
 */
export class NGramMovementBackend implements MovementModelBackend {
  readonly id = "ngram";

  train(dataset: MovementDataset, options: MovementTrainingOptions = {}): TrainedMovementModel {
    const order = normalizeOrder(options.order);
    const counts = new Map<string, Map<MovementToken, number>>();

    for (const sequence of dataset.sequences) {
      const padded = padSequence(sequence.tokens, order);
      for (let i = order - 1; i < padded.length; i += 1) {
        const next = padded[i]!;
        // Record every backoff context (length order-1 down to 0) so inference can
        // fall back deterministically without re-deriving suffixes.
        for (let contextLength = 0; contextLength < order; contextLength += 1) {
          const context = padded.slice(i - contextLength, i);
          bump(counts, contextKey(context), next);
        }
      }
    }

    return new NGramMovementModel(this.id, order, counts);
  }

  static fromSerialized(serialized: SerializedMovementModel): TrainedMovementModel {
    const counts = new Map<string, Map<MovementToken, number>>();
    for (const [key, tokenCounts] of Object.entries(serialized.counts)) {
      const inner = new Map<MovementToken, number>();
      for (const [token, count] of Object.entries(tokenCounts)) {
        inner.set(token, count);
      }
      counts.set(key, inner);
    }
    return new NGramMovementModel(serialized.backendId, normalizeOrder(serialized.order), counts);
  }
}

class NGramMovementModel implements TrainedMovementModel {
  constructor(
    readonly backendId: string,
    readonly order: number,
    private readonly counts: Map<string, Map<MovementToken, number>>,
  ) {}

  predictNext(context: MovementToken[]): MovementPrediction[] {
    // Stupid-backoff: try the longest suffix (up to order-1) that has counts.
    for (let contextLength = Math.min(context.length, this.order - 1); contextLength >= 0; contextLength -= 1) {
      const suffix = context.slice(context.length - contextLength);
      const tokenCounts = this.counts.get(contextKey(suffix));
      if (!tokenCounts || tokenCounts.size === 0) {
        continue;
      }
      const total = [...tokenCounts.values()].reduce((sum, count) => sum + count, 0);
      return [...tokenCounts.entries()]
        .map(([token, count]) => ({ token, probability: count / total, matchedContext: contextLength }))
        .sort(comparePredictions);
    }
    return [];
  }

  generate(prompt: MovementToken[] = [], options: GenerateOptions = {}): MovementToken[] {
    const maxSteps = options.maxSteps ?? 64;
    const startPad = Array.from({ length: this.order - 1 }, () => MOVEMENT_START_TOKEN);
    const history = [...startPad, ...prompt.filter((token) => token !== MOVEMENT_START_TOKEN)];
    const produced: MovementToken[] = [];

    for (let step = 0; step < maxSteps; step += 1) {
      const [best] = this.predictNext(history);
      if (!best || best.token === MOVEMENT_END_TOKEN) {
        break;
      }
      produced.push(best.token);
      history.push(best.token);
    }
    return produced;
  }

  serialize(): SerializedMovementModel {
    const counts: Record<string, Record<MovementToken, number>> = {};
    for (const [key, tokenCounts] of this.counts.entries()) {
      const inner: Record<MovementToken, number> = {};
      for (const [token, count] of [...tokenCounts.entries()].sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))) {
        inner[token] = count;
      }
      counts[key] = inner;
    }
    return { backendId: this.backendId, order: this.order, counts };
  }
}

export type MovementModelEvaluation = {
  sequenceCount: number;
  /** Total next-token predictions scored across all held-out sequences. */
  predictions: number;
  /** Correct top-1 next-token predictions. */
  correct: number;
  /** correct / predictions (0 when there was nothing to predict). */
  accuracy: number;
  /** Sequences reproduced exactly by greedy generation from their first real token. */
  exactSequenceMatches: number;
};

/**
 * Replay-fidelity / generalization eval (ROADMAP: "generalization eval harness").
 * Measures next-movement top-1 accuracy on held-out sequences and how many are
 * reproduced end-to-end by greedy generation.
 */
export function evaluateMovementModel(
  model: TrainedMovementModel,
  heldOut: MovementSequence[],
): MovementModelEvaluation {
  let predictions = 0;
  let correct = 0;
  let exactSequenceMatches = 0;

  for (const sequence of heldOut) {
    const padded = padSequence(sequence.tokens, model.order);
    for (let i = model.order - 1; i < padded.length; i += 1) {
      const expected = padded[i]!;
      const context = padded.slice(0, i);
      const [best] = model.predictNext(context);
      predictions += 1;
      if (best && best.token === expected) {
        correct += 1;
      }
    }

    if (sequence.tokens.length > 0) {
      const [seed, ...rest] = sequence.tokens;
      const generated = model.generate([seed!]);
      if (arraysEqual([seed!, ...generated], [seed!, ...rest])) {
        exactSequenceMatches += 1;
      }
    }
  }

  return {
    sequenceCount: heldOut.length,
    predictions,
    correct,
    accuracy: predictions === 0 ? 0 : correct / predictions,
    exactSequenceMatches,
  };
}

/** Extract a training token from a single replay/trajectory action event. */
export function movementTokenFromAction(action: { tool: string; summary: string }): MovementToken {
  return `${normalizeField(action.tool)}:${normalizeField(action.summary)}`;
}

/** Build a replayable movement dataset from reviewed replay manifests. */
export function buildMovementDataset(replays: ReplayManifest[]): MovementDataset {
  const sequences: MovementSequence[] = [];
  for (const replay of replays) {
    const tokens = replay.events
      .filter((event): event is Extract<ReplayTimelineEvent, { kind: "action" }> => event.kind === "action")
      .map((event) => movementTokenFromAction(event));
    if (tokens.length > 0) {
      sequences.push({ id: replay.sessionId, tokens });
    }
  }
  return { version: 1, sequences };
}

/** Build a movement dataset directly from captured trajectory spans. */
export function buildMovementDatasetFromTrajectories(trajectories: TrajectorySpan[]): MovementDataset {
  const sequences: MovementSequence[] = [];
  for (const trajectory of trajectories) {
    const tokens = [...trajectory.actions]
      .sort((a, b) => a.ts - b.ts)
      .map((action) => movementTokenFromAction(action));
    if (tokens.length > 0) {
      sequences.push({ id: trajectory.id, tokens });
    }
  }
  return { version: 1, sequences };
}

function normalizeOrder(order: number | undefined): number {
  if (order === undefined || !Number.isFinite(order)) {
    return DEFAULT_ORDER;
  }
  return Math.max(1, Math.floor(order));
}

function padSequence(tokens: MovementToken[], order: number): MovementToken[] {
  const startPad = Array.from({ length: order - 1 }, () => MOVEMENT_START_TOKEN);
  return [...startPad, ...tokens, MOVEMENT_END_TOKEN];
}

function bump(counts: Map<string, Map<MovementToken, number>>, key: string, token: MovementToken): void {
  let inner = counts.get(key);
  if (!inner) {
    inner = new Map<MovementToken, number>();
    counts.set(key, inner);
  }
  inner.set(token, (inner.get(token) ?? 0) + 1);
}

function contextKey(context: MovementToken[]): string {
  // JSON encoding is collision-free across arbitrary token contents (a plain
  // delimiter join could conflate ["a","bc"] with ["ab","c"]).
  return JSON.stringify(context);
}

function comparePredictions(a: MovementPrediction, b: MovementPrediction): number {
  if (a.probability !== b.probability) {
    return b.probability - a.probability;
  }
  return a.token < b.token ? -1 : a.token > b.token ? 1 : 0;
}

function normalizeField(value: string): string {
  return value.trim().replace(/\s+/g, " ");
}

function arraysEqual(a: MovementToken[], b: MovementToken[]): boolean {
  if (a.length !== b.length) {
    return false;
  }
  return a.every((value, index) => value === b[index]);
}
