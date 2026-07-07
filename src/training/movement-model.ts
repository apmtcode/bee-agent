import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * In-process, pluggable local-movement model.
 *
 * The training runner ({@link ../training/runner.ts}) emits launch scripts for
 * real on-device trainers (MLX / axolotl) that only run on the user's machine.
 * This module provides the *cloud-runnable* seam that the standing objective
 * calls for: a backend interface plus a deterministic mock backend that can
 * actually (a) train on recorded movement sequences, (b) reproduce them
 * exactly, and (c) generalize to new-but-related movements — all without any
 * real OS access, so it is fully testable in CI.
 *
 * A "movement" is the ordered sequence of action tokens a trajectory / replay
 * performed. The default backend is a variable-order Markov model with
 * stupid-backoff interpolation: an exactly-seen prefix reproduces its recorded
 * continuation (memorization / replay), while a novel prefix falls back to
 * shorter matching contexts and still predicts a plausible next movement
 * (generalization).
 */

export type MovementToken = string;

/** Sentinel tokens framing every training sequence. */
export const MOVEMENT_START: MovementToken = "<start>";
export const MOVEMENT_END: MovementToken = "<end>";

/** Separator used to key an ordered context deterministically. */
const CONTEXT_SEPARATOR = "␟";

export type MovementSequence = {
  /** Source trajectory or replay id, for traceability. */
  id: string;
  tokens: MovementToken[];
};

export type MovementPrediction = {
  token: MovementToken;
  /** Interpolated stupid-backoff score (higher = more likely). */
  score: number;
  /** Length of the matched context that produced the top contribution. */
  matchedContextLength: number;
};

export type TrainMovementOptions = {
  /** Maximum context length (n-gram order). Defaults to 3. */
  order?: number;
  /** Backoff discount applied per dropped context token. Defaults to 0.4. */
  backoff?: number;
};

export type GenerateMovementOptions = {
  /** Hard cap on generated tokens (excludes the stop token). Defaults to 64. */
  maxSteps?: number;
};

export type MovementModelSnapshot = {
  version: 1;
  backendId: string;
  order: number;
  backoff: number;
  vocab: MovementToken[];
  /** contextKey -> { nextToken -> observationCount }. */
  grams: Record<string, Record<MovementToken, number>>;
};

/** A trained, serializable, deterministic movement model. */
export interface MovementModel {
  readonly backendId: string;
  readonly order: number;
  /** Ranked next-token candidates for a context (most-recent token last). */
  rank(context: MovementToken[]): MovementPrediction[];
  /** Highest-scoring next token, or undefined if the model is empty. */
  predictNext(context: MovementToken[]): MovementPrediction | undefined;
  /** Greedy roll-out from a prefix; stops at {@link MOVEMENT_END} or maxSteps. */
  generate(prefix: MovementToken[], options?: GenerateMovementOptions): MovementToken[];
  toJSON(): MovementModelSnapshot;
}

/** Pluggable backend that turns movement sequences into a {@link MovementModel}. */
export interface MovementModelBackend {
  readonly id: string;
  train(sequences: MovementSequence[], options?: TrainMovementOptions): MovementModel;
  /** Rehydrate a previously trained model from its snapshot. */
  load(snapshot: MovementModelSnapshot): MovementModel;
}

function contextKey(tokens: MovementToken[]): string {
  return tokens.join(CONTEXT_SEPARATOR);
}

/**
 * Deterministic variable-order Markov backend with stupid-backoff.
 *
 * Deterministic in every respect (no RNG, no clock): given the same sequences
 * and options it always yields byte-identical snapshots and predictions, so it
 * is a reliable stand-in for a real local model in tests and a stable baseline
 * to compare a real backend against.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov-backoff";

  train(sequences: MovementSequence[], options: TrainMovementOptions = {}): MovementModel {
    const order = Math.max(1, Math.floor(options.order ?? 3));
    const backoff = clampBackoff(options.backoff ?? 0.4);
    const grams: Record<string, Record<MovementToken, number>> = {};
    const vocab = new Set<MovementToken>();

    for (const sequence of sequences) {
      const padded = [
        ...Array<MovementToken>(order).fill(MOVEMENT_START),
        ...sequence.tokens,
        MOVEMENT_END,
      ];
      for (let i = order; i < padded.length; i += 1) {
        const next = padded[i];
        vocab.add(next);
        // Record this continuation under every context length 0..order.
        for (let ctxLen = 0; ctxLen <= order; ctxLen += 1) {
          const context = padded.slice(i - ctxLen, i);
          const key = contextKey(context);
          const table = (grams[key] ??= {});
          table[next] = (table[next] ?? 0) + 1;
        }
      }
    }

    return new MarkovMovementModel(this.id, order, backoff, grams, [...vocab].sort());
  }

  load(snapshot: MovementModelSnapshot): MovementModel {
    return new MarkovMovementModel(
      snapshot.backendId,
      snapshot.order,
      clampBackoff(snapshot.backoff),
      snapshot.grams,
      [...snapshot.vocab],
    );
  }
}

class MarkovMovementModel implements MovementModel {
  constructor(
    readonly backendId: string,
    readonly order: number,
    private readonly backoff: number,
    private readonly grams: Record<string, Record<MovementToken, number>>,
    private readonly vocab: MovementToken[],
  ) {}

  rank(context: MovementToken[]): MovementPrediction[] {
    // Left-pad short contexts with START so queries line up with the START
    // padding used during training — this lets an empty context condition on
    // the start-of-sequence position and reproduce a memorized first move.
    const trimmed =
      context.length >= this.order
        ? context.slice(-this.order)
        : [...Array<MovementToken>(this.order - context.length).fill(MOVEMENT_START), ...context];
    const scores = new Map<MovementToken, { score: number; matchedContextLength: number }>();

    // Stupid-backoff: blend every context length from longest to unigram,
    // discounting by `backoff` for each token we drop off the front.
    for (let ctxLen = trimmed.length; ctxLen >= 0; ctxLen -= 1) {
      const key = contextKey(ctxLen === 0 ? [] : trimmed.slice(trimmed.length - ctxLen));
      const table = this.grams[key];
      if (!table) {
        continue;
      }
      const total = Object.values(table).reduce((sum, count) => sum + count, 0);
      if (total === 0) {
        continue;
      }
      const weight = Math.pow(this.backoff, trimmed.length - ctxLen);
      for (const [token, count] of Object.entries(table)) {
        const contribution = weight * (count / total);
        const existing = scores.get(token);
        if (existing) {
          existing.score += contribution;
          existing.matchedContextLength = Math.max(existing.matchedContextLength, ctxLen);
        } else {
          scores.set(token, { score: contribution, matchedContextLength: ctxLen });
        }
      }
    }

    return [...scores.entries()]
      .map(([token, value]) => ({ token, score: value.score, matchedContextLength: value.matchedContextLength }))
      // Deterministic ordering: score desc, then longest match, then token asc.
      .sort((a, b) => {
        if (b.score !== a.score) {
          return b.score - a.score;
        }
        if (b.matchedContextLength !== a.matchedContextLength) {
          return b.matchedContextLength - a.matchedContextLength;
        }
        return a.token < b.token ? -1 : a.token > b.token ? 1 : 0;
      });
  }

  predictNext(context: MovementToken[]): MovementPrediction | undefined {
    return this.rank(context)[0];
  }

  generate(prefix: MovementToken[], options: GenerateMovementOptions = {}): MovementToken[] {
    const maxSteps = Math.max(0, Math.floor(options.maxSteps ?? 64));
    const generated: MovementToken[] = [];
    const context = [...prefix];
    for (let step = 0; step < maxSteps; step += 1) {
      const next = this.predictNext(context);
      if (!next || next.token === MOVEMENT_END) {
        break;
      }
      generated.push(next.token);
      context.push(next.token);
    }
    return generated;
  }

  toJSON(): MovementModelSnapshot {
    return {
      version: 1,
      backendId: this.backendId,
      order: this.order,
      backoff: this.backoff,
      vocab: [...this.vocab],
      grams: this.grams,
    };
  }
}

function clampBackoff(value: number): number {
  if (!Number.isFinite(value)) {
    return 0.4;
  }
  return Math.min(1, Math.max(0.01, value));
}

/**
 * Deterministically tokenize a single action into a movement token. The token
 * captures the tool plus a normalized summary/gesture so that structurally
 * identical movements share a token (enabling generalization) while distinct
 * movements stay distinct.
 */
export function movementTokenForAction(action: {
  tool: string;
  summary: string;
  metadata?: Record<string, unknown>;
}): MovementToken {
  const gesture = typeof action.metadata?.gesture === "string" ? action.metadata.gesture : undefined;
  const target = typeof action.metadata?.target === "string" ? action.metadata.target : undefined;
  const direction = typeof action.metadata?.direction === "string" ? action.metadata.direction : undefined;
  const descriptor = gesture
    ? [gesture, target, direction].filter((part): part is string => Boolean(part)).join(":")
    : normalizeSummary(action.summary);
  return `${normalizeSummary(action.tool)}#${descriptor}`;
}

function normalizeSummary(value: string): string {
  return value
    .trim()
    .toLowerCase()
    // Collapse volatile substrings (ids, numbers, timestamps) so related
    // movements that differ only in a variable share a token.
    .replace(/\b[0-9a-f]{8,}\b/g, "<id>")
    .replace(/\b\d+\b/g, "<n>")
    .replace(/\s+/g, " ");
}

/** Build a training sequence from a single trajectory's ordered actions. */
export function movementSequenceFromTrajectory(trajectory: TrajectorySpan): MovementSequence {
  const tokens = [...trajectory.actions]
    .sort((a, b) => a.ts - b.ts)
    .map((action) => movementTokenForAction(action));
  return { id: trajectory.id, tokens };
}

/** Build training sequences from a replay manifest's action timeline. */
export function movementSequenceFromReplay(replay: Pick<ReplayManifest, "sessionId" | "events">): MovementSequence {
  const tokens = [...replay.events]
    .filter((event): event is Extract<ReplayTimelineEvent, { kind: "action" }> => event.kind === "action")
    .sort((a, b) => a.ts - b.ts)
    .map((event) => movementTokenForAction({ tool: event.tool, summary: event.summary }));
  return { id: replay.sessionId, tokens };
}

export function movementSequencesFromReplays(
  replays: Array<Pick<ReplayManifest, "sessionId" | "events">>,
): MovementSequence[] {
  return replays.map((replay) => movementSequenceFromReplay(replay));
}
