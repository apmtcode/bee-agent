/**
 * Movement-policy model: the "post-train a local model to repeat and generalize
 * recorded movements" half of the local-movement learning subsystem
 * (standing objective #2, pieces c + d).
 *
 * The capture side (recorder, device/os/browser adapters, trajectory store,
 * replay manifest) already produces structured, replayable movement data. This
 * module closes the loop: it turns recorded movements into a training dataset,
 * learns a *policy* over them, and can then (c) repeat a recorded movement and
 * (d) generalize to a new-but-related movement it never saw verbatim.
 *
 * The learning backend is pluggable behind {@link MovementPolicyBackend} so a
 * real on-device small model can slot in later. The shipped implementation is a
 * deterministic Katz-style backoff Markov model — it needs no native deps and
 * no randomness, so it trains and infers identically in the cloud and on the
 * user's machine, and its tests are stable. Generalization comes from context
 * backoff: an unseen high-order context falls back to the longest suffix the
 * model *has* seen, so related prefixes still yield a sensible next move.
 */

import type { ReplayTimelineEvent } from "../../capture/replay.js";
import type { TrajectoryAction, TrajectorySpan } from "../../capture/trajectory.js";

/** A single normalized movement, e.g. `"device:tap:submit-button"`. */
export type MovementToken = string;

/** End-of-sequence sentinel appended during training so rollout knows to stop. */
export const MOVEMENT_END: MovementToken = "<END>";

/** Context-key separator (pipe: cannot appear in a slugged token). */
const CONTEXT_SEP = "|";

/** An ordered movement recorded from one trajectory / replay. */
export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
  /** Optional free-form label used by the synthetic generator / eval reports. */
  intent?: string;
};

/** The dataset a backend trains on. */
export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

/**
 * A trained, JSON-serializable policy. Kept as plain data so it can be
 * persisted next to other reviewed-export artifacts and reloaded without the
 * backend instance that produced it.
 */
export type MovementPolicyModel = {
  version: 1;
  backend: string;
  /** Maximum context order k (number of preceding tokens conditioned on). */
  order: number;
  vocabulary: MovementToken[];
  sequenceCount: number;
  tokenCount: number;
  /** context-key -> { nextToken -> count } for every order 0..k. */
  transitions: Record<string, Record<MovementToken, number>>;
};

/** One ranked candidate for the next movement. */
export type MovementPrediction = {
  token: MovementToken;
  probability: number;
  /** How many context tokens actually matched (k = full order, 0 = unigram). */
  backoffOrder: number;
};

export type TrainMovementPolicyOptions = {
  /** Max context order; higher = more literal reproduction, less generalization. */
  order?: number;
};

export type GenerateMovementOptions = {
  /** Hard cap on rollout length (excluding the END sentinel). */
  maxSteps?: number;
  /** Seed context already performed; rollout continues from here. */
  seed?: MovementToken[];
};

/**
 * Pluggable learning backend. A real backend (a small on-device transformer,
 * an MLX/llama.cpp policy, …) implements the same three verbs; the rest of the
 * subsystem (eval harness, exporter, CLI) depends only on this interface.
 */
export interface MovementPolicyBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: TrainMovementPolicyOptions): MovementPolicyModel;
  /** Ranked next-movement candidates for `context`, best first. */
  predictNext(model: MovementPolicyModel, context: MovementToken[]): MovementPrediction[];
  /** Roll out a full movement (used to "repeat" or "generalize" a movement). */
  generate(model: MovementPolicyModel, options?: GenerateMovementOptions): MovementToken[];
}

const DEFAULT_ORDER = 3;
const DEFAULT_MAX_STEPS = 256;

/**
 * Deterministic backoff-Markov backend. Trains transition counts for every
 * context order 0..k; prediction and rollout use the longest context suffix
 * with observed continuations (Katz backoff), which is what lets it generalize
 * to novel-but-related prefixes instead of only replaying memorized ones.
 */
export class MarkovMovementBackend implements MovementPolicyBackend {
  readonly id = "markov-backoff";

  train(dataset: MovementDataset, options: TrainMovementPolicyOptions = {}): MovementPolicyModel {
    const order = Math.max(0, Math.floor(options.order ?? DEFAULT_ORDER));
    const transitions: Record<string, Record<MovementToken, number>> = {};
    const vocabulary = new Set<MovementToken>();
    let tokenCount = 0;

    for (const sequence of dataset.sequences) {
      // Terminate every training sequence so rollout can learn when to stop.
      const tokens = [...sequence.tokens, MOVEMENT_END];
      for (let i = 0; i < tokens.length; i += 1) {
        const next = tokens[i]!;
        if (next !== MOVEMENT_END) {
          vocabulary.add(next);
        }
        tokenCount += 1;
        // Record this position under every context length 0..order.
        for (let o = 0; o <= order; o += 1) {
          if (i - o < 0) {
            break;
          }
          const context = tokens.slice(i - o, i);
          const key = contextKey(context);
          const row = (transitions[key] ??= {});
          row[next] = (row[next] ?? 0) + 1;
        }
      }
    }

    return {
      version: 1,
      backend: this.id,
      order,
      vocabulary: [...vocabulary].sort(),
      sequenceCount: dataset.sequences.length,
      tokenCount,
      transitions,
    };
  }

  predictNext(model: MovementPolicyModel, context: MovementToken[]): MovementPrediction[] {
    // Try the longest allowed suffix first, backing off until a context with
    // observed continuations is found (order 0 / unigram always exists once
    // trained on any data).
    const maxLen = Math.min(model.order, context.length);
    for (let o = maxLen; o >= 0; o -= 1) {
      const suffix = context.slice(context.length - o);
      const row = model.transitions[contextKey(suffix)];
      if (!row) {
        continue;
      }
      const total = Object.values(row).reduce((sum, count) => sum + count, 0);
      if (total === 0) {
        continue;
      }
      return Object.entries(row)
        .map(([token, count]) => ({ token, probability: count / total, backoffOrder: o }))
        // Deterministic ordering: higher probability first, then token order.
        .sort((a, b) => (b.probability - a.probability) || compareTokens(a.token, b.token));
    }
    return [];
  }

  generate(model: MovementPolicyModel, options: GenerateMovementOptions = {}): MovementToken[] {
    const maxSteps = options.maxSteps ?? DEFAULT_MAX_STEPS;
    const output: MovementToken[] = [...(options.seed ?? [])];
    for (let step = 0; step < maxSteps; step += 1) {
      const [best] = this.predictNext(model, output);
      if (!best || best.token === MOVEMENT_END) {
        break;
      }
      output.push(best.token);
    }
    return output;
  }
}

/** Compact key for a context window; empty context maps to the empty string. */
function contextKey(context: MovementToken[]): string {
  return context.join(CONTEXT_SEP);
}

function compareTokens(a: MovementToken, b: MovementToken): number {
  return a < b ? -1 : a > b ? 1 : 0;
}

/** Encode an action into a stable movement token. */
export function movementTokenFromAction(action: Pick<TrajectoryAction, "tool" | "summary" | "metadata">): MovementToken {
  const gesture = typeof action.metadata?.gesture === "string" ? action.metadata.gesture : "act";
  const target =
    (typeof action.metadata?.target === "string" && action.metadata.target) ||
    (typeof action.metadata?.direction === "string" && action.metadata.direction) ||
    slug(action.summary);
  return `${slug(action.tool)}:${slug(gesture)}:${slug(target)}`;
}

/** Derive a movement sequence from a trajectory span (actions in time order). */
export function movementSequenceFromTrajectory(trajectory: TrajectorySpan): MovementSequence {
  const tokens = [...trajectory.actions]
    .sort((a, b) => a.ts - b.ts)
    .map((action) => movementTokenFromAction(action));
  return {
    id: trajectory.id,
    tokens,
    ...(trajectory.outcome?.summary ? { intent: trajectory.outcome.summary } : {}),
  };
}

/** Derive a movement sequence from replay-timeline action events. */
export function movementSequenceFromReplayEvents(
  id: string,
  events: ReplayTimelineEvent[],
): MovementSequence {
  const tokens = events
    .filter((event): event is Extract<ReplayTimelineEvent, { kind: "action" }> => event.kind === "action")
    .sort((a, b) => a.ts - b.ts)
    .map((event) => `${slug(event.tool)}:act:${slug(event.summary)}`);
  return { id, tokens };
}

function slug(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "") || "unknown";
}
