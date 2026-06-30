import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

/**
 * Movement-policy model layer.
 *
 * Objective #2 (c)+(d): post-train a *local* model on the recorded movement
 * dataset so the agent can (c) repeat recorded movements and (d) generalize to
 * new-but-related movements. The on-device path (mlx/axolotl) lives in
 * `runner.ts`, but that backend cannot run in the cloud and is not testable
 * here. This module adds a **pluggable, dependency-free, deterministic** backend
 * that genuinely learns from a dataset and predicts/regenerates movement
 * sequences — so the capture->dataset->train->infer loop can be exercised end to
 * end in CI with synthetic data, and a real on-device backend can be dropped in
 * behind the same `MovementPolicyBackend` interface.
 */

/** A single movement, tokenized to a stable string (e.g. `"click::submit"`). */
export type MovementToken = string;

/** Sentinel appended after each training sequence so the model learns where a
 * trajectory naturally ends — lets `generate()` stop instead of looping. It uses
 * the reserved tool name `<end>`, which `tokenizeMovement` never produces. */
export const MOVEMENT_END_TOKEN: MovementToken = "<end>::<end>";

/** One ordered movement sequence extracted from a reviewed trajectory/replay. */
export type MovementExample = {
  trajectoryId: string;
  tokens: MovementToken[];
};

export type MovementPrediction = {
  /** Predicted next token, or `undefined` when the model has no information. */
  token: MovementToken | undefined;
  /** Normalized probability of `token` within the matched context (0..1). */
  probability: number;
  /** Context order (suffix length) the prediction was drawn from. */
  order: number;
  /**
   * Where the prediction came from:
   * - `exact`   — matched the full available context (≤ maxOrder).
   * - `backoff` — matched a shorter suffix (the generalization path).
   * - `global`  — order-0 unigram fallback.
   * - `none`    — no data at all.
   */
  source: "exact" | "backoff" | "global" | "none";
};

export type MovementTrainingOptions = {
  /** Maximum context length the model conditions on. Default 3. */
  maxOrder?: number;
};

export type SerializedMovementGram = {
  context: MovementToken[];
  next: MovementToken;
  count: number;
};

export type SerializedMovementPolicy = {
  version: 1;
  backendId: string;
  maxOrder: number;
  vocabulary: MovementToken[];
  grams: SerializedMovementGram[];
};

/** A trained, queryable movement policy. */
export interface MovementPolicyModel {
  readonly backendId: string;
  readonly maxOrder: number;
  /** Tokens the model has observed (excluding the end sentinel). */
  vocabulary(): MovementToken[];
  /** Most-likely next token given the recent context (longest-suffix backoff). */
  predict(context: MovementToken[]): MovementPrediction;
  /** Top-`k` next-token candidates, highest probability first. */
  predictTopK(context: MovementToken[], k: number): MovementPrediction[];
  /**
   * Regenerate a movement sequence from a seed prefix, stopping at the learned
   * end sentinel or after `maxSteps`. This is how the agent "repeats" a recorded
   * movement: seed with the first movement and roll forward.
   */
  generate(seed: MovementToken[], maxSteps?: number): MovementToken[];
  serialize(): SerializedMovementPolicy;
}

/** Pluggable training backend. Swap in a real on-device model behind this. */
export interface MovementPolicyBackend {
  readonly id: string;
  train(examples: MovementExample[], options?: MovementTrainingOptions): MovementPolicyModel;
  load(serialized: SerializedMovementPolicy): MovementPolicyModel;
}

const DEFAULT_MAX_ORDER = 3;

function normalizeSummary(summary: string): string {
  return summary.trim().replace(/\s+/g, " ").toLowerCase();
}

/** Tokenize one recorded action (a "movement") into a stable token. */
export function tokenizeMovement(action: Pick<TrajectoryAction, "tool" | "summary">): MovementToken {
  return `${action.tool.trim().toLowerCase()}::${normalizeSummary(action.summary)}`;
}

/** Extract an ordered movement example from a trajectory's actions (by ts). */
export function extractMovementExample(trajectory: TrajectorySpan): MovementExample {
  const actions = trajectory.review?.redactedActions ?? trajectory.actions;
  const tokens = [...actions]
    .sort((a, b) => a.ts - b.ts)
    .map((action) => tokenizeMovement(action));
  return { trajectoryId: trajectory.id, tokens };
}

/** Extract movement examples from a replay manifest's `action` timeline events. */
export function extractMovementExamplesFromReplay(replay: ReplayManifest): MovementExample[] {
  const byTrajectory = new Map<string, { ts: number; tool: string; summary: string }[]>();
  for (const event of replay.events) {
    if (event.kind !== "action") {
      continue;
    }
    const actionEvent: Extract<ReplayTimelineEvent, { kind: "action" }> = event;
    const bucket = byTrajectory.get(actionEvent.trajectoryId) ?? [];
    bucket.push({ ts: actionEvent.ts, tool: actionEvent.tool, summary: actionEvent.summary });
    byTrajectory.set(actionEvent.trajectoryId, bucket);
  }
  return [...byTrajectory.entries()].map(([trajectoryId, actions]) => ({
    trajectoryId,
    tokens: actions.sort((a, b) => a.ts - b.ts).map((action) => tokenizeMovement(action)),
  }));
}

/** Unambiguous, printable, collision-proof map key for a context array. */
function contextKey(context: MovementToken[]): string {
  return JSON.stringify(context);
}

type GramEntry = { context: MovementToken[]; dist: Map<MovementToken, number> };

class NgramMovementPolicyModel implements MovementPolicyModel {
  readonly backendId: string;
  readonly maxOrder: number;
  /** contextKey -> { original context tokens, nextToken -> count }. */
  private readonly grams: Map<string, GramEntry>;
  private readonly vocab: Set<MovementToken>;

  constructor(params: {
    backendId: string;
    maxOrder: number;
    grams: Map<string, GramEntry>;
    vocab: Set<MovementToken>;
  }) {
    this.backendId = params.backendId;
    this.maxOrder = params.maxOrder;
    this.grams = params.grams;
    this.vocab = params.vocab;
  }

  vocabulary(): MovementToken[] {
    return [...this.vocab].filter((token) => token !== MOVEMENT_END_TOKEN).sort();
  }

  predict(context: MovementToken[]): MovementPrediction {
    return this.predictTopK(context, 1)[0] ?? { token: undefined, probability: 0, order: 0, source: "none" };
  }

  predictTopK(context: MovementToken[], k: number): MovementPrediction[] {
    const requested = Math.max(1, Math.trunc(k));
    const longest = Math.min(this.maxOrder, context.length);
    for (let order = longest; order >= 0; order -= 1) {
      const suffix = order === 0 ? [] : context.slice(context.length - order);
      const entry = this.grams.get(contextKey(suffix));
      if (!entry || entry.dist.size === 0) {
        continue;
      }
      const total = [...entry.dist.values()].reduce((sum, count) => sum + count, 0);
      const source: MovementPrediction["source"] =
        order === longest ? "exact" : order === 0 ? "global" : "backoff";
      return [...entry.dist.entries()]
        .sort((a, b) => (b[1] !== a[1] ? b[1] - a[1] : a[0].localeCompare(b[0])))
        .slice(0, requested)
        .map(([token, count]) => ({ token, probability: count / total, order, source }));
    }
    return [{ token: undefined, probability: 0, order: 0, source: "none" }];
  }

  generate(seed: MovementToken[], maxSteps = 256): MovementToken[] {
    const out = [...seed];
    const limit = Math.max(0, Math.trunc(maxSteps));
    for (let step = 0; step < limit; step += 1) {
      const context = out.slice(Math.max(0, out.length - this.maxOrder));
      const { token } = this.predict(context);
      if (token === undefined || token === MOVEMENT_END_TOKEN) {
        break;
      }
      out.push(token);
    }
    return out;
  }

  serialize(): SerializedMovementPolicy {
    const grams: SerializedMovementGram[] = [];
    for (const entry of this.grams.values()) {
      for (const [next, count] of entry.dist.entries()) {
        grams.push({ context: [...entry.context], next, count });
      }
    }
    grams.sort((a, b) => contextKey(a.context).localeCompare(contextKey(b.context)) || a.next.localeCompare(b.next));
    return {
      version: 1,
      backendId: this.backendId,
      maxOrder: this.maxOrder,
      vocabulary: this.vocabulary(),
      grams,
    };
  }
}

/**
 * Deterministic, dependency-free local backend: a variable-order n-gram model
 * with longest-suffix backoff. It memorizes recorded movement sequences (so it
 * repeats them exactly) and generalizes to unseen contexts by backing off to
 * shorter suffixes that *were* observed. No randomness, no external runtime —
 * the canonical CI/cloud stand-in for the on-device model.
 */
export class NgramMovementPolicyBackend implements MovementPolicyBackend {
  readonly id = "ngram-backoff";

  train(examples: MovementExample[], options?: MovementTrainingOptions): MovementPolicyModel {
    const maxOrder = Math.max(0, Math.trunc(options?.maxOrder ?? DEFAULT_MAX_ORDER));
    const grams = new Map<string, GramEntry>();
    const vocab = new Set<MovementToken>();

    const bump = (context: MovementToken[], next: MovementToken): void => {
      const key = contextKey(context);
      const entry = grams.get(key) ?? { context: [...context], dist: new Map<MovementToken, number>() };
      entry.dist.set(next, (entry.dist.get(next) ?? 0) + 1);
      grams.set(key, entry);
    };

    for (const example of examples) {
      if (example.tokens.length === 0) {
        continue;
      }
      const sequence = [...example.tokens, MOVEMENT_END_TOKEN];
      for (const token of example.tokens) {
        vocab.add(token);
      }
      for (let i = 0; i < sequence.length; i += 1) {
        const next = sequence[i]!;
        for (let order = 0; order <= maxOrder && i - order >= 0; order += 1) {
          bump(sequence.slice(i - order, i), next);
        }
      }
    }

    return new NgramMovementPolicyModel({ backendId: this.id, maxOrder, grams, vocab });
  }

  load(serialized: SerializedMovementPolicy): MovementPolicyModel {
    const grams = new Map<string, GramEntry>();
    const vocab = new Set<MovementToken>(serialized.vocabulary);
    for (const gram of serialized.grams) {
      const key = contextKey(gram.context);
      const entry = grams.get(key) ?? { context: [...gram.context], dist: new Map<MovementToken, number>() };
      entry.dist.set(gram.next, gram.count);
      grams.set(key, entry);
      if (gram.next !== MOVEMENT_END_TOKEN) {
        vocab.add(gram.next);
      }
    }
    return new NgramMovementPolicyModel({
      backendId: serialized.backendId,
      maxOrder: serialized.maxOrder,
      grams,
      vocab,
    });
  }
}
