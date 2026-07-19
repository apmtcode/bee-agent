import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * In-process movement-policy learning + inference.
 *
 * This module closes the loop of the local-movement learning subsystem
 * (standing objective #2 c/d): it takes recorded movement/action sequences,
 * trains a small *local* model on them, and can then (c) repeat the recorded
 * movements and (d) generalize to new-but-related movements — all in-process,
 * with no external tooling, so it is fully exercisable in the cloud on
 * synthetic data.
 *
 * The learning backend is pluggable: `MovementPolicyBackend` is the seam, and
 * `MarkovMovementBackend` is a deterministic default that requires no native
 * deps. A real on-device small model can be dropped in behind the same
 * interface without touching the exporter, eval harness, or call sites.
 */

/** A single movement token, e.g. `"action:mouse:click"` or `"obs:screen:menu-open"`. */
export type MovementToken = string;

/** Marks the start of a sequence so the model can predict the first move. */
export const MOVEMENT_START: MovementToken = "<start>";
/** Marks the end of a sequence so a rollout knows when to stop. */
export const MOVEMENT_END: MovementToken = "<end>";

/** An ordered movement sequence learned from / evaluated against. */
export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

export type MovementDataset = {
  sequences: MovementSequence[];
};

export type MovementTokenizerOptions = {
  /** Include observation steps as tokens (default true). Actions are always included. */
  includeObservations?: boolean;
  /** Coarsen a summary into a stable gesture verb (default: first lowercased word). */
  gestureOf?: (summary: string) => string;
};

export type MovementTrainingOptions = {
  /** Maximum context length the model conditions on (n-gram order). Default 2. */
  order?: number;
};

export type RankedMovement = {
  token: MovementToken;
  probability: number;
};

export type MovementPrediction = {
  token: MovementToken;
  probability: number;
  /** All candidates ranked by probability then token (deterministic). */
  ranked: RankedMovement[];
  /** Effective context length used after backoff (0 = unigram prior). */
  backoffOrder: number;
};

export type SerializedMovementPolicy = {
  version: 1;
  backendId: string;
  order: number;
  vocabulary: MovementToken[];
  /** context-key (`\n`-joined) -> next-token -> count. */
  counts: Record<string, Record<MovementToken, number>>;
};

export interface TrainedMovementPolicy {
  readonly backendId: string;
  readonly order: number;
  /** Predict the most likely next movement given a (possibly novel) context. */
  predictNext(context: MovementToken[]): MovementPrediction | undefined;
  /** Greedily unroll a full movement sequence from a seed until END/maxLength. */
  rollout(seed: MovementToken[], options?: { maxLength?: number }): MovementToken[];
  toJSON(): SerializedMovementPolicy;
}

export interface MovementPolicyBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainingOptions): TrainedMovementPolicy;
}

const CONTEXT_SEP = "\n";

function defaultGesture(summary: string): string {
  const first = summary.trim().toLowerCase().split(/\s+/u)[0] ?? "";
  return first.replace(/[^a-z0-9-]/gu, "") || "step";
}

/**
 * Turn a recorded trajectory into an ordered movement-token sequence. Uses the
 * reviewed/redacted view when present so training never sees unreviewed data.
 */
export function tokenizeTrajectory(
  trajectory: TrajectorySpan,
  options: MovementTokenizerOptions = {},
): MovementSequence {
  const includeObservations = options.includeObservations ?? true;
  const gestureOf = options.gestureOf ?? defaultGesture;

  const observations = trajectory.review?.redactedObservations
    ? trajectory.review.redactedObservations.map((o) => ({ ts: o.ts, source: o.source, summary: o.summary }))
    : trajectory.observations.map((o) => ({ ts: o.ts, source: o.source, summary: o.summary }));
  const actions = trajectory.review?.redactedActions
    ? trajectory.review.redactedActions.map((a) => ({ ts: a.ts, tool: a.tool, summary: a.summary }))
    : trajectory.actions.map((a) => ({ ts: a.ts, tool: a.tool, summary: a.summary }));

  const steps: Array<{ ts: number; token: MovementToken }> = [];
  if (includeObservations) {
    for (const o of observations) {
      steps.push({ ts: o.ts, token: `obs:${o.source}:${gestureOf(o.summary)}` });
    }
  }
  for (const a of actions) {
    steps.push({ ts: a.ts, token: `action:${a.tool}:${gestureOf(a.summary)}` });
  }
  // Stable order: by timestamp, observations before actions on ties (obs < action).
  steps.sort((x, y) => (x.ts !== y.ts ? x.ts - y.ts : rank(x.token) - rank(y.token)));

  return { id: trajectory.id, tokens: steps.map((s) => s.token) };
}

function rank(token: MovementToken): number {
  return token.startsWith("obs:") ? 0 : 1;
}

export function buildMovementDataset(
  trajectories: TrajectorySpan[],
  options?: MovementTokenizerOptions,
): MovementDataset {
  return {
    sequences: trajectories
      .map((trajectory) => tokenizeTrajectory(trajectory, options))
      .filter((sequence) => sequence.tokens.length > 0),
  };
}

class MarkovMovementPolicy implements TrainedMovementPolicy {
  constructor(
    readonly backendId: string,
    readonly order: number,
    private readonly vocabulary: MovementToken[],
    private readonly counts: Map<string, Map<MovementToken, number>>,
  ) {}

  predictNext(context: MovementToken[]): MovementPrediction | undefined {
    for (let len = Math.min(this.order, context.length); len >= 0; len--) {
      const key = keyFor(context.slice(context.length - len));
      const table = this.counts.get(key);
      if (!table || table.size === 0) {
        continue;
      }
      const ranked = rankTable(table);
      const best = ranked[0];
      if (!best) {
        continue;
      }
      return { token: best.token, probability: best.probability, ranked, backoffOrder: len };
    }
    return undefined;
  }

  rollout(seed: MovementToken[], options: { maxLength?: number } = {}): MovementToken[] {
    const maxLength = options.maxLength ?? 128;
    const context = [MOVEMENT_START, ...seed];
    const produced: MovementToken[] = [];
    while (produced.length < maxLength) {
      const prediction = this.predictNext(context);
      if (!prediction || prediction.token === MOVEMENT_END) {
        break;
      }
      produced.push(prediction.token);
      context.push(prediction.token);
    }
    return produced;
  }

  toJSON(): SerializedMovementPolicy {
    const counts: Record<string, Record<MovementToken, number>> = {};
    for (const [key, table] of this.counts) {
      counts[key] = Object.fromEntries(table);
    }
    return {
      version: 1,
      backendId: this.backendId,
      order: this.order,
      vocabulary: [...this.vocabulary],
      counts,
    };
  }

  static fromJSON(serialized: SerializedMovementPolicy): MarkovMovementPolicy {
    const counts = new Map<string, Map<MovementToken, number>>();
    for (const [key, table] of Object.entries(serialized.counts)) {
      counts.set(key, new Map(Object.entries(table)));
    }
    return new MarkovMovementPolicy(serialized.backendId, serialized.order, [...serialized.vocabulary], counts);
  }
}

function keyFor(context: MovementToken[]): string {
  return context.join(CONTEXT_SEP);
}

function rankTable(table: Map<MovementToken, number>): RankedMovement[] {
  let total = 0;
  for (const count of table.values()) {
    total += count;
  }
  return [...table.entries()]
    .map(([token, count]) => ({ token, probability: total === 0 ? 0 : count / total }))
    .sort((a, b) => (a.probability !== b.probability ? b.probability - a.probability : a.token < b.token ? -1 : 1));
}

/**
 * Deterministic default backend: an order-k Markov model with stupid-backoff.
 * Learns n-gram transition counts over movement tokens (with START/END
 * sentinels), so it reproduces recorded sequences exactly and generalizes to
 * novel contexts by backing off to shorter, seen contexts.
 */
export class MarkovMovementBackend implements MovementPolicyBackend {
  readonly id = "markov";

  train(dataset: MovementDataset, options: MovementTrainingOptions = {}): TrainedMovementPolicy {
    const order = Math.max(1, options.order ?? 2);
    const counts = new Map<string, Map<MovementToken, number>>();
    const vocabulary = new Set<MovementToken>();

    for (const sequence of dataset.sequences) {
      const tokens = [MOVEMENT_START, ...sequence.tokens, MOVEMENT_END];
      for (const token of tokens) {
        vocabulary.add(token);
      }
      for (let i = 1; i < tokens.length; i++) {
        const next = tokens[i]!;
        for (let len = 0; len <= order; len++) {
          if (i - len < 0) {
            break;
          }
          const context = tokens.slice(i - len, i);
          const key = keyFor(context);
          let table = counts.get(key);
          if (!table) {
            table = new Map();
            counts.set(key, table);
          }
          table.set(next, (table.get(next) ?? 0) + 1);
        }
      }
    }

    vocabulary.delete(MOVEMENT_START);
    return new MarkovMovementPolicy(this.id, order, [...vocabulary].sort(), counts);
  }
}

export function deserializeMovementPolicy(serialized: SerializedMovementPolicy): TrainedMovementPolicy {
  if (serialized.backendId === "markov" || serialized.version === 1) {
    return MarkovMovementPolicy.fromJSON(serialized);
  }
  throw new Error(`unknown movement policy backend: ${serialized.backendId}`);
}

export type MovementEvalOptions = {
  topK?: number;
  /** Context length used for teacher-forced prediction (defaults to policy.order). */
  order?: number;
  /** Max rollout length when checking perfect replay (default: 2x longest held-out seq). */
  maxRolloutLength?: number;
};

export type MovementEvalResult = {
  sequenceCount: number;
  /** Number of (context -> next) prediction points scored. */
  predictionCount: number;
  /** Top-1 next-token accuracy under teacher forcing. */
  nextTokenAccuracy: number;
  /** Top-K next-token accuracy under teacher forcing. */
  topKAccuracy: number;
  /** Mean reciprocal rank of the true next token in the ranked candidates. */
  meanReciprocalRank: number;
  /** Fraction of held-out sequences a greedy rollout reproduces exactly from START. */
  perfectReplayRate: number;
};

/**
 * Generalization eval harness: measures replay fidelity + next-move accuracy on
 * held-out (ideally related-but-unseen) movement sequences. Use it to quantify
 * how well a trained policy generalizes across runs.
 */
export function evaluateMovementPolicy(
  policy: TrainedMovementPolicy,
  heldOut: MovementDataset,
  options: MovementEvalOptions = {},
): MovementEvalResult {
  const topK = Math.max(1, options.topK ?? 3);
  const order = options.order ?? policy.order;

  let predictionCount = 0;
  let top1 = 0;
  let topk = 0;
  let reciprocalRankSum = 0;
  let perfectReplays = 0;
  let longest = 0;

  for (const sequence of heldOut.sequences) {
    longest = Math.max(longest, sequence.tokens.length);
    const tokens = [MOVEMENT_START, ...sequence.tokens, MOVEMENT_END];
    for (let i = 1; i < tokens.length; i++) {
      const expected = tokens[i]!;
      const contextStart = Math.max(0, i - order);
      const prediction = policy.predictNext(tokens.slice(contextStart, i));
      predictionCount++;
      if (!prediction) {
        continue;
      }
      if (prediction.token === expected) {
        top1++;
      }
      const rankIndex = prediction.ranked.findIndex((candidate) => candidate.token === expected);
      if (rankIndex >= 0) {
        reciprocalRankSum += 1 / (rankIndex + 1);
        if (rankIndex < topK) {
          topk++;
        }
      }
    }
  }

  const maxRolloutLength = options.maxRolloutLength ?? Math.max(1, longest * 2);
  for (const sequence of heldOut.sequences) {
    const rolled = policy.rollout([], { maxLength: maxRolloutLength });
    if (arraysEqual(rolled, sequence.tokens)) {
      perfectReplays++;
    }
  }

  const sequenceCount = heldOut.sequences.length;
  return {
    sequenceCount,
    predictionCount,
    nextTokenAccuracy: predictionCount === 0 ? 0 : top1 / predictionCount,
    topKAccuracy: predictionCount === 0 ? 0 : topk / predictionCount,
    meanReciprocalRank: predictionCount === 0 ? 0 : reciprocalRankSum / predictionCount,
    perfectReplayRate: sequenceCount === 0 ? 0 : perfectReplays / sequenceCount,
  };
}

function arraysEqual(a: MovementToken[], b: MovementToken[]): boolean {
  if (a.length !== b.length) {
    return false;
  }
  return a.every((value, index) => value === b[index]);
}

/**
 * Deterministic synthetic movement-stream generator. Produces families of
 * related sequences from a small grammar so capture -> dataset -> train ->
 * replay -> generalize can be validated without any real OS input. The `seed`
 * makes it reproducible (no `Math.random`), so tests are stable across runs.
 */
export function generateSyntheticMovementDataset(params: {
  seed: number;
  sequenceCount: number;
  /** Token vocabulary to draw movements from (default: a small mouse/keyboard set). */
  vocabulary?: MovementToken[];
  minLength?: number;
  maxLength?: number;
}): MovementDataset {
  const vocabulary = params.vocabulary ?? [
    "action:mouse:move",
    "action:mouse:click",
    "action:mouse:drag",
    "action:keyboard:type",
    "action:keyboard:hotkey",
    "action:window:focus",
  ];
  const minLength = params.minLength ?? 3;
  const maxLength = params.maxLength ?? 8;
  const rng = makeRng(params.seed);

  const sequences: MovementSequence[] = [];
  for (let index = 0; index < params.sequenceCount; index++) {
    const length = minLength + Math.floor(rng() * (maxLength - minLength + 1));
    const tokens: MovementToken[] = [];
    // First-order structure: the next token depends on the previous one, so a
    // Markov model can actually learn something generalizable rather than noise.
    let prev = Math.floor(rng() * vocabulary.length);
    for (let step = 0; step < length; step++) {
      tokens.push(vocabulary[prev]!);
      const jump = rng() < 0.7 ? 1 : Math.floor(rng() * vocabulary.length);
      prev = (prev + jump) % vocabulary.length;
    }
    sequences.push({ id: `synthetic-${params.seed}-${index}`, tokens });
  }
  return { sequences };
}

/** Small deterministic PRNG (mulberry32) — reproducible synthetic streams. */
function makeRng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state |= 0;
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
