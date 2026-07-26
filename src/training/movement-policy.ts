import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";
import type { ReplayTimelineEvent } from "../capture/replay.js";

/**
 * Movement-policy learning: the in-process, cloud-safe half of standing
 * objective #2(d) — post-train a *local* model on recorded movement
 * trajectories so it can (a) repeat the recorded movements and (b) generalize
 * to new-but-related movements.
 *
 * The real on-device training (mlx/axolotl) is described by
 * {@link ./runner.ts LocalAppleSiliconTrainingRunner}; that executes only when
 * the user runs bee-agent on their own machine. This module provides the
 * pluggable {@link MovementPolicyBackend} seam plus a deterministic reference
 * backend ({@link NgramMovementPolicyBackend}) so the full learn → repeat →
 * generalize loop can be trained and evaluated in the cloud/CI against
 * synthetic movement streams, with zero access to a real OS.
 */

/** A single movement, tokenized to a stable string (default: the tool name). */
export type MovementToken = string;

/** An ordered movement sequence extracted from one trajectory / episode. */
export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

/** A dataset of movement sequences the backend trains on. */
export type MovementDataset = {
  sequences: MovementSequence[];
};

/** A ranked next-movement candidate. */
export type MovementCandidate = {
  token: MovementToken;
  probability: number;
};

/** The result of predicting the next movement given a context prefix. */
export type MovementPrediction = {
  /** Most likely next movement, or `undefined` for an empty/untrained model. */
  token: MovementToken | undefined;
  /** Probability of {@link token} within the matched context. */
  probability: number;
  /** How many prior tokens were actually matched (backoff depth; 0 = unigram). */
  contextOrder: number;
  /**
   * `true` when the full requested context was NOT found and the model backed
   * off to a shorter context — i.e. this is a *generalized* prediction rather
   * than a verbatim replay of a seen transition.
   */
  fromBackoff: boolean;
  /** All candidates, most likely first (deterministic tie-break by token). */
  ranked: MovementCandidate[];
};

/** Options controlling how a backend is trained. */
export type TrainMovementPolicyOptions = {
  /** Maximum context length (Markov order). Defaults to 3. */
  maxOrder?: number;
};

/**
 * Pluggable local-model backend. Swap {@link NgramMovementPolicyBackend} for a
 * real small on-device model (e.g. an mlx-trained policy) without touching
 * callers: the dataset in, the {@link MovementPrediction} out, are the contract.
 */
export type MovementPolicyBackend<Model> = {
  readonly id: string;
  train(dataset: MovementDataset, options?: TrainMovementPolicyOptions): Model;
  predict(model: Model, context: MovementToken[]): MovementPrediction;
  /** Whether the full-order context for `context` was seen during training. */
  hasContext(model: Model, context: MovementToken[]): boolean;
};

const CONTEXT_SEPARATOR = "";

function defaultActionTokenizer(action: TrajectoryAction): MovementToken {
  return action.tool;
}

/**
 * Extract an ordered movement sequence from a trajectory's recorded actions.
 * Honors a review redaction (uses `redactedActions` when present) so only
 * reviewed movements ever reach a training dataset.
 */
export function movementSequenceFromTrajectory(
  trajectory: TrajectorySpan,
  tokenize: (action: TrajectoryAction) => MovementToken = defaultActionTokenizer,
): MovementSequence {
  const source: TrajectoryAction[] = trajectory.review?.redactedActions
    ? trajectory.review.redactedActions.map((action) => ({
        kind: "action",
        ts: action.ts,
        tool: action.tool,
        summary: action.summary,
      }))
    : trajectory.actions;

  const tokens = [...source]
    .sort((a, b) => a.ts - b.ts)
    .map((action) => tokenize(action));

  return { id: trajectory.id, tokens };
}

/** Build a movement dataset from a set of trajectories. Empty sequences drop. */
export function buildMovementDataset(
  trajectories: TrajectorySpan[],
  tokenize: (action: TrajectoryAction) => MovementToken = defaultActionTokenizer,
): MovementDataset {
  const sequences = trajectories
    .map((trajectory) => movementSequenceFromTrajectory(trajectory, tokenize))
    .filter((sequence) => sequence.tokens.length > 0);
  return { sequences };
}

/** Build a movement sequence from a replay manifest's ordered action events. */
export function movementSequenceFromReplay(
  id: string,
  events: ReplayTimelineEvent[],
): MovementSequence {
  const tokens = events
    .filter((event): event is Extract<ReplayTimelineEvent, { kind: "action" }> => event.kind === "action")
    .map((event) => event.tool);
  return { id, tokens };
}

/** A serializable, variable-order Markov movement model. */
export type NgramMovementModel = {
  backendId: "ngram";
  maxOrder: number;
  /** counts[k] maps a k-token context key to next-token counts (k = 1..maxOrder). */
  counts: Array<Record<string, Record<string, number>>>;
  /** Order-0 next-token frequencies (the backoff floor). */
  unigram: Record<string, number>;
  totalTokens: number;
  vocab: string[];
};

function contextKey(tokens: MovementToken[]): string {
  return tokens.join(CONTEXT_SEPARATOR);
}

function argmax(counts: Record<string, number>): MovementCandidate | undefined {
  const total = Object.values(counts).reduce((sum, count) => sum + count, 0);
  if (total === 0) {
    return undefined;
  }
  const ranked = rankCounts(counts, total);
  return ranked[0];
}

function rankCounts(counts: Record<string, number>, total: number): MovementCandidate[] {
  return Object.entries(counts)
    .map(([token, count]) => ({ token, probability: count / total }))
    .sort((a, b) => {
      if (b.probability !== a.probability) {
        return b.probability - a.probability;
      }
      return a.token < b.token ? -1 : a.token > b.token ? 1 : 0;
    });
}

/**
 * Deterministic reference backend: a variable-order Markov chain with
 * stupid-backoff. It reproduces recorded movements exactly when the full-order
 * context was seen (repeat), and generalizes by backing off to shorter shared
 * contexts — and finally to the global next-move frequency — when it wasn't.
 */
export class NgramMovementPolicyBackend implements MovementPolicyBackend<NgramMovementModel> {
  readonly id = "ngram";

  train(dataset: MovementDataset, options?: TrainMovementPolicyOptions): NgramMovementModel {
    const maxOrder = Math.max(1, Math.floor(options?.maxOrder ?? 3));
    const counts: Array<Record<string, Record<string, number>>> = Array.from({ length: maxOrder + 1 }, () => ({}));
    const unigram: Record<string, number> = {};
    const vocab = new Set<string>();
    let totalTokens = 0;

    for (const sequence of dataset.sequences) {
      const tokens = sequence.tokens;
      for (let i = 0; i < tokens.length; i += 1) {
        const next = tokens[i]!;
        vocab.add(next);
        unigram[next] = (unigram[next] ?? 0) + 1;
        totalTokens += 1;
        for (let k = 1; k <= maxOrder && i - k >= 0; k += 1) {
          const key = contextKey(tokens.slice(i - k, i));
          const table = (counts[k]![key] ??= {});
          table[next] = (table[next] ?? 0) + 1;
        }
      }
    }

    return {
      backendId: "ngram",
      maxOrder,
      counts,
      unigram,
      totalTokens,
      vocab: [...vocab].sort(),
    };
  }

  hasContext(model: NgramMovementModel, context: MovementToken[]): boolean {
    const order = Math.min(model.maxOrder, context.length);
    if (order === 0) {
      return false;
    }
    const key = contextKey(context.slice(context.length - order));
    return Boolean(model.counts[order]?.[key]);
  }

  predict(model: NgramMovementModel, context: MovementToken[]): MovementPrediction {
    const requestedOrder = Math.min(model.maxOrder, context.length);
    for (let k = requestedOrder; k >= 1; k -= 1) {
      const key = contextKey(context.slice(context.length - k));
      const table = model.counts[k]?.[key];
      if (!table) {
        continue;
      }
      const total = Object.values(table).reduce((sum, count) => sum + count, 0);
      const ranked = rankCounts(table, total);
      const best = ranked[0];
      if (best) {
        return {
          token: best.token,
          probability: best.probability,
          contextOrder: k,
          fromBackoff: k < requestedOrder,
          ranked,
        };
      }
    }

    const total = model.totalTokens;
    const ranked = total > 0 ? rankCounts(model.unigram, total) : [];
    const best = ranked[0] ?? argmax(model.unigram);
    return {
      token: best?.token,
      probability: best?.probability ?? 0,
      contextOrder: 0,
      fromBackoff: requestedOrder > 0,
      ranked,
    };
  }
}

/** Per-bucket accuracy tally. */
export type MovementEvalBucket = {
  total: number;
  correct: number;
  accuracy: number;
};

/** Result of evaluating a trained policy on held-out movement sequences. */
export type MovementEvalResult = {
  total: number;
  correct: number;
  accuracy: number;
  meanProbability: number;
  /** Positions whose full-order context was seen in training (repeat fidelity). */
  seenContext: MovementEvalBucket;
  /** Positions whose full-order context was novel (generalization). */
  novelContext: MovementEvalBucket;
};

function emptyBucket(): { total: number; correct: number } {
  return { total: 0, correct: 0 };
}

function finalizeBucket(bucket: { total: number; correct: number }): MovementEvalBucket {
  return {
    total: bucket.total,
    correct: bucket.correct,
    accuracy: bucket.total === 0 ? 0 : bucket.correct / bucket.total,
  };
}

/**
 * Generalization eval harness. For every predictable position in the held-out
 * set, ask the trained model for the next movement and compare to ground truth,
 * splitting the score into *seen-context* (verbatim repeat) and *novel-context*
 * (generalized via backoff) buckets so repeat fidelity and generalization can
 * be tracked independently.
 */
export function evaluateMovementPolicy<Model>(
  backend: MovementPolicyBackend<Model>,
  model: Model,
  heldOut: MovementDataset,
): MovementEvalResult {
  let total = 0;
  let correct = 0;
  let probabilitySum = 0;
  const seen = emptyBucket();
  const novel = emptyBucket();

  for (const sequence of heldOut.sequences) {
    const tokens = sequence.tokens;
    for (let i = 1; i < tokens.length; i += 1) {
      const context = tokens.slice(0, i);
      const expected = tokens[i]!;
      const prediction = backend.predict(model, context);
      const bucket = backend.hasContext(model, context) ? seen : novel;
      const isCorrect = prediction.token === expected;
      total += 1;
      bucket.total += 1;
      probabilitySum += prediction.probability;
      if (isCorrect) {
        correct += 1;
        bucket.correct += 1;
      }
    }
  }

  return {
    total,
    correct,
    accuracy: total === 0 ? 0 : correct / total,
    meanProbability: total === 0 ? 0 : probabilitySum / total,
    seenContext: finalizeBucket(seen),
    novelContext: finalizeBucket(novel),
  };
}

/**
 * Deterministic (seeded) synthetic movement-stream generator. Simulates
 * recorded episodes without any real OS input so the capture → train → replay
 * loop can be validated in the cloud. Each sequence is a concatenation of whole
 * "motifs" — recurring sub-routines with internally deterministic transitions
 * (like a UI macro) — so a trained model has strong repeatable intra-motif
 * structure plus novel motif recombinations at the boundaries to generalize
 * over. Supply `motifs` explicitly for full control, or let the generator build
 * disjoint-token motifs from `vocabulary`.
 */
export function generateSyntheticMovementDataset(params: {
  seed: number;
  sequenceCount: number;
  vocabulary?: MovementToken[];
  motifs?: MovementToken[][];
  motifCount?: number;
  motifLength?: number;
  minLength?: number;
  maxLength?: number;
  noiseRate?: number;
}): MovementDataset {
  const vocabulary = params.vocabulary ?? [
    "move", "click", "type", "scroll", "drag", "submit", "focus", "hotkey",
    "open", "close", "select", "copy", "paste", "search", "send", "back",
  ];
  const motifCount = Math.max(1, params.motifCount ?? 4);
  const motifLength = Math.max(2, params.motifLength ?? 5);
  const minLength = Math.max(2, params.minLength ?? 8);
  const maxLength = Math.max(minLength, params.maxLength ?? 16);
  const noiseRate = params.noiseRate ?? 0.1;
  const rng = mulberry32(params.seed >>> 0);
  const pick = <T>(items: T[]): T => items[Math.floor(rng() * items.length)]!;

  // Build motifs from disjoint slices of the vocabulary so intra-motif n-grams
  // are unique — a model that memorizes a motif reproduces it verbatim.
  const motifs: MovementToken[][] =
    params.motifs ??
    Array.from({ length: motifCount }, (_, motifIndex) =>
      Array.from({ length: motifLength }, (_, tokenIndex) => {
        const vocabIndex = (motifIndex * motifLength + tokenIndex) % vocabulary.length;
        return vocabulary[vocabIndex]!;
      }),
    );

  const sequences: MovementSequence[] = Array.from({ length: Math.max(0, params.sequenceCount) }, (_, index) => {
    const tokens: MovementToken[] = [];
    const targetLength = minLength + Math.floor(rng() * (maxLength - minLength + 1));
    while (tokens.length < targetLength) {
      if (rng() < noiseRate) {
        tokens.push(pick(vocabulary));
      } else {
        tokens.push(...pick(motifs));
      }
    }
    return { id: `synthetic-${params.seed}-${index}`, tokens };
  });

  return { sequences };
}

/** Small deterministic PRNG (mulberry32) — avoids Math.random for reproducibility. */
function mulberry32(seed: number): () => number {
  let state = seed;
  return () => {
    state |= 0;
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
