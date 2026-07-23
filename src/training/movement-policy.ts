import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Movement policy: the *inference* half of the local-movement learning
 * subsystem. The capture/export/replay pipeline records what the user did; this
 * module learns a model from those recordings that can (a) repeat a recorded
 * movement sequence and (b) generalize to new-but-related sequences.
 *
 * The concrete model here is a dependency-free, fully deterministic
 * variable-order backoff n-gram over *action tokens*, so it trains and runs
 * in-process (cloud/CI friendly, no native deps, no external training run). It
 * sits behind {@link MovementPolicyBackend} so a real on-device small model can
 * be swapped in later without changing call sites — that is the documented seam
 * for objective #2's pluggable local-model backend.
 */

export type MovementToken = string;

/** Minimal shape a trajectory action needs to be tokenized. */
export type TokenizableAction = {
  tool: string;
  summary: string;
  ts?: number;
  metadata?: Record<string, unknown> | undefined;
};

export type MovementSequence = {
  trajectoryId: string;
  tokens: MovementToken[];
};

export type MovementPredictionSource = "exact" | "backoff" | "prior" | "empty";

export type MovementPrediction = {
  /** Best next token, or undefined when the model has learned nothing. */
  token: MovementToken | undefined;
  /** Probability mass of {@link token} within the matched context (0..1). */
  confidence: number;
  /**
   * `exact` — the full provided context matched a learned context.
   * `backoff` — only a shorter suffix of the context matched.
   * `prior` — no context matched; fell back to the global unigram prior.
   * `empty` — the model is untrained / has no vocabulary.
   */
  source: MovementPredictionSource;
  /** Length of the context suffix that produced the prediction (0 for prior). */
  order: number;
  /** Ranked candidates (includes {@link token}), highest probability first. */
  alternatives: { token: MovementToken; probability: number }[];
};

/**
 * Pluggable model backend. The n-gram implementation below is the default,
 * deterministic, cloud-runnable backend; a real on-device model would implement
 * this same surface.
 */
export interface MovementPolicyBackend {
  readonly name: string;
  readonly maxOrder: number;
  train(sequences: MovementSequence[]): void;
  predictNext(context: MovementToken[]): MovementPrediction;
  vocabulary(): MovementToken[];
}

const CONTEXT_SEP = "";
const DEFAULT_MAX_ORDER = 3;
const DEFAULT_ALTERNATIVES = 5;

function slug(value: string, maxLength = 48): string {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, maxLength);
}

/**
 * Turn an action into a stable, low-cardinality token. Structured movement
 * metadata (gesture/direction/target, os event) is preferred over free-text
 * summaries so that semantically identical movements collapse to the same
 * token — which is what lets the model generalize rather than memorize prose.
 */
export function tokenizeMovementAction(action: TokenizableAction): MovementToken {
  const tool = slug(action.tool) || "action";
  const meta = action.metadata ?? {};
  const parts: string[] = [tool];

  const gesture = meta["gesture"];
  const event = meta["event"];
  if (typeof gesture === "string" && gesture.length > 0) {
    parts.push(slug(gesture) || "gesture");
    const direction = meta["direction"];
    const target = meta["target"];
    if (typeof direction === "string" && direction.length > 0) {
      parts.push(slug(direction));
    } else if (typeof target === "string" && target.length > 0) {
      parts.push(slug(target));
    }
  } else if (typeof event === "string" && event.length > 0) {
    parts.push(slug(event) || "event");
  } else {
    parts.push(slug(action.summary) || "step");
  }

  return parts.filter((part) => part.length > 0).join(":");
}

export function sequenceFromActions(trajectoryId: string, actions: TokenizableAction[]): MovementSequence {
  const ordered = [...actions].sort((a, b) => (a.ts ?? 0) - (b.ts ?? 0));
  return { trajectoryId, tokens: ordered.map(tokenizeMovementAction) };
}

/**
 * Extract the ordered action-token sequence from a trajectory. When
 * `useReviewed` is set and a redacted review exists, the reviewed (privacy-safe)
 * actions are used — matching the training pipeline's reviewed-export gate.
 */
export function extractMovementSequence(
  trajectory: TrajectorySpan,
  options: { useReviewed?: boolean } = {},
): MovementSequence {
  if (options.useReviewed && trajectory.review?.redactedActions) {
    return sequenceFromActions(trajectory.id, trajectory.review.redactedActions);
  }
  return sequenceFromActions(trajectory.id, trajectory.actions);
}

type SerializedNgramPolicy = {
  version: 1;
  kind: "ngram-movement-policy";
  maxOrder: number;
  unigram: Record<MovementToken, number>;
  contexts: Record<string, Record<MovementToken, number>>;
};

/**
 * Variable-order backoff n-gram over movement tokens.
 *
 * Repetition: a sequence whose full prefix was seen in training predicts its
 * recorded continuation exactly. Generalization: an unseen prefix backs off to
 * successively shorter learned suffixes, and finally to the global unigram
 * prior — so a novel-but-related movement still yields a sensible next action.
 * Fully deterministic: ties break lexicographically, so results are stable and
 * testable in the cloud with no randomness.
 */
export class NgramMovementPolicy implements MovementPolicyBackend {
  readonly name = "ngram-movement-policy";
  readonly maxOrder: number;

  private readonly contexts = new Map<string, Map<MovementToken, number>>();
  private readonly unigram = new Map<MovementToken, number>();
  private unigramTotal = 0;

  constructor(maxOrder: number = DEFAULT_MAX_ORDER) {
    this.maxOrder = Math.max(1, Math.floor(maxOrder));
  }

  train(sequences: MovementSequence[]): void {
    this.contexts.clear();
    this.unigram.clear();
    this.unigramTotal = 0;
    for (const sequence of sequences) {
      this.observe(sequence.tokens);
    }
  }

  /** Incrementally fold one more sequence into the model (online learning). */
  observe(tokens: MovementToken[]): void {
    for (let i = 0; i < tokens.length; i += 1) {
      const token = tokens[i]!;
      this.unigram.set(token, (this.unigram.get(token) ?? 0) + 1);
      this.unigramTotal += 1;
      for (let order = 1; order <= this.maxOrder; order += 1) {
        if (i - order < 0) {
          break;
        }
        const context = tokens.slice(i - order, i);
        this.increment(context, token);
      }
    }
  }

  predictNext(context: MovementToken[]): MovementPrediction {
    const maxUsable = Math.min(this.maxOrder, context.length);
    for (let order = maxUsable; order >= 1; order -= 1) {
      const suffix = context.slice(context.length - order);
      const counts = this.contexts.get(suffix.join(CONTEXT_SEP));
      if (counts && counts.size > 0) {
        const alternatives = rank(counts);
        return {
          token: alternatives[0]!.token,
          confidence: alternatives[0]!.probability,
          source: order === context.length ? "exact" : "backoff",
          order,
          alternatives,
        };
      }
    }

    if (this.unigramTotal > 0) {
      const alternatives = rank(this.unigram);
      return {
        token: alternatives[0]!.token,
        confidence: alternatives[0]!.probability,
        source: "prior",
        order: 0,
        alternatives,
      };
    }

    return { token: undefined, confidence: 0, source: "empty", order: 0, alternatives: [] };
  }

  vocabulary(): MovementToken[] {
    return [...this.unigram.keys()].sort();
  }

  toJSON(): SerializedNgramPolicy {
    const contexts: Record<string, Record<MovementToken, number>> = {};
    for (const [key, counts] of this.contexts) {
      contexts[key] = Object.fromEntries(counts);
    }
    return {
      version: 1,
      kind: "ngram-movement-policy",
      maxOrder: this.maxOrder,
      unigram: Object.fromEntries(this.unigram),
      contexts,
    };
  }

  static fromJSON(serialized: SerializedNgramPolicy): NgramMovementPolicy {
    const policy = new NgramMovementPolicy(serialized.maxOrder);
    for (const [token, count] of Object.entries(serialized.unigram)) {
      policy.unigram.set(token, count);
      policy.unigramTotal += count;
    }
    for (const [key, counts] of Object.entries(serialized.contexts)) {
      const map = new Map<MovementToken, number>();
      for (const [token, count] of Object.entries(counts)) {
        map.set(token, count);
      }
      policy.contexts.set(key, map);
    }
    return policy;
  }

  private increment(context: MovementToken[], token: MovementToken): void {
    const key = context.join(CONTEXT_SEP);
    let counts = this.contexts.get(key);
    if (!counts) {
      counts = new Map<MovementToken, number>();
      this.contexts.set(key, counts);
    }
    counts.set(token, (counts.get(token) ?? 0) + 1);
  }
}

function rank(counts: Map<MovementToken, number>): { token: MovementToken; probability: number }[] {
  let total = 0;
  for (const count of counts.values()) {
    total += count;
  }
  return [...counts.entries()]
    .map(([token, count]) => ({ token, probability: total > 0 ? count / total : 0 }))
    .sort((a, b) => (b.probability !== a.probability ? b.probability - a.probability : a.token < b.token ? -1 : 1))
    .slice(0, DEFAULT_ALTERNATIVES);
}

export type TrainMovementPolicyOptions = {
  maxOrder?: number;
  useReviewed?: boolean;
};

/** Convenience: build and train an n-gram policy directly from trajectories. */
export function trainMovementPolicy(
  trajectories: TrajectorySpan[],
  options: TrainMovementPolicyOptions = {},
): NgramMovementPolicy {
  const policy = new NgramMovementPolicy(options.maxOrder ?? DEFAULT_MAX_ORDER);
  policy.train(trajectories.map((trajectory) => extractMovementSequence(trajectory, options)));
  return policy;
}

export type GenerateMovementOptions = {
  length: number;
  includeSeed?: boolean;
};

/**
 * Greedily roll the policy forward from a seed context to synthesize a
 * new-but-related movement sequence — the "perform new movements" half of the
 * objective. Deterministic (argmax); stops early if the model predicts nothing.
 */
export function generateMovementSequence(
  backend: MovementPolicyBackend,
  seed: MovementToken[],
  options: GenerateMovementOptions,
): MovementToken[] {
  const generated: MovementToken[] = [];
  const context = [...seed];
  for (let i = 0; i < Math.max(0, options.length); i += 1) {
    const prediction = backend.predictNext(context);
    if (prediction.token === undefined) {
      break;
    }
    generated.push(prediction.token);
    context.push(prediction.token);
  }
  return options.includeSeed ? [...seed, ...generated] : generated;
}

export type MovementEvalResult = {
  totalPredictions: number;
  correct: number;
  accuracy: number;
  k: number;
  topKCorrect: number;
  topKAccuracy: number;
  bySource: Record<MovementPredictionSource, { total: number; correct: number }>;
};

export type EvaluateMovementPolicyOptions = {
  /** Consider a prediction correct if the expected token is in the top-k. */
  k?: number;
  /** Only score positions with at least this many preceding context tokens. */
  minContext?: number;
};

/**
 * Generalization eval harness: for every position in each held-out sequence,
 * predict the next token from its prefix and compare to ground truth. Reports
 * top-1 and top-k accuracy, broken down by which backoff level produced each
 * prediction — so exact-recall vs. generalized predictions are separable.
 */
export function evaluateMovementPolicy(
  backend: MovementPolicyBackend,
  testSequences: MovementSequence[],
  options: EvaluateMovementPolicyOptions = {},
): MovementEvalResult {
  const k = Math.max(1, options.k ?? 1);
  const minContext = Math.max(0, options.minContext ?? 0);
  const bySource: MovementEvalResult["bySource"] = {
    exact: { total: 0, correct: 0 },
    backoff: { total: 0, correct: 0 },
    prior: { total: 0, correct: 0 },
    empty: { total: 0, correct: 0 },
  };

  let totalPredictions = 0;
  let correct = 0;
  let topKCorrect = 0;

  for (const sequence of testSequences) {
    for (let i = minContext; i < sequence.tokens.length; i += 1) {
      if (i === 0) {
        continue;
      }
      const context = sequence.tokens.slice(0, i);
      const expected = sequence.tokens[i]!;
      const prediction = backend.predictNext(context);
      totalPredictions += 1;
      const sourceStats = bySource[prediction.source];
      sourceStats.total += 1;

      const isTop1 = prediction.token === expected;
      if (isTop1) {
        correct += 1;
        sourceStats.correct += 1;
      }
      const inTopK = prediction.alternatives.slice(0, k).some((candidate) => candidate.token === expected);
      if (inTopK) {
        topKCorrect += 1;
      }
    }
  }

  return {
    totalPredictions,
    correct,
    accuracy: totalPredictions > 0 ? correct / totalPredictions : 0,
    k,
    topKCorrect,
    topKAccuracy: totalPredictions > 0 ? topKCorrect / totalPredictions : 0,
    bySource,
  };
}
