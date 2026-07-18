import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

/**
 * In-process, backend-pluggable movement-prediction model.
 *
 * The reference training runner ({@link ../runner.js}) only *plans* a real
 * on-device MLX/axolotl job — nothing actually learns from recorded movement
 * data inside the process, so there is no way to validate the "train a local
 * model to repeat and generalize movements" objective in the cloud.
 *
 * This module fills that gap with a deterministic, dependency-free learner that
 * runs anywhere. It tokenizes recorded gesture/action sequences, trains a
 * pluggable {@link MovementModelBackend} on them, and predicts / generates the
 * next movements. The default {@link MarkovMovementBackend} is a genuine
 * (small, on-device-friendly) sequence model; the interface is the seam where a
 * real neural backend drops in without touching call sites.
 *
 * Everything here is deterministic (argmax with lexicographic tie-breaking, no
 * `Math.random`), so training and generalization are reproducible and testable.
 */

/** A single normalized movement token, e.g. `"device:swipe:left"`. */
export type MovementToken = string;

/** An ordered sequence of movement tokens drawn from one trajectory. */
export type MovementSequence = MovementToken[];

/** A ranked next-movement prediction. */
export type MovementPrediction = {
  token: MovementToken;
  /** Estimated probability in [0, 1]. */
  probability: number;
};

/** Options shared by every backend's training entry point. */
export type MovementTrainOptions = {
  /** Highest context order the backend may use (>= 1). Default 2. */
  order?: number;
};

/**
 * A trained, serializable movement policy. Implementations must round-trip
 * losslessly through {@link MovementModelBackend.deserialize}, so a model
 * trained locally can be persisted next to the other training artifacts.
 */
export type MovementPolicy = {
  /** Identifies the producing backend so deserialize can validate it. */
  readonly backendId: string;
  /**
   * Ranked next-token predictions for the given context (most recent token
   * last). Returns `[]` only when the model has learned nothing at all.
   */
  predictNext(context: MovementSequence): MovementPrediction[];
  /** Plain-JSON snapshot for persistence. */
  serialize(): unknown;
};

/** The pluggable local-model seam. */
export type MovementModelBackend = {
  readonly id: string;
  train(sequences: MovementSequence[], options?: MovementTrainOptions): MovementPolicy;
  deserialize(state: unknown): MovementPolicy;
};

/**
 * Deterministically tokenize a recorded action into a movement token.
 *
 * Prefers structured gesture metadata (kind + direction/target) over the free
 * text summary so semantically identical movements collapse to one token
 * regardless of how the summary was phrased.
 */
export function tokenizeAction(action: Pick<TrajectoryAction, "tool" | "summary" | "metadata">): MovementToken {
  const tool = normalizeSegment(action.tool) || "action";
  const metadata = action.metadata ?? {};
  const gesture = normalizeSegment(stringOrUndefined(metadata.gesture) ?? stringOrUndefined(metadata.kind));
  const qualifier = normalizeSegment(
    stringOrUndefined(metadata.direction) ??
      stringOrUndefined(metadata.target) ??
      stringOrUndefined(metadata.key) ??
      stringOrUndefined(metadata.valueSummary),
  );

  if (gesture) {
    return qualifier ? `${tool}:${gesture}:${qualifier}` : `${tool}:${gesture}`;
  }
  // Fall back to a compact summary slug so untagged actions still tokenize.
  const slug = normalizeSegment(action.summary).slice(0, 40);
  return slug ? `${tool}:${slug}` : tool;
}

/** Tokenize an ordered trajectory's actions into a single movement sequence. */
export function tokenizeTrajectory(span: TrajectorySpan): MovementSequence {
  return [...span.actions]
    .sort((a, b) => a.ts - b.ts)
    .map((action) => tokenizeAction(action));
}

// ---------------------------------------------------------------------------
// Default backend: back-off Markov chain over movement tokens.
// ---------------------------------------------------------------------------

type MarkovCounts = Record<string, Record<MovementToken, number>>;

type MarkovState = {
  backendId: "markov";
  version: 1;
  order: number;
  /** counts[order][contextKey][token] = observed count */
  counts: Record<number, MarkovCounts>;
};

const CONTEXT_DELIMITER = "";

/**
 * A back-off n-gram model: it predicts from the highest order with evidence for
 * the current context and backs off to shorter contexts (finally the unigram
 * distribution) when the exact context was never seen. This gives it real
 * generalization — a novel prefix still yields sensible movements via its
 * shorter, previously-seen suffix.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov";

  train(sequences: MovementSequence[], options?: MovementTrainOptions): MovementPolicy {
    const order = Math.max(1, Math.floor(options?.order ?? 2));
    const counts: Record<number, MarkovCounts> = {};
    for (let k = 0; k <= order; k += 1) {
      counts[k] = {};
    }

    for (const sequence of sequences) {
      for (let i = 0; i < sequence.length; i += 1) {
        const token = sequence[i]!;
        for (let k = 0; k <= order; k += 1) {
          if (i < k) {
            break;
          }
          const contextKey = k === 0 ? "" : sequence.slice(i - k, i).join(CONTEXT_DELIMITER);
          const bucket = (counts[k]![contextKey] ??= {});
          bucket[token] = (bucket[token] ?? 0) + 1;
        }
      }
    }

    return new MarkovPolicy({ backendId: "markov", version: 1, order, counts });
  }

  deserialize(state: unknown): MovementPolicy {
    const parsed = state as MarkovState | undefined;
    if (!parsed || parsed.backendId !== "markov" || parsed.version !== 1 || typeof parsed.order !== "number") {
      throw new Error("invalid markov movement policy state");
    }
    return new MarkovPolicy(parsed);
  }
}

class MarkovPolicy implements MovementPolicy {
  readonly backendId = "markov";

  constructor(private readonly state: MarkovState) {}

  predictNext(context: MovementSequence): MovementPrediction[] {
    for (let k = this.state.order; k >= 1; k -= 1) {
      if (context.length < k) {
        continue;
      }
      const contextKey = context.slice(context.length - k).join(CONTEXT_DELIMITER);
      const bucket = this.state.counts[k]?.[contextKey];
      if (bucket && Object.keys(bucket).length > 0) {
        return rankDistribution(bucket);
      }
    }
    const unigram = this.state.counts[0]?.[""];
    return unigram ? rankDistribution(unigram) : [];
  }

  serialize(): MarkovState {
    return this.state;
  }
}

/** Rank a count bucket into a normalized, deterministically ordered distribution. */
function rankDistribution(bucket: Record<MovementToken, number>): MovementPrediction[] {
  const total = Object.values(bucket).reduce((sum, count) => sum + count, 0);
  if (total === 0) {
    return [];
  }
  return Object.entries(bucket)
    .map(([token, count]) => ({ token, probability: count / total }))
    .sort((a, b) => (b.probability - a.probability) || (a.token < b.token ? -1 : a.token > b.token ? 1 : 0));
}

// ---------------------------------------------------------------------------
// High-level model + generalization eval.
// ---------------------------------------------------------------------------

export type MovementModelOptions = {
  backend?: MovementModelBackend;
  order?: number;
};

/**
 * Convenience wrapper tying a backend to the trajectory/token helpers so
 * callers work in terms of recorded movements rather than raw sequences.
 */
export class MovementModel {
  private readonly backend: MovementModelBackend;
  private readonly order: number;
  private policy: MovementPolicy | undefined;

  constructor(options: MovementModelOptions = {}) {
    this.backend = options.backend ?? new MarkovMovementBackend();
    this.order = Math.max(1, Math.floor(options.order ?? 2));
  }

  get backendId(): string {
    return this.backend.id;
  }

  trainFromSequences(sequences: MovementSequence[]): MovementPolicy {
    this.policy = this.backend.train(sequences, { order: this.order });
    return this.policy;
  }

  trainFromTrajectories(spans: TrajectorySpan[]): MovementPolicy {
    return this.trainFromSequences(spans.map((span) => tokenizeTrajectory(span)));
  }

  /** Highest-probability next movement, or `undefined` if none learned. */
  predictNext(context: MovementSequence): MovementToken | undefined {
    return this.requirePolicy().predictNext(context)[0]?.token;
  }

  rankNext(context: MovementSequence): MovementPrediction[] {
    return this.requirePolicy().predictNext(context);
  }

  /**
   * Greedily continue `seed` for up to `length` tokens, appending each
   * prediction to the context. Deterministic: argmax at every step. Stops early
   * if the model has no prediction for the running context.
   */
  generate(seed: MovementSequence, length: number): MovementSequence {
    const out: MovementSequence = [...seed];
    for (let i = 0; i < length; i += 1) {
      const next = this.predictNext(out);
      if (next === undefined) {
        break;
      }
      out.push(next);
    }
    return out.slice(seed.length);
  }

  serialize(): unknown {
    return this.requirePolicy().serialize();
  }

  load(state: unknown): MovementPolicy {
    this.policy = this.backend.deserialize(state);
    return this.policy;
  }

  private requirePolicy(): MovementPolicy {
    if (!this.policy) {
      throw new Error("movement model has not been trained or loaded");
    }
    return this.policy;
  }
}

export type GeneralizationReport = {
  /** Number of (context → next) predictions scored. */
  evaluated: number;
  /** Fraction where the top-1 prediction matched the held-out next token. */
  nextTokenAccuracy: number;
  /** Fraction where the true next token appeared anywhere in the ranking. */
  coverage: number;
  /** Held-out sequences the model could complete end-to-end via greedy argmax. */
  fullSequenceMatches: number;
  sequencesEvaluated: number;
};

/**
 * Measure how well a trained policy reproduces *held-out* movement sequences —
 * the generalization signal for objective #2(d). For each held-out sequence it
 * scores every next-token step (top-1 accuracy + rank coverage) and separately
 * checks whether greedy generation reproduces the whole tail from a one-token
 * seed.
 */
export function evaluateGeneralization(
  policy: MovementPolicy,
  heldOut: MovementSequence[],
): GeneralizationReport {
  let evaluated = 0;
  let correct = 0;
  let covered = 0;
  let fullMatches = 0;
  let sequencesEvaluated = 0;

  for (const sequence of heldOut) {
    if (sequence.length < 2) {
      continue;
    }
    sequencesEvaluated += 1;
    let sequenceIntact = true;

    for (let i = 1; i < sequence.length; i += 1) {
      const context = sequence.slice(0, i);
      const expected = sequence[i]!;
      const ranked = policy.predictNext(context);
      evaluated += 1;
      if (ranked[0]?.token === expected) {
        correct += 1;
      } else {
        sequenceIntact = false;
      }
      if (ranked.some((prediction) => prediction.token === expected)) {
        covered += 1;
      }
    }

    if (sequenceIntact) {
      fullMatches += 1;
    }
  }

  return {
    evaluated,
    nextTokenAccuracy: evaluated === 0 ? 0 : correct / evaluated,
    coverage: evaluated === 0 ? 0 : covered / evaluated,
    fullSequenceMatches: fullMatches,
    sequencesEvaluated,
  };
}

function normalizeSegment(value: string | undefined): string {
  if (!value) {
    return "";
  }
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function stringOrUndefined(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}
