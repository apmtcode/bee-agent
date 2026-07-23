import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Local-movement learning: pluggable model backend.
 *
 * The rest of the training subsystem (`runner.ts`, `execution-service.ts`)
 * generates launch scripts for *external* on-device runtimes (mlx / axolotl).
 * Those run only on a real Apple-silicon machine and cannot execute — or be
 * tested — in the cloud.
 *
 * This module closes objective #2 parts (c) and (d) with an *in-process*,
 * fully deterministic model that actually learns from a recorded movement
 * dataset and can (c) repeat the recorded movements and (d) generalize to new
 * but related movements. The backend is pluggable via {@link MovementModelBackend}
 * so a real on-device small model can be dropped in later behind the same seam;
 * the shipped {@link MarkovMovementBackend} is deterministic so cloud/CI tests
 * pass without any OS access.
 */

/** A single canonicalized movement, e.g. `"tap:submit-button"`. */
export type MovementToken = string;

/** One recorded (or synthetic) sequence of movements. */
export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
  /** Optional discrete context (e.g. `{ appId: "mail" }`) folded into learning. */
  context?: Record<string, string>;
};

export type MovementDataset = {
  sequences: MovementSequence[];
};

export type MovementPrediction = {
  token: MovementToken;
  /** Interpolated probability in `[0, 1]`, ranked descending. */
  score: number;
};

export type MovementTrainingOptions = {
  /** Highest n-gram order to learn (context length). Default 2 (trigram). */
  maxOrder?: number;
  /** Backoff weight per dropped order for interpolation. Default 0.4. */
  backoff?: number;
};

export type MovementPredictOptions = {
  /** Max candidates to return. Default 5. */
  limit?: number;
};

export type MovementGenerateOptions = {
  /** Hard cap on generated tokens (excludes the seed). Default 64. */
  maxLength?: number;
};

/** Portable, JSON-serializable snapshot of a trained model. */
export type MovementModelSnapshot = {
  version: 1;
  backendId: string;
  maxOrder: number;
  backoff: number;
  /** `orders[o]` maps a context key to `{ token: count }` for n-gram order `o`. */
  orders: Array<Record<string, Record<MovementToken, number>>>;
  vocabulary: MovementToken[];
};

/** A trained model. Deterministic: identical inputs yield identical outputs. */
export interface TrainedMovementModel {
  readonly backendId: string;
  readonly vocabulary: readonly MovementToken[];
  /** Ranked next-movement candidates given the trailing context. */
  predictNext(context: MovementToken[], options?: MovementPredictOptions): MovementPrediction[];
  /** Roll out a full continuation from `seed` (most-likely path). */
  generate(seed: MovementToken[], options?: MovementGenerateOptions): MovementToken[];
  serialize(): MovementModelSnapshot;
}

/** A pluggable training backend. Swap this for a real on-device model later. */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainingOptions): Promise<TrainedMovementModel>;
  load(snapshot: MovementModelSnapshot): TrainedMovementModel;
}

const START = "start";
const END = "end";
const KEY_SEP = "";

function keyFor(suffix: MovementToken[]): string {
  return suffix.join(KEY_SEP);
}

function isBoundary(token: MovementToken): boolean {
  return token === START || token === END;
}

/**
 * Interpolated variable-order Markov model with stupid-backoff.
 *
 * Learning is a pure count of `(context -> next)` transitions across every
 * order `0..maxOrder`. Prediction interpolates all orders whose context suffix
 * was seen, weighting higher orders more (`backoff^(maxOrder-o)`). Because
 * lower orders are always mixed in, an *unseen* high-order context degrades
 * gracefully to a shorter-context prediction instead of failing — that backoff
 * is exactly what lets the model generalize to new-but-related movements.
 */
class MarkovMovementModel implements TrainedMovementModel {
  readonly backendId: string;
  private readonly maxOrder: number;
  private readonly backoff: number;
  // orders[o]: contextKey -> (token -> count)
  private readonly orders: Array<Map<string, Map<MovementToken, number>>>;
  private readonly totals: Array<Map<string, number>>;
  private readonly vocab: Set<MovementToken>;

  constructor(params: {
    backendId: string;
    maxOrder: number;
    backoff: number;
    orders: Array<Map<string, Map<MovementToken, number>>>;
    vocab: Set<MovementToken>;
  }) {
    this.backendId = params.backendId;
    this.maxOrder = params.maxOrder;
    this.backoff = params.backoff;
    this.orders = params.orders;
    this.vocab = params.vocab;
    this.totals = params.orders.map((byContext) => {
      const totals = new Map<string, number>();
      for (const [key, dist] of byContext) {
        let sum = 0;
        for (const count of dist.values()) sum += count;
        totals.set(key, sum);
      }
      return totals;
    });
  }

  get vocabulary(): readonly MovementToken[] {
    return [...this.vocab].filter((token) => !isBoundary(token)).sort();
  }

  /** Score every candidate token by interpolating all applicable orders. */
  private scoreCandidates(context: MovementToken[]): Map<MovementToken, number> {
    const padded = [...Array(this.maxOrder).fill(START), ...context];
    const scores = new Map<MovementToken, number>();
    for (let order = 0; order <= this.maxOrder; order++) {
      const suffix = order === 0 ? [] : padded.slice(padded.length - order);
      const key = keyFor(suffix);
      const dist = this.orders[order]?.get(key);
      if (!dist) continue;
      const total = this.totals[order]?.get(key) ?? 0;
      if (total === 0) continue;
      const weight = Math.pow(this.backoff, this.maxOrder - order);
      for (const [token, count] of dist) {
        scores.set(token, (scores.get(token) ?? 0) + weight * (count / total));
      }
    }
    return scores;
  }

  private rank(scores: Map<MovementToken, number>): MovementPrediction[] {
    // Normalize interpolation weights back into a probability distribution.
    let sum = 0;
    for (const value of scores.values()) sum += value;
    const ranked: MovementPrediction[] = [];
    for (const [token, value] of scores) {
      ranked.push({ token, score: sum > 0 ? value / sum : 0 });
    }
    // Deterministic ordering: score desc, then token asc.
    ranked.sort((a, b) => (b.score - a.score) || (a.token < b.token ? -1 : a.token > b.token ? 1 : 0));
    return ranked;
  }

  predictNext(context: MovementToken[], options: MovementPredictOptions = {}): MovementPrediction[] {
    const limit = options.limit ?? 5;
    const ranked = this.rank(this.scoreCandidates(context)).filter((p) => !isBoundary(p.token));
    return ranked.slice(0, Math.max(0, limit));
  }

  generate(seed: MovementToken[], options: MovementGenerateOptions = {}): MovementToken[] {
    const maxLength = options.maxLength ?? 64;
    const generated: MovementToken[] = [];
    let context = [...seed];
    for (let step = 0; step < maxLength; step++) {
      const ranked = this.rank(this.scoreCandidates(context));
      const top = ranked[0];
      if (!top || top.token === END) break;
      if (isBoundary(top.token)) break;
      generated.push(top.token);
      context = [...context, top.token];
    }
    return generated;
  }

  serialize(): MovementModelSnapshot {
    const orders = this.orders.map((byContext) => {
      const out: Record<string, Record<MovementToken, number>> = {};
      // Sort keys/tokens for a stable, diff-friendly snapshot.
      for (const key of [...byContext.keys()].sort()) {
        const dist = byContext.get(key)!;
        const tokenCounts: Record<MovementToken, number> = {};
        for (const token of [...dist.keys()].sort()) {
          tokenCounts[token] = dist.get(token)!;
        }
        out[key] = tokenCounts;
      }
      return out;
    });
    return {
      version: 1,
      backendId: this.backendId,
      maxOrder: this.maxOrder,
      backoff: this.backoff,
      orders,
      vocabulary: this.vocabulary as MovementToken[],
    };
  }
}

export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov-v1";

  async train(dataset: MovementDataset, options: MovementTrainingOptions = {}): Promise<TrainedMovementModel> {
    const maxOrder = Math.max(0, options.maxOrder ?? 2);
    const backoff = options.backoff ?? 0.4;
    const orders: Array<Map<string, Map<MovementToken, number>>> = Array.from(
      { length: maxOrder + 1 },
      () => new Map<string, Map<MovementToken, number>>(),
    );
    const vocab = new Set<MovementToken>();

    for (const sequence of dataset.sequences) {
      const contextPrefix = contextTokens(sequence.context);
      // Fold discrete context into the leading START pad so different contexts
      // learn distinct high-order transitions but share the unigram fallback.
      const padded = [...Array(maxOrder).fill(START), ...contextPrefix, ...sequence.tokens, END];
      for (const token of sequence.tokens) vocab.add(token);
      for (let i = 0; i < padded.length; i++) {
        const token = padded[i]!;
        if (token === START) continue; // START is context-only, never a target.
        for (let order = 0; order <= maxOrder; order++) {
          if (i - order < 0) continue;
          const suffix = order === 0 ? [] : padded.slice(i - order, i);
          const key = keyFor(suffix);
          let byToken = orders[order]!.get(key);
          if (!byToken) {
            byToken = new Map<MovementToken, number>();
            orders[order]!.set(key, byToken);
          }
          byToken.set(token, (byToken.get(token) ?? 0) + 1);
        }
      }
    }

    return new MarkovMovementModel({ backendId: this.id, maxOrder, backoff, orders, vocab });
  }

  load(snapshot: MovementModelSnapshot): TrainedMovementModel {
    const orders = snapshot.orders.map((byContext) => {
      const map = new Map<string, Map<MovementToken, number>>();
      for (const [key, tokenCounts] of Object.entries(byContext)) {
        const dist = new Map<MovementToken, number>();
        for (const [token, count] of Object.entries(tokenCounts)) dist.set(token, count);
        map.set(key, dist);
      }
      return map;
    });
    const vocab = new Set<MovementToken>(snapshot.vocabulary);
    return new MarkovMovementModel({
      backendId: snapshot.backendId,
      maxOrder: snapshot.maxOrder,
      backoff: snapshot.backoff,
      orders,
      vocab,
    });
  }
}

/** Registry so movement-model backends are pluggable by id. */
export class MovementModelRegistry {
  private readonly backends = new Map<string, MovementModelBackend>();

  register(backend: MovementModelBackend): this {
    this.backends.set(backend.id, backend);
    return this;
  }

  get(id: string): MovementModelBackend | undefined {
    return this.backends.get(id);
  }

  require(id: string): MovementModelBackend {
    const backend = this.backends.get(id);
    if (!backend) {
      throw new Error(`movement-model backend not registered: ${id}`);
    }
    return backend;
  }

  list(): string[] {
    return [...this.backends.keys()].sort();
  }
}

export function createDefaultMovementModelRegistry(): MovementModelRegistry {
  return new MovementModelRegistry().register(new MarkovMovementBackend());
}

function contextTokens(context: Record<string, string> | undefined): MovementToken[] {
  if (!context) return [];
  return Object.keys(context)
    .sort()
    .map((key) => `ctx:${key}=${context[key]}`);
}

/**
 * Canonicalize a recorded {@link TrajectorySpan} into a movement sequence.
 * Actions become movement tokens (`tool:summary`, slugified) in timestamp order
 * so recorded operator/device trajectories feed the model directly.
 */
export function tokenizeTrajectorySpan(span: TrajectorySpan): MovementSequence {
  const tokens = [...span.actions]
    .sort((a, b) => a.ts - b.ts)
    .map((action) => {
      const gesture = typeof action.metadata?.gesture === "string" ? action.metadata.gesture : undefined;
      const target =
        typeof action.metadata?.target === "string"
          ? action.metadata.target
          : typeof action.metadata?.direction === "string"
            ? action.metadata.direction
            : undefined;
      const verb = gesture ?? slug(action.tool);
      const object = target ? slug(target) : slug(action.summary);
      return object ? `${verb}:${object}` : verb;
    });
  return { id: span.id, tokens };
}

function slug(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

export type MovementEvalResult = {
  /** Held-out (context, next) prediction points scored. */
  samples: number;
  /** Fraction where the true next token was the top-ranked prediction. */
  top1Accuracy: number;
  /** Fraction where the true next token appeared in the top-k predictions. */
  topKAccuracy: number;
  k: number;
};

export type MovementEvalOptions = {
  /** k for top-k accuracy. Default 3. */
  k?: number;
  /** Context window fed to the model per prediction point. Default model order. */
  contextWindow?: number;
};

/**
 * Generalization eval harness: for each held-out sequence, walk every position
 * and score whether the model predicts the true next movement from the prefix.
 * Run this against sequences *not* in the training set to measure generalization.
 */
export function evaluateMovementModel(
  model: TrainedMovementModel,
  heldOut: MovementDataset,
  options: MovementEvalOptions = {},
): MovementEvalResult {
  const k = options.k ?? 3;
  const window = options.contextWindow;
  let samples = 0;
  let top1 = 0;
  let topK = 0;
  for (const sequence of heldOut.sequences) {
    for (let i = 0; i < sequence.tokens.length; i++) {
      const start = window === undefined ? 0 : Math.max(0, i - window);
      const context = sequence.tokens.slice(start, i);
      const truth = sequence.tokens[i]!;
      const predictions = model.predictNext(context, { limit: k });
      samples++;
      if (predictions[0]?.token === truth) top1++;
      if (predictions.some((p) => p.token === truth)) topK++;
    }
  }
  return {
    samples,
    top1Accuracy: samples > 0 ? top1 / samples : 0,
    topKAccuracy: samples > 0 ? topK / samples : 0,
    k,
  };
}

export type SyntheticMovementSpec = {
  /** Deterministic seed — same seed yields the same dataset (no wall-clock/RNG). */
  seed: number;
  /** Number of sequences to emit. */
  sequenceCount: number;
  /** Vocabulary of movement tokens the grammar draws from. */
  vocabulary: MovementToken[];
  /** Min/max tokens per sequence. */
  minLength?: number;
  maxLength?: number;
  /** Optional shared context stamped on every sequence. */
  context?: Record<string, string>;
};

/**
 * Deterministic synthetic event-stream generator. Emits sequences from a small
 * first-order grammar (each token biases the next), so the produced streams
 * have learnable structure — exactly what the movement model should capture —
 * without any real OS input. Uses a seeded LCG; forbidden `Math.random`/clock
 * are never touched, keeping it reproducible in tests and cloud runs.
 */
export function generateSyntheticMovementDataset(spec: SyntheticMovementSpec): MovementDataset {
  const vocab = spec.vocabulary;
  if (vocab.length === 0) {
    throw new Error("generateSyntheticMovementDataset requires a non-empty vocabulary");
  }
  const minLength = Math.max(1, spec.minLength ?? 3);
  const maxLength = Math.max(minLength, spec.maxLength ?? 8);
  const rng = createLcg(spec.seed);
  const sequences: MovementSequence[] = [];
  for (let s = 0; s < spec.sequenceCount; s++) {
    const length = minLength + Math.floor(rng() * (maxLength - minLength + 1));
    const tokens: MovementToken[] = [];
    let index = Math.floor(rng() * vocab.length);
    for (let t = 0; t < length; t++) {
      tokens.push(vocab[index]!);
      // Bias: 70% advance to the next token in the ring, else jump randomly.
      index = rng() < 0.7 ? (index + 1) % vocab.length : Math.floor(rng() * vocab.length);
    }
    sequences.push({
      id: `synthetic-${spec.seed}-${s}`,
      tokens,
      ...(spec.context ? { context: spec.context } : {}),
    });
  }
  return { sequences };
}

/** Deterministic linear-congruential PRNG (Numerical Recipes constants). */
function createLcg(seed: number): () => number {
  let state = (Math.abs(Math.floor(seed)) % 2147483647) || 1;
  return () => {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    return state / 0x7fffffff;
  };
}
