/**
 * Pluggable local-model backend for the movement-learning subsystem.
 *
 * The existing {@link LocalAppleSiliconTrainingRunner} emits shell commands for a
 * real MLX/Axolotl training run — those cannot execute in the cloud/CI. This
 * module provides the *pluggable backend seam* the roadmap calls for, together
 * with a deterministic in-process n-gram backend that genuinely learns from
 * recorded movement sequences: it can (a) reproduce recorded movements exactly
 * and (b) generalize to unseen-but-related sequences by n-gram backoff.
 *
 * Everything here is deterministic and dependency-free so it runs identically in
 * the cloud and on-device. A real small-model backend (e.g. an on-device LoRA
 * adapter) can implement {@link MovementModelBackend} without touching callers.
 */

// Sentinels are prefixed with a control char so they can never collide with a
// real tokenized action derived from user text.
const START = "START";
const END = "END";
const CTX_SEP = "";
const TOKEN_JOIN = "›"; // "›" between tool and summary in tool-summary granularity

/** A single learnable movement token (a normalized action). */
export type MovementToken = string;

/** An ordered movement sequence keyed by its originating trajectory/session. */
export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

/** A training-ready dataset: sequences plus the sorted unique vocabulary. */
export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
  vocabulary: MovementToken[];
};

/** A ranked next-token prediction with its (normalized) probability. */
export type MovementPrediction = {
  token: MovementToken;
  probability: number;
};

/** Portable, JSON-serializable model weights (n-gram counts). */
export type SerializedMovementModel = {
  version: 1;
  backend: string;
  order: number;
  vocabulary: MovementToken[];
  /** context-key → (next-token → count), for context lengths 0..order. */
  counts: Record<string, Record<MovementToken, number>>;
};

export type GenerateOptions = {
  /** Optional prefix to continue from. Defaults to the sequence start. */
  seed?: MovementToken[];
  /** Hard cap on generated tokens (excluding the seed). Defaults to 256. */
  maxLength?: number;
};

/** A trained model instance: repeat (generate) and generalize (predictNext). */
export interface TrainedMovementModel {
  readonly backend: string;
  readonly order: number;
  readonly vocabulary: readonly MovementToken[];
  /** Rank the most likely next tokens given a prefix (empty prefix ⇒ start). */
  predictNext(prefix: MovementToken[], topK?: number): MovementPrediction[];
  /** Greedily generate a full sequence (stops at END or maxLength). */
  generate(options?: GenerateOptions): MovementToken[];
  serialize(): SerializedMovementModel;
}

export type TrainMovementOptions = {
  /** Context length (number of preceding tokens conditioned on). Default 2. */
  order?: number;
};

/** The pluggable backend contract. Real on-device backends implement this. */
export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, options?: TrainMovementOptions): TrainedMovementModel;
  restore(serialized: SerializedMovementModel): TrainedMovementModel;
}

function keyOf(tokens: MovementToken[]): string {
  return tokens.join(CTX_SEP);
}

function totalOf(row: Record<MovementToken, number>): number {
  let total = 0;
  for (const value of Object.values(row)) {
    total += value;
  }
  return total;
}

/**
 * Deterministic n-gram movement model with stupid-backoff generalization.
 *
 * Scores are ranked argmax with a lexicographic tie-break, so a given dataset
 * always yields byte-identical predictions — essential for reproducible tests
 * and for a training gate that must behave the same in the cloud and on-device.
 */
class NgramMovementModel implements TrainedMovementModel {
  readonly backend = "ngram";
  readonly order: number;
  readonly vocabulary: readonly MovementToken[];
  private readonly counts: Map<string, Map<MovementToken, number>>;

  constructor(order: number, vocabulary: MovementToken[], counts: Map<string, Map<MovementToken, number>>) {
    this.order = order;
    this.vocabulary = vocabulary;
    this.counts = counts;
  }

  private paddedContext(prefix: MovementToken[]): MovementToken[] {
    const tail = prefix.slice(-this.order);
    const padding: MovementToken[] = [];
    for (let i = tail.length; i < this.order; i += 1) {
      padding.push(START);
    }
    return [...padding, ...tail];
  }

  /** Stupid-backoff score of `token` given a padded, order-length context. */
  private score(context: MovementToken[], token: MovementToken): number {
    for (let len = this.order; len >= 1; len -= 1) {
      const row = this.counts.get(keyOf(context.slice(this.order - len)));
      if (!row) {
        continue;
      }
      const hit = row.get(token);
      if (hit !== undefined && hit > 0) {
        const total = totalOf(Object.fromEntries(row));
        return 0.4 ** (this.order - len) * (hit / total);
      }
    }
    // Unigram (context length 0) with add-one smoothing so nothing is zero.
    const unigram = this.counts.get("");
    const uniTotal = unigram ? totalOf(Object.fromEntries(unigram)) : 0;
    const uniHit = unigram?.get(token) ?? 0;
    return 0.4 ** this.order * ((uniHit + 1) / (uniTotal + this.vocabulary.length + 1));
  }

  predictNext(prefix: MovementToken[], topK = 5): MovementPrediction[] {
    const context = this.paddedContext(prefix);
    const candidates = [...this.vocabulary, END];
    const scored = candidates.map((token) => ({ token, raw: this.score(context, token) }));
    const total = scored.reduce((sum, entry) => sum + entry.raw, 0) || 1;
    return scored
      .map((entry) => ({ token: entry.token, probability: entry.raw / total }))
      .sort((a, b) => (b.probability !== a.probability ? b.probability - a.probability : a.token < b.token ? -1 : 1))
      .slice(0, Math.max(0, topK));
  }

  generate(options: GenerateOptions = {}): MovementToken[] {
    const maxLength = options.maxLength ?? 256;
    const generated: MovementToken[] = [...(options.seed ?? [])];
    for (let step = 0; step < maxLength; step += 1) {
      const [next] = this.predictNext(generated, 1);
      if (!next || next.token === END) {
        break;
      }
      generated.push(next.token);
    }
    return options.seed ? generated.slice(options.seed.length) : generated;
  }

  /** Per-candidate normalized probability (proper distribution) of `token`. */
  probabilityOf(prefix: MovementToken[], token: MovementToken): number {
    const context = this.paddedContext(prefix);
    const candidates = new Set<MovementToken>([...this.vocabulary, END, token]);
    let total = 0;
    let target = 0;
    for (const candidate of candidates) {
      const raw = this.score(context, candidate);
      total += raw;
      if (candidate === token) {
        target = raw;
      }
    }
    return total > 0 ? target / total : 0;
  }

  serialize(): SerializedMovementModel {
    const counts: Record<string, Record<MovementToken, number>> = {};
    for (const [key, row] of this.counts) {
      counts[key] = Object.fromEntries(row);
    }
    return { version: 1, backend: this.backend, order: this.order, vocabulary: [...this.vocabulary], counts };
  }
}

/** The default deterministic, in-process movement-model backend. */
export class NgramMovementBackend implements MovementModelBackend {
  readonly name = "ngram";

  train(dataset: MovementDataset, options: TrainMovementOptions = {}): NgramMovementModel {
    const order = Math.max(1, Math.floor(options.order ?? 2));
    const counts = new Map<string, Map<MovementToken, number>>();
    const record = (context: MovementToken[], next: MovementToken): void => {
      const key = keyOf(context);
      let row = counts.get(key);
      if (!row) {
        row = new Map<MovementToken, number>();
        counts.set(key, row);
      }
      row.set(next, (row.get(next) ?? 0) + 1);
    };

    for (const sequence of dataset.sequences) {
      const padded = [...Array<MovementToken>(order).fill(START), ...sequence.tokens, END];
      for (let position = order; position < padded.length; position += 1) {
        const next = padded[position]!;
        // Record all backoff orders 0..order for this position.
        for (let len = 0; len <= order; len += 1) {
          record(padded.slice(position - len, position), next);
        }
      }
    }

    return new NgramMovementModel(order, [...dataset.vocabulary], counts);
  }

  restore(serialized: SerializedMovementModel): NgramMovementModel {
    const counts = new Map<string, Map<MovementToken, number>>();
    for (const [key, row] of Object.entries(serialized.counts)) {
      counts.set(key, new Map(Object.entries(row)));
    }
    return new NgramMovementModel(serialized.order, [...serialized.vocabulary], counts);
  }
}

// --------------------------------------------------------------------------
// Dataset construction from replay manifests
// --------------------------------------------------------------------------

/** Minimal structural shape shared by ReplayManifest and ExportedReplayManifest. */
export type ReplayActionLike = {
  kind: string;
  ts: number;
  tool?: string;
  summary?: string;
  trajectoryId?: string;
};

export type ReplayLike = {
  sessionId?: string;
  trajectoryIds?: string[];
  events: ReplayActionLike[];
};

export type BuildMovementDatasetOptions = {
  /** `tool` = coarse/general; `tool-summary` = fine/repeatable. Default fine. */
  granularity?: "tool" | "tool-summary";
  /** One sequence per replay, or per trajectory within a replay. Default replay. */
  groupBy?: "replay" | "trajectory";
};

function tokenizeAction(event: ReplayActionLike, granularity: "tool" | "tool-summary"): MovementToken {
  const tool = (event.tool ?? "unknown").trim() || "unknown";
  if (granularity === "tool") {
    return tool;
  }
  const summary = (event.summary ?? "").trim();
  return summary ? `${tool}${TOKEN_JOIN}${summary}` : tool;
}

/** Derive a training-ready {@link MovementDataset} from replay manifests. */
export function buildMovementDataset(
  replays: ReplayLike[],
  options: BuildMovementDatasetOptions = {},
): MovementDataset {
  const granularity = options.granularity ?? "tool-summary";
  const groupBy = options.groupBy ?? "replay";
  const sequences: MovementSequence[] = [];

  for (let index = 0; index < replays.length; index += 1) {
    const replay = replays[index]!;
    const actions = replay.events
      .filter((event) => event.kind === "action")
      .slice()
      .sort((a, b) => a.ts - b.ts);

    if (groupBy === "trajectory") {
      const byTrajectory = new Map<string, ReplayActionLike[]>();
      for (const action of actions) {
        const key = action.trajectoryId ?? replay.sessionId ?? `replay-${index}`;
        const bucket = byTrajectory.get(key) ?? [];
        bucket.push(action);
        byTrajectory.set(key, bucket);
      }
      for (const [key, bucket] of byTrajectory) {
        if (bucket.length > 0) {
          sequences.push({ id: key, tokens: bucket.map((event) => tokenizeAction(event, granularity)) });
        }
      }
    } else if (actions.length > 0) {
      sequences.push({
        id: replay.sessionId ?? replay.trajectoryIds?.[0] ?? `replay-${index}`,
        tokens: actions.map((event) => tokenizeAction(event, granularity)),
      });
    }
  }

  const vocabulary = [...new Set(sequences.flatMap((sequence) => sequence.tokens))].sort();
  return { version: 1, sequences, vocabulary };
}

// --------------------------------------------------------------------------
// Synthetic event-stream generator (validates the pipeline without real OS input)
// --------------------------------------------------------------------------

/** A tiny deterministic LCG so synthetic data is reproducible across runs. */
function lcg(seed: number): () => number {
  let state = (seed >>> 0) || 1;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

export type SyntheticMovementOptions = {
  count: number;
  seed: number;
  /** Base action alphabet; sequences follow a learnable deterministic chain. */
  vocabulary: MovementToken[];
  minLength?: number;
  maxLength?: number;
  idPrefix?: string;
};

/**
 * Generate related synthetic movement sequences that follow a fixed
 * deterministic "next" chain (token i ⇒ token i+1). Because every sequence
 * obeys the same transition rule, a model trained on one split should predict
 * next tokens on a held-out split — the crisp generalization signal the eval
 * harness measures.
 */
export function generateSyntheticMovementSequences(options: SyntheticMovementOptions): MovementSequence[] {
  const { vocabulary } = options;
  if (vocabulary.length === 0) {
    return [];
  }
  const rand = lcg(options.seed);
  const minLength = Math.max(2, options.minLength ?? 3);
  const maxLength = Math.max(minLength, options.maxLength ?? 6);
  const prefix = options.idPrefix ?? "synthetic";
  const sequences: MovementSequence[] = [];

  for (let i = 0; i < options.count; i += 1) {
    const length = minLength + Math.floor(rand() * (maxLength - minLength + 1));
    // Anchor every sequence at the chain's head so the START→first-token
    // transition is deterministic and the chain is fully learnable.
    let index = 0;
    const tokens: MovementToken[] = [];
    for (let step = 0; step < length; step += 1) {
      tokens.push(vocabulary[index]!);
      index = (index + 1) % vocabulary.length; // deterministic learnable chain
    }
    sequences.push({ id: `${prefix}-${i}`, tokens });
  }
  return sequences;
}

export function toMovementDataset(sequences: MovementSequence[]): MovementDataset {
  const vocabulary = [...new Set(sequences.flatMap((sequence) => sequence.tokens))].sort();
  return { version: 1, sequences, vocabulary };
}

// --------------------------------------------------------------------------
// Generalization eval harness
// --------------------------------------------------------------------------

export type MovementEvalResult = {
  sequenceCount: number;
  /** Next-token predictions scored (one per non-first position + END). */
  predictionCount: number;
  /** Top-1 next-token accuracy on held-out sequences (generalization). */
  nextTokenAccuracy: number;
  /** Top-K next-token accuracy. */
  topKAccuracy: number;
  /** exp(mean cross-entropy) over held-out next tokens; lower is better. */
  perplexity: number;
  /** Fraction of sequences the model regenerates exactly from the start (repeat). */
  exactReplayRate: number;
};

export type EvaluateMovementModelOptions = {
  topK?: number;
  /** Whether to include the END stop token in accuracy scoring. Default true. */
  scoreEnd?: boolean;
};

/**
 * Measure how well a trained model repeats and generalizes. `nextTokenAccuracy`
 * captures generalization on held-out sequences; `exactReplayRate` captures
 * faithful reproduction of memorized ones.
 */
export function evaluateMovementModel(
  model: TrainedMovementModel,
  heldOut: MovementSequence[],
  options: EvaluateMovementModelOptions = {},
): MovementEvalResult {
  const topK = options.topK ?? 3;
  const scoreEnd = options.scoreEnd ?? true;
  let predictionCount = 0;
  let top1 = 0;
  let topKHits = 0;
  let logProbSum = 0;
  let exactReplays = 0;

  for (const sequence of heldOut) {
    const targets = scoreEnd ? [...sequence.tokens, END] : [...sequence.tokens];
    for (let position = 0; position < targets.length; position += 1) {
      const prefix = sequence.tokens.slice(0, position);
      const target = targets[position]!;
      const ranked = model.predictNext(prefix, topK);
      predictionCount += 1;
      if (ranked[0]?.token === target) {
        top1 += 1;
      }
      if (ranked.some((entry) => entry.token === target)) {
        topKHits += 1;
      }
      const probability =
        model instanceof NgramMovementModel
          ? model.probabilityOf(prefix, target)
          : (ranked.find((entry) => entry.token === target)?.probability ?? Number.EPSILON);
      logProbSum += Math.log(Math.max(probability, Number.EPSILON));
    }

    // "Repeat" = given the first recorded action, does the model reproduce the
    // rest of the recorded movement? (Generating from nothing can only ever
    // yield one sequence when several sequences share the start context.)
    if (sequence.tokens.length === 0) {
      exactReplays += 1;
    } else {
      const seed = sequence.tokens.slice(0, 1);
      const expected = sequence.tokens.slice(1);
      const continuation = model.generate({ seed, maxLength: expected.length + 4 });
      if (continuation.length === expected.length && continuation.every((token, i) => token === expected[i])) {
        exactReplays += 1;
      }
    }
  }

  return {
    sequenceCount: heldOut.length,
    predictionCount,
    nextTokenAccuracy: predictionCount > 0 ? top1 / predictionCount : 0,
    topKAccuracy: predictionCount > 0 ? topKHits / predictionCount : 0,
    perplexity: predictionCount > 0 ? Math.exp(-logProbSum / predictionCount) : 0,
    exactReplayRate: heldOut.length > 0 ? exactReplays / heldOut.length : 0,
  };
}
