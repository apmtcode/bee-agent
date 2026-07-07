import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";

/**
 * In-process, pluggable movement-model backend for the local-movement learning
 * subsystem (standing objective #2, pieces (c) train and (d) generalize).
 *
 * The training {@link LocalAppleSiliconTrainingRunner} emits launch scripts for
 * external tools (mlx/axolotl) that can only run on a real Apple-silicon device.
 * That is the production on-device path, but it cannot be exercised or validated
 * in the cloud/CI. This module provides a *backend seam* plus a deterministic,
 * dependency-free reference backend so the capture → dataset → train → infer
 * loop can be validated end-to-end with synthetic event streams, and so a real
 * on-device small model can be dropped in behind the same interface.
 */

export const MOVEMENT_BOS = "<bos>";
export const MOVEMENT_EOS = "<eos>";

/**
 * A discrete movement token — the salient, learnable feature of a timeline
 * event. Continuous detail (free-text summaries, exact timestamps) is dropped so
 * the model learns *movement structure* (which action follows which) rather than
 * memorising prose. `<bos>`/`<eos>` mark sequence boundaries.
 */
export type MovementToken = string;

/** A single recorded movement sequence, ready for training. */
export type MovementSequence = {
  /** Optional provenance (trajectory/session id) for eval reporting. */
  id?: string;
  tokens: MovementToken[];
};

export type MovementModelConfig = {
  /** Max Markov order (context length). Higher = more specific, less general. */
  order: number;
  /** Add-alpha (Laplace) smoothing mass. Must be > 0 for well-defined scoring. */
  alpha: number;
};

export const DEFAULT_MOVEMENT_MODEL_CONFIG: MovementModelConfig = {
  order: 2,
  alpha: 0.01,
};

export type MovementPrediction = {
  /** Most likely next token (argmax; ties broken by lexical token order). */
  token: MovementToken;
  probability: number;
  /** Full smoothed distribution, sorted by descending probability. */
  distribution: Array<{ token: MovementToken; probability: number }>;
  /**
   * Which context order actually produced the prediction after Katz-style
   * backoff. `order` when the full context matched; lower when the model backed
   * off to a shorter (more general) context; 0 for the unigram prior.
   */
  backoffOrder: number;
};

export type MovementGenerateOptions = {
  /** Hard cap on generated tokens (excluding the seed). */
  maxSteps: number;
  /**
   * 0 (default) = greedy/deterministic argmax. > 0 enables seeded sampling from
   * the smoothed distribution using a deterministic PRNG (no global RNG), so
   * output is still reproducible for a given `seedValue`.
   */
  temperature?: number;
  seedValue?: number;
};

export type MovementModelStats = {
  backend: string;
  order: number;
  vocabulary: number;
  sequences: number;
  observedTokens: number;
};

/** A trained movement model: repeats recorded movements and generalises. */
export interface TrainedMovementModel {
  readonly stats: MovementModelStats;
  /** Predict the next movement given a (possibly long) context of tokens. */
  predictNext(context: MovementToken[]): MovementPrediction;
  /** Autoregressively continue from a seed until `<eos>` or `maxSteps`. */
  generate(seed: MovementToken[], options: MovementGenerateOptions): MovementToken[];
  /** Log-probability the model assigns to a full sequence (higher = better fit). */
  scoreSequence(tokens: MovementToken[]): number;
}

/** Pluggable training backend. Real on-device models implement this too. */
export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementSequence[], config: MovementModelConfig): TrainedMovementModel;
}

/**
 * Convert timeline events into a movement token stream. `<bos>`/`<eos>` bracket
 * the sequence so the model learns how movements begin and end.
 */
export function tokenizeEvents(events: ReplayTimelineEvent[]): MovementToken[] {
  const body = events.map(movementTokenForEvent);
  return [MOVEMENT_BOS, ...body, MOVEMENT_EOS];
}

/** Convenience: tokenize a whole replay manifest into one movement sequence. */
export function tokenizeReplayManifest(manifest: ReplayManifest): MovementSequence {
  return { id: manifest.sessionId, tokens: tokenizeEvents(manifest.events) };
}

export function movementTokenForEvent(event: ReplayTimelineEvent): MovementToken {
  switch (event.kind) {
    case "action":
      return `action:${normalizeFeature(event.tool)}`;
    case "observation":
      return `observation:${normalizeFeature(event.source)}`;
    case "transcript":
      return `transcript:${normalizeFeature(event.role)}`;
  }
}

function normalizeFeature(value: string): string {
  const normalized = value.trim().toLowerCase().replace(/\s+/g, "-");
  return normalized.length > 0 ? normalized : "unknown";
}

/**
 * Deterministic order-N Markov backend with add-alpha smoothing and Katz-style
 * backoff. It learns transition statistics from recorded movements (so it can
 * *repeat* them) and backs off to shorter contexts for unseen situations (so it
 * *generalises* to new-but-related movements) — all without any native deps,
 * network, or randomness, which makes the whole loop testable in the cloud.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly name = "markov";

  train(dataset: MovementSequence[], config: MovementModelConfig): TrainedMovementModel {
    return new MarkovMovementModel(dataset, config);
  }
}

class MarkovMovementModel implements TrainedMovementModel {
  readonly stats: MovementModelStats;
  private readonly order: number;
  private readonly alpha: number;
  private readonly vocab: MovementToken[];
  private readonly vocabIndex: Map<MovementToken, number>;
  /** contexts[k] maps a k-token context key -> (nextToken -> count). k in 1..order; contexts[0] is the unigram row. */
  private readonly contexts: Array<Map<string, Map<MovementToken, number>>>;

  constructor(dataset: MovementSequence[], config: MovementModelConfig) {
    this.order = Math.max(1, Math.floor(config.order));
    this.alpha = config.alpha > 0 ? config.alpha : DEFAULT_MOVEMENT_MODEL_CONFIG.alpha;

    const vocab = new Set<MovementToken>([MOVEMENT_BOS, MOVEMENT_EOS]);
    this.contexts = Array.from({ length: this.order + 1 }, () => new Map<string, Map<MovementToken, number>>());

    let observedTokens = 0;
    for (const sequence of dataset) {
      const tokens = normalizeTrainingSequence(sequence.tokens);
      for (const token of tokens) {
        vocab.add(token);
      }
      // Predict token at position i from the preceding up-to-`order` tokens.
      for (let i = 1; i < tokens.length; i += 1) {
        const next = tokens[i]!;
        observedTokens += 1;
        for (let k = 0; k <= this.order; k += 1) {
          if (i - k < 0) {
            break;
          }
          const contextKey = k === 0 ? "" : tokens.slice(i - k, i).join("");
          this.increment(k, contextKey, next);
        }
      }
    }

    this.vocab = [...vocab].sort();
    this.vocabIndex = new Map(this.vocab.map((token, index) => [token, index]));
    this.stats = {
      backend: "markov",
      order: this.order,
      vocabulary: this.vocab.length,
      sequences: dataset.length,
      observedTokens,
    };
  }

  predictNext(context: MovementToken[]): MovementPrediction {
    const { counts, backoffOrder } = this.resolveContext(context);
    const total = sumCounts(counts);
    const denominator = total + this.alpha * this.vocab.length;

    const distribution = this.vocab
      .map((token) => ({
        token,
        probability: ((counts.get(token) ?? 0) + this.alpha) / denominator,
      }))
      .sort((a, b) => {
        if (b.probability !== a.probability) {
          return b.probability - a.probability;
        }
        return a.token < b.token ? -1 : a.token > b.token ? 1 : 0;
      });

    const top = distribution[0]!;
    return { token: top.token, probability: top.probability, distribution, backoffOrder };
  }

  generate(seed: MovementToken[], options: MovementGenerateOptions): MovementToken[] {
    const maxSteps = Math.max(0, Math.floor(options.maxSteps));
    const temperature = options.temperature ?? 0;
    const context = normalizeSequence(seed.length > 0 ? seed : [MOVEMENT_BOS]);
    const produced: MovementToken[] = [];
    let rngState = (options.seedValue ?? 0x9e3779b9) >>> 0;

    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predictNext(context);
      let chosen: MovementToken;
      if (temperature > 0) {
        rngState = nextRng(rngState);
        chosen = sampleFromDistribution(prediction.distribution, temperature, rngState / 0xffffffff);
      } else {
        chosen = prediction.token;
      }
      if (chosen === MOVEMENT_EOS) {
        break;
      }
      produced.push(chosen);
      context.push(chosen);
    }
    return produced;
  }

  scoreSequence(tokens: MovementToken[]): number {
    const normalized = normalizeSequence(tokens);
    let logProb = 0;
    for (let i = 1; i < normalized.length; i += 1) {
      const prediction = this.predictNext(normalized.slice(0, i));
      const entry = prediction.distribution.find((candidate) => candidate.token === normalized[i]);
      logProb += Math.log(entry?.probability ?? Number.EPSILON);
    }
    return logProb;
  }

  private increment(order: number, contextKey: string, next: MovementToken): void {
    const table = this.contexts[order]!;
    let row = table.get(contextKey);
    if (!row) {
      row = new Map<MovementToken, number>();
      table.set(contextKey, row);
    }
    row.set(next, (row.get(next) ?? 0) + 1);
  }

  /** Katz-style backoff: use the longest context (up to `order`) that was observed. */
  private resolveContext(context: MovementToken[]): { counts: Map<MovementToken, number>; backoffOrder: number } {
    const normalized = normalizeSequence(context);
    for (let k = Math.min(this.order, normalized.length); k >= 1; k -= 1) {
      const contextKey = normalized.slice(normalized.length - k).join("");
      const counts = this.contexts[k]!.get(contextKey);
      if (counts && counts.size > 0) {
        return { counts, backoffOrder: k };
      }
    }
    return { counts: this.contexts[0]!.get("") ?? new Map(), backoffOrder: 0 };
  }
}

/** Prediction/generation context: ensure a leading `<bos>` only. */
function normalizeSequence(tokens: MovementToken[]): MovementToken[] {
  const trimmed = tokens.length > 0 && tokens[0] === MOVEMENT_BOS ? tokens.slice() : [MOVEMENT_BOS, ...tokens];
  return trimmed;
}

/**
 * Training ingestion: a recorded movement sequence is *complete*, so bracket it
 * with both `<bos>` and `<eos>` (idempotently) so the model learns how movements
 * both begin and terminate. Datasets built via {@link tokenizeEvents} are
 * already bracketed; this makes hand-authored token lists behave identically.
 */
function normalizeTrainingSequence(tokens: MovementToken[]): MovementToken[] {
  const withStart = normalizeSequence(tokens);
  if (withStart[withStart.length - 1] === MOVEMENT_EOS) {
    return withStart;
  }
  return [...withStart, MOVEMENT_EOS];
}

function sumCounts(counts: Map<MovementToken, number>): number {
  let total = 0;
  for (const value of counts.values()) {
    total += value;
  }
  return total;
}

function nextRng(state: number): number {
  // xorshift32 — deterministic, no global RNG.
  let x = state === 0 ? 0x9e3779b9 : state;
  x ^= x << 13;
  x ^= x >>> 17;
  x ^= x << 5;
  return x >>> 0;
}

function sampleFromDistribution(
  distribution: Array<{ token: MovementToken; probability: number }>,
  temperature: number,
  roll: number,
): MovementToken {
  const weights = distribution.map((entry) => Math.pow(entry.probability, 1 / temperature));
  const total = weights.reduce((sum, weight) => sum + weight, 0);
  let cumulative = 0;
  const target = roll * total;
  for (let i = 0; i < distribution.length; i += 1) {
    cumulative += weights[i]!;
    if (target <= cumulative) {
      return distribution[i]!.token;
    }
  }
  return distribution[distribution.length - 1]!.token;
}

/** Train a model with the deterministic Markov backend (or a supplied backend). */
export function trainMovementModel(
  dataset: MovementSequence[],
  config: Partial<MovementModelConfig> = {},
  backend: MovementModelBackend = new MarkovMovementBackend(),
): TrainedMovementModel {
  return backend.train(dataset, { ...DEFAULT_MOVEMENT_MODEL_CONFIG, ...config });
}

export type MovementEvalResult = {
  sequences: number;
  predictedTokens: number;
  correct: number;
  /** Next-movement top-1 accuracy on held-out sequences (replay fidelity). */
  accuracy: number;
  /** Mean per-token log-probability (higher = better generalisation). */
  meanLogProb: number;
};

/**
 * Generalization eval harness: measure how well a trained model reproduces
 * held-out (but related) movement sequences, both by greedy next-token accuracy
 * and by average log-probability. Feeds the roadmap's fidelity metric.
 */
export function evaluateMovementModel(
  model: TrainedMovementModel,
  heldOut: MovementSequence[],
): MovementEvalResult {
  let predictedTokens = 0;
  let correct = 0;
  let totalLogProb = 0;
  for (const sequence of heldOut) {
    const tokens = normalizeSequence(sequence.tokens);
    for (let i = 1; i < tokens.length; i += 1) {
      const prediction = model.predictNext(tokens.slice(0, i));
      predictedTokens += 1;
      if (prediction.token === tokens[i]) {
        correct += 1;
      }
      const entry = prediction.distribution.find((candidate) => candidate.token === tokens[i]);
      totalLogProb += Math.log(entry?.probability ?? Number.EPSILON);
    }
  }
  return {
    sequences: heldOut.length,
    predictedTokens,
    correct,
    accuracy: predictedTokens > 0 ? correct / predictedTokens : 0,
    meanLogProb: predictedTokens > 0 ? totalLogProb / predictedTokens : 0,
  };
}
