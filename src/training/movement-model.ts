import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Local-movement learning model (standing objective #2, pieces c + d).
 *
 * A movement model learns to *repeat* recorded movement sequences and to
 * *generalize* to new-but-related ones. Movements are reduced to a stream of
 * string tokens (one per action); the model learns the conditional
 * distribution of the next token given a bounded context of prior tokens.
 *
 * - "Repeat": rolling out predictions from a seen prefix reproduces the
 *   recorded sequence.
 * - "Generalize": an n-gram model with back-off predicts a plausible next
 *   token for prefixes it never saw verbatim, by falling back to shorter
 *   contexts it did see.
 *
 * The concrete learner is pluggable behind {@link LocalMovementModelBackend}.
 * The default {@link MarkovMovementBackend} is fully deterministic so it can be
 * trained and evaluated in the cloud/CI with synthetic data; a real on-device
 * small model can implement the same interface (see {@link createMovementModelBackend}).
 */

/** Sentinel tokens. Chosen to be extremely unlikely to collide with real tokens. */
export const MOVEMENT_BOS = "\u0002bos";
export const MOVEMENT_EOS = "\u0003eos";
const CONTEXT_SEP = "\u0001";

export type MovementSequence = {
  id: string;
  tokens: string[];
};

export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

/** A JSON-serializable trained model. */
export type LocalMovementModel = {
  version: 1;
  backendId: string;
  order: number;
  vocabulary: string[];
  sequenceCount: number;
  tokenCount: number;
  /**
   * Per-order transition counts. `transitions[k]` maps a context of the last
   * `k` tokens (joined by {@link CONTEXT_SEP}, `""` for the unigram order 0) to
   * a map of `nextToken -> count`.
   */
  transitions: Record<string, Record<string, Record<string, number>>>;
};

export type MovementAlternative = {
  token: string;
  probability: number;
};

export type MovementPrediction = {
  /** Highest-probability next token, or `null` if the model is empty. */
  token: string | null;
  probability: number;
  /** Length of the context that produced the prediction (after back-off). */
  contextOrder: number;
  /** Whether the predicted token is the end-of-sequence sentinel. */
  isEnd: boolean;
  alternatives: MovementAlternative[];
};

export type MovementTrainOptions = {
  /** Maximum n-gram context length. Defaults to the backend's configured order. */
  order?: number;
};

export type MovementGenerateOptions = {
  /** Hard cap on generated tokens (excluding the seed). Defaults to 64. */
  maxSteps?: number;
};

/**
 * Pluggable local-model backend. Implement this to swap the deterministic mock
 * for a real on-device small model (e.g. an MLX/GGUF sequence model). `train`
 * is async so a real backend can shell out to a trainer; the mock resolves
 * synchronously.
 */
export interface LocalMovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainOptions): Promise<LocalMovementModel>;
  predictNext(model: LocalMovementModel, context: readonly string[]): MovementPrediction;
  generate(model: LocalMovementModel, seed: readonly string[], options?: MovementGenerateOptions): string[];
}

/**
 * Deterministic n-gram Markov backend with stupid-backoff prediction.
 * Ties are broken by count then lexicographic token order, so training and
 * inference are reproducible across runs and machines.
 */
export class MarkovMovementBackend implements LocalMovementModelBackend {
  readonly id: string;
  private readonly order: number;

  constructor(options: { order?: number } = {}) {
    this.order = Math.max(1, Math.floor(options.order ?? 2));
    this.id = `markov-${this.order}`;
  }

  async train(dataset: MovementDataset, options: MovementTrainOptions = {}): Promise<LocalMovementModel> {
    const order = Math.max(1, Math.floor(options.order ?? this.order));
    const transitions: Record<string, Record<string, Record<string, number>>> = {};
    for (let k = 0; k <= order; k += 1) {
      transitions[String(k)] = {};
    }

    const vocabulary = new Set<string>();
    let tokenCount = 0;

    for (const sequence of dataset.sequences) {
      const padded = [...Array<string>(order).fill(MOVEMENT_BOS), ...sequence.tokens, MOVEMENT_EOS];
      for (const token of sequence.tokens) {
        vocabulary.add(token);
        tokenCount += 1;
      }
      for (let i = order; i < padded.length; i += 1) {
        const target = padded[i]!;
        for (let k = 0; k <= order; k += 1) {
          const context = padded.slice(i - k, i);
          const key = contextKey(context);
          const table = transitions[String(k)]!;
          const row = (table[key] ??= {});
          row[target] = (row[target] ?? 0) + 1;
        }
      }
    }

    return {
      version: 1,
      backendId: this.id,
      order,
      vocabulary: [...vocabulary].sort(),
      sequenceCount: dataset.sequences.length,
      tokenCount,
      transitions,
    };
  }

  predictNext(model: LocalMovementModel, context: readonly string[]): MovementPrediction {
    const padded = padContext(context, model.order);
    for (let k = model.order; k >= 0; k -= 1) {
      const suffix = padded.slice(padded.length - k);
      const table = model.transitions[String(k)];
      const row = table?.[contextKey(suffix)];
      if (!row) {
        continue;
      }
      const ranked = rankRow(row);
      if (ranked.length === 0) {
        continue;
      }
      const total = ranked.reduce((sum, entry) => sum + entry.count, 0);
      const best = ranked[0]!;
      return {
        token: best.token,
        probability: total > 0 ? best.count / total : 0,
        contextOrder: k,
        isEnd: best.token === MOVEMENT_EOS,
        alternatives: ranked.map((entry) => ({
          token: entry.token,
          probability: total > 0 ? entry.count / total : 0,
        })),
      };
    }
    return { token: null, probability: 0, contextOrder: 0, isEnd: false, alternatives: [] };
  }

  generate(
    model: LocalMovementModel,
    seed: readonly string[],
    options: MovementGenerateOptions = {},
  ): string[] {
    const maxSteps = Math.max(0, Math.floor(options.maxSteps ?? 64));
    const context = [...seed];
    const output: string[] = [];
    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predictNext(model, context);
      if (prediction.token === null || prediction.token === MOVEMENT_EOS) {
        break;
      }
      output.push(prediction.token);
      context.push(prediction.token);
    }
    return output;
  }
}

export type MovementBackendKind = "markov";

/**
 * Factory for the pluggable backend seam. Today only the deterministic
 * `markov` mock exists; a real on-device backend registers here later.
 */
export function createMovementModelBackend(
  kind: MovementBackendKind = "markov",
  options: { order?: number } = {},
): LocalMovementModelBackend {
  switch (kind) {
    case "markov":
      return new MarkovMovementBackend(options);
    default: {
      const exhaustive: never = kind;
      throw new Error(`unknown movement backend: ${String(exhaustive)}`);
    }
  }
}

/** Normalize a captured action into a stable movement token. */
export function movementActionToken(action: { tool: string; summary: string }): string {
  const summary = action.summary.trim().toLowerCase().replace(/\s+/g, " ");
  return `${action.tool.trim().toLowerCase()}:${summary}`;
}

export type BuildMovementDatasetOptions = {
  /** Only include spans with at least this many action tokens. Defaults to 1. */
  minTokens?: number;
};

/**
 * Derive a training dataset from captured trajectory spans by turning each
 * span's time-ordered actions into a movement token sequence. This is the
 * bridge from the capture schema (`src/capture`) into the training pipeline.
 */
export function buildMovementDatasetFromSpans(
  spans: readonly TrajectorySpan[],
  options: BuildMovementDatasetOptions = {},
): MovementDataset {
  const minTokens = Math.max(1, Math.floor(options.minTokens ?? 1));
  const sequences: MovementSequence[] = [];
  for (const span of spans) {
    const tokens = [...span.actions]
      .sort((a, b) => a.ts - b.ts)
      .map((action) => movementActionToken(action));
    if (tokens.length >= minTokens) {
      sequences.push({ id: span.id, tokens });
    }
  }
  return { version: 1, sequences };
}

function padContext(context: readonly string[], order: number): string[] {
  const padCount = Math.max(0, order - context.length);
  return [...Array<string>(padCount).fill(MOVEMENT_BOS), ...context];
}

function contextKey(context: readonly string[]): string {
  return context.join(CONTEXT_SEP);
}

function rankRow(row: Record<string, number>): { token: string; count: number }[] {
  return Object.entries(row)
    .map(([token, count]) => ({ token, count }))
    .sort((a, b) => (b.count !== a.count ? b.count - a.count : a.token.localeCompare(b.token)));
}
