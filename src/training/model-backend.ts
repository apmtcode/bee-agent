import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

/**
 * Pluggable local-movement model backend.
 *
 * The training runner (`runner.ts`) emits shell *plans* that launch real
 * on-device trainers (mlx / axolotl) when bee-agent runs on the user's machine.
 * That path cannot execute in the cloud. This module provides the missing
 * counterpart: an in-process, deterministic model backend that actually
 * *trains* on recorded movement sequences and *infers* new ones, so the
 * capture -> dataset -> train -> replay/generalize loop (standing objective #2,
 * parts c and d) can be validated end-to-end with synthetic data in CI.
 *
 * The `MovementModelBackend` interface is the seam: the bundled
 * `MarkovMovementBackend` is a small, fully local model (an order-k Markov
 * policy with stupid-backoff smoothing) that generalizes to unseen-but-related
 * contexts via backoff. A real on-device small model implements the same
 * interface without any caller changes.
 */

export type MovementToken = string;

/** Sentinel prepended to every sequence so the model can learn openings. */
export const SEQUENCE_START: MovementToken = "<s>";
/** Sentinel appended to every sequence so generation knows when to stop. */
export const SEQUENCE_END: MovementToken = "</s>";

/** Field separator for context keys; a control char no real token contains. */
const CONTEXT_SEPARATOR = "␟";

export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

export type MovementDataset = {
  sequences: MovementSequence[];
};

export type MovementPrediction = {
  token: MovementToken;
  /** Normalized probability of the chosen token across the vocabulary. */
  probability: number;
  /** Length of the matched context after backoff (0 = unigram fallback). */
  contextOrder: number;
};

export type MovementModelMetadata = {
  backend: string;
  order: number;
  vocabularySize: number;
  sequenceCount: number;
  tokenCount: number;
};

export interface TrainedMovementModel {
  readonly metadata: MovementModelMetadata;
  /** Learned vocabulary, excluding the START sentinel, sorted for determinism. */
  vocabulary(): MovementToken[];
  /** Deterministic argmax next-token prediction for a (possibly novel) context. */
  predictNext(context: MovementToken[]): MovementPrediction | undefined;
  /** Greedily roll out a movement sequence from an optional prompt. */
  generate(params: { prompt?: MovementToken[]; maxTokens: number; stopToken?: MovementToken }): MovementToken[];
  /** Total smoothed log-probability of a full observed sequence (base e). */
  sequenceLogProb(sequence: MovementToken[]): number;
}

export type TrainMovementModelParams = {
  dataset: MovementDataset;
};

export interface MovementModelBackend {
  readonly name: string;
  train(params: TrainMovementModelParams): Promise<TrainedMovementModel>;
}

// --- Tokenization helpers -------------------------------------------------

/**
 * Default action tokenizer. Uses the tool plus a coarse direction/gesture hint
 * from metadata when present, so distinct movements map to distinct tokens
 * without leaking raw high-cardinality coordinates.
 */
export function movementTokenFromAction(action: TrajectoryAction): MovementToken {
  const hint = action.metadata?.["direction"] ?? action.metadata?.["gesture"];
  if (typeof hint === "string" && hint.trim().length > 0) {
    return `${action.tool}:${hint.trim()}`;
  }
  return action.tool;
}

export function trajectoryToSequence(
  span: TrajectorySpan,
  tokenize: (action: TrajectoryAction) => MovementToken = movementTokenFromAction,
): MovementSequence {
  return {
    id: span.id,
    tokens: span.actions.map((action) => tokenize(action)),
  };
}

export function buildMovementDataset(
  spans: TrajectorySpan[],
  tokenize: (action: TrajectoryAction) => MovementToken = movementTokenFromAction,
): MovementDataset {
  return {
    sequences: spans
      .map((span) => trajectoryToSequence(span, tokenize))
      .filter((sequence) => sequence.tokens.length > 0),
  };
}

// --- Markov backend -------------------------------------------------------

export type MarkovMovementBackendOptions = {
  /** Maximum context length (n-gram order). Clamped to >= 1. Default 3. */
  order?: number;
  /** Multiplicative penalty applied each time the model backs off. Default 0.4. */
  backoffDiscount?: number;
};

type ContextNode = {
  total: number;
  counts: Map<MovementToken, number>;
};

export class MarkovMovementBackend implements MovementModelBackend {
  readonly name = "markov";
  private readonly order: number;
  private readonly backoffDiscount: number;

  constructor(options: MarkovMovementBackendOptions = {}) {
    this.order = Math.max(1, Math.floor(options.order ?? 3));
    this.backoffDiscount = options.backoffDiscount ?? 0.4;
  }

  async train(params: TrainMovementModelParams): Promise<TrainedMovementModel> {
    // tables[k] maps a k-length context key -> next-token counts.
    const tables: Array<Map<string, ContextNode>> = [];
    for (let k = 0; k <= this.order; k += 1) {
      tables.push(new Map());
    }
    const vocabulary = new Set<MovementToken>();
    let tokenCount = 0;

    for (const sequence of params.dataset.sequences) {
      const padded = [
        ...Array.from({ length: this.order }, () => SEQUENCE_START),
        ...sequence.tokens,
        SEQUENCE_END,
      ];
      for (const token of sequence.tokens) {
        vocabulary.add(token);
      }
      vocabulary.add(SEQUENCE_END);

      for (let i = this.order; i < padded.length; i += 1) {
        const next = padded[i]!;
        tokenCount += 1;
        for (let k = 0; k <= this.order; k += 1) {
          const context = padded.slice(i - k, i);
          const key = context.join(CONTEXT_SEPARATOR);
          const table = tables[k]!;
          let node = table.get(key);
          if (!node) {
            node = { total: 0, counts: new Map() };
            table.set(key, node);
          }
          node.total += 1;
          node.counts.set(next, (node.counts.get(next) ?? 0) + 1);
        }
      }
    }

    return new MarkovMovementModel({
      order: this.order,
      backoffDiscount: this.backoffDiscount,
      tables,
      vocabulary,
      tokenCount,
      sequenceCount: params.dataset.sequences.length,
      backendName: this.name,
    });
  }
}

type MarkovModelState = {
  order: number;
  backoffDiscount: number;
  tables: Array<Map<string, ContextNode>>;
  vocabulary: Set<MovementToken>;
  tokenCount: number;
  sequenceCount: number;
  backendName: string;
};

class MarkovMovementModel implements TrainedMovementModel {
  readonly metadata: MovementModelMetadata;
  private readonly state: MarkovModelState;
  private readonly sortedVocab: MovementToken[];

  constructor(state: MarkovModelState) {
    this.state = state;
    // Vocabulary excludes the START sentinel (never a prediction target).
    this.sortedVocab = [...state.vocabulary].filter((token) => token !== SEQUENCE_START).sort();
    this.metadata = {
      backend: state.backendName,
      order: state.order,
      vocabularySize: this.sortedVocab.length,
      sequenceCount: state.sequenceCount,
      tokenCount: state.tokenCount,
    };
  }

  vocabulary(): MovementToken[] {
    return [...this.sortedVocab];
  }

  /**
   * Stupid-backoff conditional score. Not normalized across the vocabulary, but
   * always strictly positive (the unigram base case is add-1 smoothed), so log
   * scores are always finite. Also reports the deepest context length that
   * produced a direct (non-backoff) hit.
   */
  private scoreWithOrder(context: MovementToken[], token: MovementToken): { score: number; order: number } {
    const maxK = Math.min(context.length, this.state.order);
    for (let k = maxK; k >= 1; k -= 1) {
      const ctx = context.slice(context.length - k);
      const node = this.state.tables[k]!.get(ctx.join(CONTEXT_SEPARATOR));
      if (node) {
        const count = node.counts.get(token) ?? 0;
        if (count > 0) {
          const penalty = this.state.backoffDiscount ** (maxK - k);
          return { score: penalty * (count / node.total), order: k };
        }
      }
    }
    // Unigram add-1 base case — guarantees a positive floor for unseen tokens.
    const unigram = this.state.tables[0]!.get("");
    const rawCount = unigram?.counts.get(token) ?? 0;
    const total = unigram?.total ?? 0;
    const vocabSize = this.sortedVocab.length || 1;
    const penalty = this.state.backoffDiscount ** maxK;
    return { score: penalty * ((rawCount + 1) / (total + vocabSize)), order: 0 };
  }

  predictNext(context: MovementToken[]): MovementPrediction | undefined {
    if (this.sortedVocab.length === 0) {
      return undefined;
    }
    let best: { token: MovementToken; score: number; order: number } | undefined;
    let scoreSum = 0;
    for (const token of this.sortedVocab) {
      const { score, order } = this.scoreWithOrder(context, token);
      scoreSum += score;
      // Deterministic tie-break: higher score wins; equal scores keep the
      // lexicographically smaller token (sortedVocab is ascending, so the
      // first-seen best is already smallest).
      if (!best || score > best.score) {
        best = { token, score, order };
      }
    }
    if (!best) {
      return undefined;
    }
    return {
      token: best.token,
      probability: scoreSum > 0 ? best.score / scoreSum : 0,
      contextOrder: best.order,
    };
  }

  generate(params: { prompt?: MovementToken[]; maxTokens: number; stopToken?: MovementToken }): MovementToken[] {
    const stop = params.stopToken ?? SEQUENCE_END;
    const history: MovementToken[] = [
      ...Array.from({ length: this.state.order }, () => SEQUENCE_START),
      ...(params.prompt ?? []),
    ];
    const generated: MovementToken[] = [];
    for (let i = 0; i < params.maxTokens; i += 1) {
      const prediction = this.predictNext(history);
      if (!prediction || prediction.token === stop) {
        break;
      }
      generated.push(prediction.token);
      history.push(prediction.token);
    }
    return generated;
  }

  sequenceLogProb(sequence: MovementToken[]): number {
    const history: MovementToken[] = Array.from({ length: this.state.order }, () => SEQUENCE_START);
    const tokens = [...sequence, SEQUENCE_END];
    let logProb = 0;
    for (const token of tokens) {
      const { score } = this.scoreWithOrder(history, token);
      logProb += Math.log(score);
      history.push(token);
    }
    return logProb;
  }
}
