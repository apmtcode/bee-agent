import type { ReviewedExportManifest } from "./export-manifest.js";

/**
 * Local-movement model backend (standing objective #2, parts c & d).
 *
 * The training {@link LocalAppleSiliconTrainingRunner} emits shell commands that
 * hand a reviewed dataset off to an external on-device trainer (MLX / Axolotl).
 * That seam only exists on the user's real machine. This module adds the missing
 * piece: a *pluggable, in-process* model backend that can actually **train on the
 * recorded movement dataset and perform inference** — predict the next movement to
 * repeat a recorded sequence, and generalize to unseen-but-related sequences.
 *
 * The default {@link MarkovMovementBackend} is a deterministic n-gram model with
 * stupid-backoff. It has no external dependencies and produces identical output
 * for identical input, so the full train -> infer -> generalize loop is exercised
 * by cloud/CI tests. A real small on-device model can implement the same
 * {@link MovementModelBackend} interface behind the identical seam.
 */

/** A single discrete movement, tokenized from a captured action/observation. */
export type MovementToken = string;

/** One recorded movement sequence (an ordered run of tokens from a trajectory). */
export type MovementSequence = {
  trajectoryId: string;
  sessionId: string;
  tokens: MovementToken[];
};

/** The training dataset: sequences plus the derived vocabulary. */
export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
  vocabulary: MovementToken[];
};

/** A ranked next-token candidate. */
export type MovementCandidate = {
  token: MovementToken;
  probability: number;
};

/** Result of a single next-movement inference. */
export type MovementPrediction = {
  /** Most-likely next token, or undefined if the model is empty. */
  token: MovementToken | undefined;
  /** Probability of {@link token} within the context actually used. */
  probability: number;
  /**
   * Context order that produced the prediction after backoff. Equals the model
   * order when the full context was seen in training; a lower value means the
   * model generalized by backing off to a shorter (seen) suffix; 0 is the
   * unigram prior (pure generalization to an entirely unseen context).
   */
  order: number;
  /** Ranked candidates for the chosen context, best first. */
  candidates: MovementCandidate[];
};

/** Serialized model — a plain JSON value safe to persist and restore. */
export type SerializedMovementModel = {
  backend: string;
  order: number;
  /** context-key -> (nextToken -> count). "" is the unigram bucket. */
  counts: Record<string, Record<MovementToken, number>>;
  vocabulary: MovementToken[];
  sequenceCount: number;
  tokenCount: number;
};

/** A trained model instance capable of inference. */
export interface TrainedMovementModel {
  readonly backend: string;
  readonly order: number;
  /** Predict the movement that most likely follows `context`. */
  predictNext(context: MovementToken[]): MovementPrediction;
  /**
   * Autoregressively roll out `length` movements starting from `seed`,
   * feeding each prediction back in. Deterministic (always takes argmax).
   */
  generate(seed: MovementToken[], length: number): MovementToken[];
  vocabulary(): MovementToken[];
  serialize(): SerializedMovementModel;
}

export type MovementTrainingOptions = {
  /** Max context length (n-gram order). Defaults to the backend default. */
  order?: number;
};

/** Pluggable backend contract — swap the mock for a real on-device model here. */
export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, options?: MovementTrainingOptions): Promise<TrainedMovementModel>;
  restore(serialized: SerializedMovementModel): TrainedMovementModel;
}

const CONTEXT_SEPARATOR = "␟"; // ␟ — unlikely to collide with token text.

/**
 * Default tokenizer: turn a reviewed replay `action` event into a movement
 * token. Uses the tool name as the primary signal so the model generalizes
 * across differing free-text summaries of the same movement.
 */
export function defaultActionTokenizer(event: { tool: string; summary: string }): MovementToken {
  return `action:${event.tool}`;
}

/**
 * Build a {@link MovementDataset} from a reviewed export manifest by extracting
 * the ordered `action` events of each replay into a token sequence. Replays are
 * already time-sorted by {@link buildReplayManifest}, so token order reflects the
 * recorded movement order.
 */
export function deriveMovementDataset(
  manifest: Pick<ReviewedExportManifest, "replays">,
  tokenize: (event: { tool: string; summary: string }) => MovementToken = defaultActionTokenizer,
): MovementDataset {
  const sequences: MovementSequence[] = [];
  const vocabulary = new Set<MovementToken>();

  for (const replay of manifest.replays) {
    const perTrajectory = new Map<string, MovementToken[]>();
    for (const event of replay.events) {
      if (event.kind !== "action") {
        continue;
      }
      const token = tokenize({ tool: event.tool, summary: event.summary });
      vocabulary.add(token);
      const bucket = perTrajectory.get(event.trajectoryId) ?? [];
      bucket.push(token);
      perTrajectory.set(event.trajectoryId, bucket);
    }
    for (const [trajectoryId, tokens] of perTrajectory) {
      if (tokens.length > 0) {
        sequences.push({ trajectoryId, sessionId: replay.sessionId, tokens });
      }
    }
  }

  return { version: 1, sequences, vocabulary: [...vocabulary].sort() };
}

function contextKey(context: MovementToken[]): string {
  return context.join(CONTEXT_SEPARATOR);
}

/**
 * Deterministic n-gram movement model with stupid-backoff.
 *
 * Training tallies, for every context length 0..order, how often each token
 * follows that context. Inference tries the longest context first and backs off
 * to shorter suffixes until a seen context is found — which is exactly what lets
 * the model *repeat* recorded movements (full context seen) yet still *generalize*
 * to novel prefixes (shorter suffix seen, or unigram prior).
 */
export class MarkovMovementModel implements TrainedMovementModel {
  readonly backend: string;
  readonly order: number;
  private readonly counts: Map<string, Map<MovementToken, number>>;
  private readonly vocab: MovementToken[];
  readonly sequenceCount: number;
  readonly tokenCount: number;

  constructor(params: {
    backend: string;
    order: number;
    counts: Map<string, Map<MovementToken, number>>;
    vocabulary: MovementToken[];
    sequenceCount: number;
    tokenCount: number;
  }) {
    this.backend = params.backend;
    this.order = params.order;
    this.counts = params.counts;
    this.vocab = params.vocabulary;
    this.sequenceCount = params.sequenceCount;
    this.tokenCount = params.tokenCount;
  }

  predictNext(context: MovementToken[]): MovementPrediction {
    for (let used = Math.min(context.length, this.order); used >= 0; used -= 1) {
      const suffix = used === 0 ? [] : context.slice(context.length - used);
      const bucket = this.counts.get(contextKey(suffix));
      if (!bucket || bucket.size === 0) {
        continue;
      }
      const candidates = rankCandidates(bucket);
      const total = [...bucket.values()].reduce((sum, count) => sum + count, 0);
      const best = candidates[0];
      return {
        token: best?.token,
        probability: best ? best.probability : 0,
        order: used,
        candidates: candidates.map((candidate) => ({
          token: candidate.token,
          probability: candidate.rawCount / total,
        })),
      };
    }
    return { token: undefined, probability: 0, order: 0, candidates: [] };
  }

  generate(seed: MovementToken[], length: number): MovementToken[] {
    const output: MovementToken[] = [];
    const window: MovementToken[] = [...seed];
    for (let step = 0; step < length; step += 1) {
      const prediction = this.predictNext(window);
      if (prediction.token === undefined) {
        break;
      }
      output.push(prediction.token);
      window.push(prediction.token);
    }
    return output;
  }

  vocabulary(): MovementToken[] {
    return [...this.vocab];
  }

  serialize(): SerializedMovementModel {
    const counts: Record<string, Record<MovementToken, number>> = {};
    for (const [key, bucket] of this.counts) {
      const serializedBucket: Record<MovementToken, number> = {};
      for (const [token, count] of bucket) {
        serializedBucket[token] = count;
      }
      counts[key] = serializedBucket;
    }
    return {
      backend: this.backend,
      order: this.order,
      counts,
      vocabulary: [...this.vocab],
      sequenceCount: this.sequenceCount,
      tokenCount: this.tokenCount,
    };
  }
}

type RankedCandidate = MovementCandidate & { rawCount: number };

function rankCandidates(bucket: Map<MovementToken, number>): RankedCandidate[] {
  const total = [...bucket.values()].reduce((sum, count) => sum + count, 0);
  return [...bucket.entries()]
    .map(([token, count]) => ({ token, rawCount: count, probability: count / total }))
    // Deterministic ordering: higher count first, then lexicographic token.
    .sort((a, b) => (b.rawCount !== a.rawCount ? b.rawCount - a.rawCount : a.token < b.token ? -1 : 1));
}

export const DEFAULT_MOVEMENT_MODEL_ORDER = 3;

/**
 * Deterministic in-process movement backend. Serves as the CI/cloud-safe mock
 * for the pluggable {@link MovementModelBackend} seam and as a genuinely useful
 * lightweight local policy.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly name = "markov";

  constructor(private readonly defaultOrder: number = DEFAULT_MOVEMENT_MODEL_ORDER) {}

  async train(dataset: MovementDataset, options?: MovementTrainingOptions): Promise<TrainedMovementModel> {
    const order = Math.max(1, options?.order ?? this.defaultOrder);
    const counts = new Map<string, Map<MovementToken, number>>();
    let tokenCount = 0;

    const tally = (context: MovementToken[], next: MovementToken): void => {
      const key = contextKey(context);
      const bucket = counts.get(key) ?? new Map<MovementToken, number>();
      bucket.set(next, (bucket.get(next) ?? 0) + 1);
      counts.set(key, bucket);
    };

    for (const sequence of dataset.sequences) {
      const tokens = sequence.tokens;
      tokenCount += tokens.length;
      for (let index = 0; index < tokens.length; index += 1) {
        const next = tokens[index]!;
        // Record this token under every context length 0..order.
        for (let ctxLen = 0; ctxLen <= order; ctxLen += 1) {
          if (index - ctxLen < 0) {
            break;
          }
          tally(tokens.slice(index - ctxLen, index), next);
        }
      }
    }

    return new MarkovMovementModel({
      backend: this.name,
      order,
      counts,
      vocabulary: [...dataset.vocabulary],
      sequenceCount: dataset.sequences.length,
      tokenCount,
    });
  }

  restore(serialized: SerializedMovementModel): TrainedMovementModel {
    const counts = new Map<string, Map<MovementToken, number>>();
    for (const [key, bucket] of Object.entries(serialized.counts)) {
      counts.set(key, new Map(Object.entries(bucket)));
    }
    return new MarkovMovementModel({
      backend: serialized.backend,
      order: serialized.order,
      counts,
      vocabulary: [...serialized.vocabulary],
      sequenceCount: serialized.sequenceCount,
      tokenCount: serialized.tokenCount,
    });
  }
}

/** Fidelity of a model's rollout against a recorded/held-out target sequence. */
export type ReplayFidelity = {
  /** Number of positions where the predicted next token matched the target. */
  matches: number;
  /** Total predictions scored. */
  total: number;
  /** matches / total (1.0 = perfect reproduction), 0 when nothing was scored. */
  accuracy: number;
};

/**
 * Teacher-forced next-movement accuracy: walk `target`, and at each step ask the
 * model to predict the next token given the true prefix. This measures how well
 * the model reproduces (or generalizes to) a sequence without compounding its own
 * errors — the basis of the generalization eval harness.
 */
export function scoreReplayFidelity(
  model: TrainedMovementModel,
  target: MovementToken[],
): ReplayFidelity {
  let matches = 0;
  let total = 0;
  for (let index = 1; index < target.length; index += 1) {
    const prediction = model.predictNext(target.slice(0, index));
    total += 1;
    if (prediction.token === target[index]) {
      matches += 1;
    }
  }
  return { matches, total, accuracy: total === 0 ? 0 : matches / total };
}
