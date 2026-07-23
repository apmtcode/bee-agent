/**
 * Pluggable local movement-model backend.
 *
 * The local-movement learning subsystem records movements/actions (see
 * `src/capture`), exports a reviewed dataset (see `exporter.ts`), and — for a
 * real on-device run — hands that dataset to an external trainer (MLX / axolotl,
 * see `runner.ts`). That external path cannot run in the cloud/CI, so nothing in
 * bee-agent could previously *close the loop*: learn from recorded movements,
 * predict the next movement, and generalize to new-but-related movements.
 *
 * This module provides that loop in-process behind a small, pluggable interface:
 *
 *   - {@link MovementModelBackend} — the training seam. A real on-device small
 *     model can implement this later; the {@link MarkovMovementBackend} here is a
 *     deterministic reference/mock implementation so cloud/CI tests pass without
 *     any OS access, RNG, or heavyweight ML runtime.
 *   - {@link TrainedMovementModel} — the inference seam: `predictNext` (repeat a
 *     recorded movement) and `generate` (roll out a movement sequence). Katz-style
 *     back-off gives it generalization: an unseen prefix falls back to shorter
 *     context and still produces a plausible, related continuation.
 *   - Dataset helpers ({@link buildMovementDataset}) turn replay manifests /
 *     trajectories into token sequences, and {@link evaluateMovementModel} scores
 *     held-out sequences (perplexity + top-1 accuracy) for a generalization eval.
 *
 * Everything here is pure and deterministic: identical dataset in → identical
 * model and predictions out. Ties are broken by frequency then lexical order, so
 * there is no hidden nondeterminism to make tests flaky.
 */

/** A single movement primitive, e.g. `"mouse.move"`, `"key.press"`. */
export type MovementToken = string;

/** An ordered run of movement primitives (one recorded trajectory). */
export type MovementSequence = MovementToken[];

/** Training corpus: one sequence per recorded trajectory, plus its vocabulary. */
export type MovementTrainingDataset = {
  version: 1;
  sequences: MovementSequence[];
  vocabulary: MovementToken[];
};

/** Minimal shape of a replay event we can tokenize (structural, so both
 * `ReplayManifest` and `ExportedReplayManifest` events satisfy it). */
export type TokenizableReplayEvent =
  | { kind: "action"; ts: number; trajectoryId?: string; tool: string; summary?: string }
  | { kind: "observation"; ts: number; trajectoryId?: string; source: string; summary?: string }
  | { kind: string; ts: number; [key: string]: unknown };

/** Minimal shape of a replay manifest we can build a dataset from. */
export type TokenizableReplayManifest = {
  trajectoryIds?: string[];
  events: TokenizableReplayEvent[];
};

export type BuildMovementDatasetOptions = {
  /**
   * Which event kinds contribute tokens. Defaults to `["action"]` — the
   * movement primitives. Include `"observation"` to let observed context steer
   * predictions.
   */
  includeKinds?: Array<"action" | "observation">;
  /**
   * When true, append a slug of the event summary to the tool/source token so
   * distinct gestures under the same tool become distinct tokens. Defaults to
   * false (token == tool/source), which keeps the vocabulary coarse and the
   * back-off generalization strong.
   */
  includeSummary?: boolean;
};

/** Tokenize a single replay event, or `undefined` if it is not a movement. */
export function tokenizeMovementEvent(
  event: TokenizableReplayEvent,
  options: BuildMovementDatasetOptions = {},
): MovementToken | undefined {
  const includeKinds = options.includeKinds ?? ["action"];
  let base: string | undefined;
  if (event.kind === "action" && includeKinds.includes("action")) {
    base = (event as { tool?: string }).tool;
  } else if (event.kind === "observation" && includeKinds.includes("observation")) {
    base = `obs:${(event as { source?: string }).source ?? ""}`;
  }
  if (base === undefined) {
    return undefined;
  }
  const token = base.trim();
  if (token.length === 0) {
    return undefined;
  }
  if (options.includeSummary) {
    const summary = (event as { summary?: string }).summary;
    const slug = slugifySummary(summary);
    if (slug.length > 0) {
      return `${token}#${slug}`;
    }
  }
  return token;
}

function slugifySummary(summary: string | undefined): string {
  if (!summary) {
    return "";
  }
  return summary
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 32);
}

/**
 * Build a training dataset from replay manifests. Events are grouped by
 * `trajectoryId` (falling back to the manifest's single trajectory / the whole
 * manifest), ordered by timestamp, and tokenized into one sequence per
 * trajectory. Empty sequences are dropped.
 */
export function buildMovementDataset(
  manifests: TokenizableReplayManifest[],
  options: BuildMovementDatasetOptions = {},
): MovementTrainingDataset {
  const sequencesByTrajectory = new Map<string, Array<{ ts: number; token: MovementToken }>>();

  for (const manifest of manifests) {
    const fallbackId = manifest.trajectoryIds?.[0] ?? "__manifest__";
    for (const event of manifest.events) {
      const token = tokenizeMovementEvent(event, options);
      if (token === undefined) {
        continue;
      }
      const trajectoryId = (event as { trajectoryId?: string }).trajectoryId ?? fallbackId;
      const bucket = sequencesByTrajectory.get(trajectoryId) ?? [];
      bucket.push({ ts: event.ts, token });
      sequencesByTrajectory.set(trajectoryId, bucket);
    }
  }

  const sequences: MovementSequence[] = [];
  // Deterministic order: sort trajectories by id so identical input → identical dataset.
  for (const trajectoryId of [...sequencesByTrajectory.keys()].sort()) {
    const bucket = sequencesByTrajectory.get(trajectoryId)!;
    // Stable sort by ts (ties keep insertion order, which is manifest order).
    const ordered = bucket
      .map((entry, index) => ({ ...entry, index }))
      .sort((a, b) => (a.ts !== b.ts ? a.ts - b.ts : a.index - b.index))
      .map((entry) => entry.token);
    if (ordered.length > 0) {
      sequences.push(ordered);
    }
  }

  const vocabulary = [...new Set(sequences.flat())].sort();
  return { version: 1, sequences, vocabulary };
}

export type MovementPrediction = {
  /** The predicted next movement primitive. */
  token: MovementToken;
  /** Conditional probability of `token` under the (backed-off) context. */
  probability: number;
  /** The context actually used after back-off (suffix of the input context). */
  contextUsed: MovementSequence;
  /** True when the full-order context had no data and a shorter one was used. */
  backedOff: boolean;
};

export type GenerateMovementOptions = {
  /** Maximum number of tokens to append to the seed. Defaults to 16. */
  maxLength?: number;
  /** Stop generating as soon as one of these tokens is produced. */
  stopTokens?: MovementToken[];
};

export type SerializedMovementModel = {
  backend: string;
  order: number;
  vocabulary: MovementToken[];
  /** context (joined by ) → [token, count][] */
  contexts: Array<{ context: string; counts: Array<[MovementToken, number]> }>;
};

/** A trained, queryable movement model. */
export interface TrainedMovementModel {
  readonly backend: string;
  /** Max context length (n-gram order minus one) the model conditions on. */
  readonly order: number;
  readonly vocabulary: readonly MovementToken[];
  /** Predict the single most likely next movement given recent history. */
  predictNext(context: MovementSequence): MovementPrediction | undefined;
  /** Roll out a movement sequence starting from `seed`. */
  generate(seed: MovementSequence, options?: GenerateMovementOptions): MovementSequence;
  /** Mean log2-probability per token of `sequence` under the model (higher = better fit). */
  score(sequence: MovementSequence): number;
  toJSON(): SerializedMovementModel;
}

export type TrainMovementModelOptions = {
  /** Max context length to condition on (n-gram order − 1). Defaults to 2. */
  order?: number;
};

/** The training seam. Implement this to plug in a real on-device model. */
export interface MovementModelBackend {
  readonly name: string;
  train(
    dataset: MovementTrainingDataset,
    options?: TrainMovementModelOptions,
  ): Promise<TrainedMovementModel>;
}

const CONTEXT_SEPARATOR = "";

function contextKey(tokens: MovementSequence): string {
  return tokens.join(CONTEXT_SEPARATOR);
}

/**
 * Deterministic n-gram model with Katz-style back-off. This is the reference /
 * mock backend: it "post-trains" on the recorded movement sequences by counting
 * n-grams, repeats recorded movements exactly when the full context is known,
 * and generalizes to unseen prefixes by backing off to shorter context.
 */
export class MarkovMovementModel implements TrainedMovementModel {
  readonly backend = "markov";

  constructor(
    readonly order: number,
    /** context key → (token → count) for every context length 0..order. */
    private readonly counts: Map<string, Map<MovementToken, number>>,
    readonly vocabulary: readonly MovementToken[],
  ) {}

  predictNext(context: MovementSequence): MovementPrediction | undefined {
    const maxK = Math.min(this.order, context.length);
    for (let k = maxK; k >= 0; k -= 1) {
      const used = context.slice(context.length - k);
      const distribution = this.counts.get(contextKey(used));
      if (!distribution || distribution.size === 0) {
        continue;
      }
      const best = argmax(distribution);
      if (!best) {
        continue;
      }
      const total = totalCount(distribution);
      return {
        token: best.token,
        probability: total > 0 ? best.count / total : 0,
        contextUsed: used,
        backedOff: k < maxK,
      };
    }
    return undefined;
  }

  generate(seed: MovementSequence, options: GenerateMovementOptions = {}): MovementSequence {
    const maxLength = options.maxLength ?? 16;
    const stopTokens = new Set(options.stopTokens ?? []);
    const output: MovementSequence = [];
    const history = [...seed];
    for (let i = 0; i < maxLength; i += 1) {
      const prediction = this.predictNext(history);
      if (!prediction) {
        break;
      }
      output.push(prediction.token);
      history.push(prediction.token);
      if (stopTokens.has(prediction.token)) {
        break;
      }
    }
    return output;
  }

  score(sequence: MovementSequence): number {
    if (sequence.length === 0) {
      return 0;
    }
    let logProbSum = 0;
    const history: MovementSequence = [];
    for (const token of sequence) {
      const probability = this.probabilityOf(token, history);
      // Floor to avoid -Infinity for tokens outside the vocabulary.
      logProbSum += Math.log2(probability > 0 ? probability : 1e-9);
      history.push(token);
    }
    return logProbSum / sequence.length;
  }

  /** Back-off probability of `token` following `context`. */
  probabilityOf(token: MovementToken, context: MovementSequence): number {
    const maxK = Math.min(this.order, context.length);
    for (let k = maxK; k >= 0; k -= 1) {
      const used = context.slice(context.length - k);
      const distribution = this.counts.get(contextKey(used));
      if (!distribution || distribution.size === 0) {
        continue;
      }
      const total = totalCount(distribution);
      const count = distribution.get(token);
      if (count !== undefined && total > 0) {
        return count / total;
      }
      // Context known but token unseen here — keep backing off to broaden.
    }
    return 0;
  }

  toJSON(): SerializedMovementModel {
    const contexts = [...this.counts.entries()]
      .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
      .map(([context, distribution]) => ({
        context,
        counts: [...distribution.entries()].sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)),
      }));
    return { backend: this.backend, order: this.order, vocabulary: [...this.vocabulary], contexts };
  }

  static fromJSON(serialized: SerializedMovementModel): MarkovMovementModel {
    const counts = new Map<string, Map<MovementToken, number>>();
    for (const { context, counts: entries } of serialized.contexts) {
      counts.set(context, new Map(entries));
    }
    return new MarkovMovementModel(serialized.order, counts, serialized.vocabulary);
  }
}

/** Deterministic n-gram backend — the pluggable reference/mock implementation. */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly name = "markov";

  async train(
    dataset: MovementTrainingDataset,
    options: TrainMovementModelOptions = {},
  ): Promise<TrainedMovementModel> {
    const order = Math.max(0, options.order ?? 2);
    const counts = new Map<string, Map<MovementToken, number>>();

    const bump = (context: MovementSequence, token: MovementToken): void => {
      const key = contextKey(context);
      const distribution = counts.get(key) ?? new Map<MovementToken, number>();
      distribution.set(token, (distribution.get(token) ?? 0) + 1);
      counts.set(key, distribution);
    };

    for (const sequence of dataset.sequences) {
      for (let position = 0; position < sequence.length; position += 1) {
        const token = sequence[position]!;
        const maxK = Math.min(order, position);
        for (let k = 0; k <= maxK; k += 1) {
          bump(sequence.slice(position - k, position), token);
        }
      }
    }

    return new MarkovMovementModel(order, counts, [...dataset.vocabulary]);
  }
}

function argmax(distribution: Map<MovementToken, number>): { token: MovementToken; count: number } | undefined {
  let best: { token: MovementToken; count: number } | undefined;
  for (const [token, count] of distribution) {
    if (
      best === undefined ||
      count > best.count ||
      // Deterministic tie-break: lexically smaller token wins.
      (count === best.count && token < best.token)
    ) {
      best = { token, count };
    }
  }
  return best;
}

function totalCount(distribution: Map<MovementToken, number>): number {
  let total = 0;
  for (const count of distribution.values()) {
    total += count;
  }
  return total;
}

export type MovementModelEvaluation = {
  /** Number of held-out (context, next-token) prediction points scored. */
  predictionCount: number;
  /** Fraction of points where the model's top prediction matched the truth. */
  top1Accuracy: number;
  /** 2^(−mean log2-prob per token); lower is better, 1 is perfect. */
  perplexity: number;
  /** How often the top prediction required backing off to shorter context. */
  backoffRate: number;
};

/**
 * Generalization eval: for each held-out sequence, walk it token by token and
 * measure how well the trained model predicts the next movement it never saw in
 * this exact context. Deterministic; no OS or RNG involved.
 */
export function evaluateMovementModel(
  model: MarkovMovementModel,
  heldOut: MovementSequence[],
): MovementModelEvaluation {
  let predictionCount = 0;
  let correct = 0;
  let backoffs = 0;
  let logProbSum = 0;

  for (const sequence of heldOut) {
    const history: MovementSequence = [];
    for (const actual of sequence) {
      const prediction = model.predictNext(history);
      if (prediction) {
        predictionCount += 1;
        if (prediction.token === actual) {
          correct += 1;
        }
        if (prediction.backedOff) {
          backoffs += 1;
        }
        const probability = model.probabilityOf(actual, history);
        logProbSum += Math.log2(probability > 0 ? probability : 1e-9);
      }
      history.push(actual);
    }
  }

  const meanLogProb = predictionCount > 0 ? logProbSum / predictionCount : 0;
  return {
    predictionCount,
    top1Accuracy: predictionCount > 0 ? correct / predictionCount : 0,
    perplexity: predictionCount > 0 ? 2 ** -meanLogProb : 1,
    backoffRate: predictionCount > 0 ? backoffs / predictionCount : 0,
  };
}
