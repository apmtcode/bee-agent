import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * In-process movement-model backend for the local-movement learning subsystem.
 *
 * This closes objective 2(c)/(d): post-train a local model on a recorded
 * movement dataset so it can (a) repeat the recorded movements and (b)
 * generalize to new-but-related movements. The heavy on-device runtimes
 * (mlx/axolotl) are emitted as launch plans by ../training/runner.js; this
 * module provides a fully deterministic, dependency-free backend that runs in
 * the cloud/CI and behind a pluggable interface, so a real small on-device
 * model can be dropped in later without changing call sites.
 */

/** A single recorded movement, tokenised to a canonical, comparable form. */
export type MovementStep = {
  /** Canonical token, e.g. "device.tap:submit" -- the unit the model learns over. */
  token: string;
  /** Original human-readable summary, retained for replay/debugging. */
  summary?: string;
  /** Source timestamp, if known. */
  ts?: number;
};

/** An ordered run of movements from one trajectory/session. */
export type MovementSequence = {
  id: string;
  steps: MovementStep[];
};

/** The replayable dataset a backend trains on. */
export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

export type MovementTrainingOptions = {
  /** Maximum context length (n-gram order minus 1). Default 2. */
  order?: number;
  /** Add-k smoothing mass for generalisation to unseen in-context tokens. Default 0.01. */
  smoothing?: number;
};

export type MovementModelPrediction = {
  token: string;
  probability: number;
  /** Context length actually used after backoff (observability). */
  order: number;
};

/** Serialisable form so a trained model can be persisted and reloaded for inference. */
export type MovementModelSnapshot = {
  version: 1;
  backend: string;
  order: number;
  smoothing: number;
  vocabulary: string[];
  /** contextKey -> token -> count. The empty-string key holds the unigram distribution. */
  counts: Record<string, Record<string, number>>;
};

export interface TrainedMovementModel {
  readonly backend: string;
  readonly vocabulary: readonly string[];
  /** Most-likely next movement given a context of recent tokens, or undefined for an empty model. */
  predictNext(context: readonly string[]): MovementModelPrediction | undefined;
  /** All candidate next movements, most-likely first. */
  rankNext(context: readonly string[]): MovementModelPrediction[];
  /** Greedily extend seed up to maxSteps movements (stops at a learned end-of-sequence). */
  generate(seed: readonly string[], maxSteps: number): string[];
  toJSON(): MovementModelSnapshot;
}

export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, options?: MovementTrainingOptions): TrainedMovementModel;
  load(snapshot: MovementModelSnapshot): TrainedMovementModel;
}

/** Sentinel appended during training so the model learns where sequences end. */
const END_TOKEN = "__movement_end__";
/** Separator used to build context keys; a newline can never appear inside a token. */
const CONTEXT_SEPARATOR = "\n";

/**
 * Deterministic n-gram backend with stupid-backoff + add-k smoothing.
 *
 * - Repeat: greedy argmax over the highest-order matching context reproduces a
 *   recorded run when seeded with its prefix.
 * - Generalise: a novel high-order context that was never recorded backs off to
 *   the longest recorded suffix, so related-but-unseen prefixes still yield a
 *   plausible continuation. Add-k keeps in-vocabulary tokens non-zero.
 *
 * No randomness or wall-clock use, so training and inference are reproducible.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly name = "markov-ngram";

  train(dataset: MovementDataset, options: MovementTrainingOptions = {}): TrainedMovementModel {
    const order = Math.max(1, Math.floor(options.order ?? 2));
    const smoothing = options.smoothing ?? 0.01;
    const counts = new Map<string, Map<string, number>>();
    const vocabulary = new Set<string>();

    for (const sequence of dataset.sequences) {
      const tokens = [...sequence.steps.map((step) => step.token), END_TOKEN];
      for (const token of tokens) {
        if (token !== END_TOKEN) {
          vocabulary.add(token);
        }
      }
      for (let i = 0; i < tokens.length; i += 1) {
        const target = tokens[i]!;
        for (let k = 0; k <= order; k += 1) {
          if (i - k < 0) {
            break;
          }
          const contextKey = tokens.slice(i - k, i).join(CONTEXT_SEPARATOR);
          const bucket = counts.get(contextKey) ?? new Map<string, number>();
          bucket.set(target, (bucket.get(target) ?? 0) + 1);
          counts.set(contextKey, bucket);
        }
      }
    }

    return new MarkovMovementModel(this.name, order, smoothing, vocabulary, counts);
  }

  load(snapshot: MovementModelSnapshot): TrainedMovementModel {
    const counts = new Map<string, Map<string, number>>();
    for (const [contextKey, bucket] of Object.entries(snapshot.counts)) {
      counts.set(contextKey, new Map(Object.entries(bucket)));
    }
    return new MarkovMovementModel(
      snapshot.backend,
      snapshot.order,
      snapshot.smoothing,
      new Set(snapshot.vocabulary),
      counts,
    );
  }
}

class MarkovMovementModel implements TrainedMovementModel {
  readonly vocabulary: readonly string[];

  constructor(
    readonly backend: string,
    private readonly order: number,
    private readonly smoothing: number,
    vocabulary: Set<string>,
    private readonly counts: Map<string, Map<string, number>>,
  ) {
    this.vocabulary = [...vocabulary].sort();
  }

  predictNext(context: readonly string[]): MovementModelPrediction | undefined {
    return this.rankNext(context)[0];
  }

  rankNext(context: readonly string[]): MovementModelPrediction[] {
    const bucketMatch = this.resolveBucket(context);
    if (!bucketMatch) {
      return [];
    }
    const { bucket, order } = bucketMatch;
    const vocabForSmoothing = this.vocabulary.length + 1; // +1 for END
    let total = 0;
    for (const value of bucket.values()) {
      total += value;
    }
    const denominator = total + this.smoothing * vocabForSmoothing;
    const predictions: MovementModelPrediction[] = [];
    for (const [token, count] of bucket) {
      if (token === END_TOKEN) {
        continue;
      }
      predictions.push({
        token,
        probability: (count + this.smoothing) / denominator,
        order,
      });
    }
    // Deterministic ordering: probability desc, then token asc for stable tie-breaks.
    predictions.sort(
      (a, b) => b.probability - a.probability || (a.token < b.token ? -1 : a.token > b.token ? 1 : 0),
    );
    return predictions;
  }

  generate(seed: readonly string[], maxSteps: number): string[] {
    const generated: string[] = [];
    let context = [...seed];
    for (let step = 0; step < maxSteps; step += 1) {
      const bucketMatch = this.resolveBucket(context);
      if (!bucketMatch) {
        break;
      }
      const best = this.argmaxToken(bucketMatch.bucket);
      if (best === undefined || best === END_TOKEN) {
        break;
      }
      generated.push(best);
      context = [...context, best];
    }
    return generated;
  }

  toJSON(): MovementModelSnapshot {
    const counts: Record<string, Record<string, number>> = {};
    for (const [contextKey, bucket] of this.counts) {
      counts[contextKey] = Object.fromEntries(bucket);
    }
    return {
      version: 1,
      backend: this.backend,
      order: this.order,
      smoothing: this.smoothing,
      vocabulary: [...this.vocabulary],
      counts,
    };
  }

  /** Stupid-backoff: longest matching context wins; fall back to shorter suffixes. */
  private resolveBucket(context: readonly string[]): { bucket: Map<string, number>; order: number } | undefined {
    const maxK = Math.min(this.order, context.length);
    for (let k = maxK; k >= 0; k -= 1) {
      const contextKey = context.slice(context.length - k, context.length).join(CONTEXT_SEPARATOR);
      const bucket = this.counts.get(contextKey);
      if (bucket && bucket.size > 0) {
        return { bucket, order: k };
      }
    }
    return undefined;
  }

  /** Argmax over a bucket with deterministic (lexicographic) tie-breaking. */
  private argmaxToken(bucket: Map<string, number>): string | undefined {
    let bestToken: string | undefined;
    let bestCount = -1;
    for (const [token, count] of bucket) {
      if (count > bestCount || (count === bestCount && bestToken !== undefined && token < bestToken)) {
        bestToken = token;
        bestCount = count;
      }
    }
    return bestToken;
  }
}

/** Canonical token for a recorded movement action ("tool.gesture:target"). */
export function movementTokenFor(params: {
  tool: string;
  summary: string;
  metadata?: Record<string, unknown>;
}): string {
  const gesture = typeof params.metadata?.gesture === "string" ? params.metadata.gesture : undefined;
  const target =
    typeof params.metadata?.target === "string"
      ? params.metadata.target
      : typeof params.metadata?.direction === "string"
        ? params.metadata.direction
        : undefined;
  const head = gesture ? `${params.tool}.${gesture}` : params.tool;
  const tail = target ? normalizeTokenPart(target) : normalizeTokenPart(params.summary);
  return tail ? `${head}:${tail}` : head;
}

function normalizeTokenPart(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 48);
}

/** Build a movement dataset from reviewed trajectory spans (uses their actions). */
export function buildMovementDatasetFromTrajectories(trajectories: readonly TrajectorySpan[]): MovementDataset {
  const sequences: MovementSequence[] = trajectories.map((trajectory) => ({
    id: trajectory.id,
    steps: [...trajectory.actions]
      .sort((a, b) => a.ts - b.ts)
      .map((action) => ({
        token: movementTokenFor({ tool: action.tool, summary: action.summary, metadata: action.metadata }),
        summary: action.summary,
        ts: action.ts,
      })),
  }));
  return { version: 1, sequences: sequences.filter((sequence) => sequence.steps.length > 0) };
}

/** Build a movement dataset from replay manifests (one sequence per manifest, action events only). */
export function buildMovementDatasetFromReplays(replays: readonly ReplayManifest[]): MovementDataset {
  const sequences: MovementSequence[] = replays.map((replay) => ({
    id: replay.sessionId,
    steps: replay.events
      .filter((event): event is Extract<ReplayTimelineEvent, { kind: "action" }> => event.kind === "action")
      .map((event) => ({
        token: movementTokenFor({ tool: event.tool, summary: event.summary }),
        summary: event.summary,
        ts: event.ts,
      })),
  }));
  return { version: 1, sequences: sequences.filter((sequence) => sequence.steps.length > 0) };
}

export type MovementEvaluationResult = {
  /** Held-out next-movement prediction accuracy in [0,1]. */
  accuracy: number;
  predictions: number;
  correct: number;
  /** How often the true next token appeared anywhere in the ranked candidates. */
  coverage: number;
};

/**
 * Measure how well a trained model reproduces / generalises to held-out sequences
 * by teacher-forced next-token prediction. Supports the generalisation eval harness.
 */
export function evaluateMovementModel(
  model: TrainedMovementModel,
  heldOut: readonly MovementSequence[],
): MovementEvaluationResult {
  let predictions = 0;
  let correct = 0;
  let covered = 0;
  for (const sequence of heldOut) {
    const tokens = sequence.steps.map((step) => step.token);
    for (let i = 1; i < tokens.length; i += 1) {
      const context = tokens.slice(0, i);
      const truth = tokens[i]!;
      const ranked = model.rankNext(context);
      predictions += 1;
      if (ranked.length === 0) {
        continue;
      }
      if (ranked[0]!.token === truth) {
        correct += 1;
      }
      if (ranked.some((prediction) => prediction.token === truth)) {
        covered += 1;
      }
    }
  }
  return {
    accuracy: predictions === 0 ? 0 : correct / predictions,
    predictions,
    correct,
    coverage: predictions === 0 ? 0 : covered / predictions,
  };
}
