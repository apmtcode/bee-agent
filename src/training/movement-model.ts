import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";

/**
 * Local-movement learning: pluggable model backend.
 *
 * This is the "train a local model to repeat recorded movements and generalize
 * to new but related ones" layer of the movement subsystem (objective 2c/2d).
 *
 * A movement is represented as a discrete {@link MovementToken}, derived from
 * the replay timeline the capture pipeline already produces. A backend learns
 * the sequence structure of those tokens and can, given a context prefix,
 * predict the next movement — enabling both faithful replay (seen prefixes) and
 * generalization (unseen prefixes fall back to shorter, related context).
 *
 * The backend is intentionally an interface: the default {@link MarkovMovementBackend}
 * is a deterministic, dependency-free model that trains and infers in-process so
 * cloud/CI tests are meaningful, while a real on-device small model can be
 * registered under a different id and swapped in without touching call sites.
 */

/** A single discrete movement, e.g. `action:device:tapped submit`. */
export type MovementToken = string;

/** An ordered run of movements the model learns from / rolls out. */
export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

export type MovementTrainingDataset = {
  sequences: MovementSequence[];
};

export type MovementTrainConfig = {
  /** Maximum context length the model conditions on. Default 2. */
  order?: number;
};

/** Serializable trained-model artifact — persistable and portable across runs. */
export type TrainedMovementModel = {
  version: 1;
  backendId: string;
  order: number;
  vocabulary: MovementToken[];
  /**
   * Transition counts keyed by `"<k>|<context>"` where `k` is the context
   * length and `context` is the last `k` tokens joined by {@link TOKEN_SEP}.
   * `k = 0` is the unigram (unconditional) distribution.
   */
  transitions: Record<string, Record<MovementToken, number>>;
  trainedSequenceCount: number;
  trainedTokenCount: number;
};

export type MovementCandidate = {
  token: MovementToken;
  probability: number;
};

export type MovementPrediction = {
  /** Highest-probability next movement, or undefined if nothing is known. */
  token: MovementToken | undefined;
  probability: number;
  /** Context length actually matched (backoff level). `order` = exact match. */
  contextOrderUsed: number;
  candidates: MovementCandidate[];
};

export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementTrainingDataset, config?: MovementTrainConfig): Promise<TrainedMovementModel>;
  predict(model: TrainedMovementModel, context: MovementToken[]): MovementPrediction;
}

/** Separator for context keys — a control char that will not appear in tokens. */
export const TOKEN_SEP = "";
const KEY_SEP = "|";
const DEFAULT_ORDER = 2;

/** Canonicalize a replay timeline event into a single movement token. */
export function tokenizeEvent(event: ReplayTimelineEvent): MovementToken {
  switch (event.kind) {
    case "action":
      return `action:${event.tool}:${normalizeSummary(event.summary)}`;
    case "observation":
      return `obs:${event.source}`;
    case "transcript":
      return `msg:${event.role}`;
  }
}

function normalizeSummary(summary: string): string {
  return summary.trim().toLowerCase().replace(/\s+/g, " ").replaceAll(TOKEN_SEP, " ");
}

/** Build a training sequence from a single replay manifest (movements only). */
export function toMovementSequence(
  manifest: ReplayManifest,
  options: { includeObservations?: boolean; includeTranscript?: boolean } = {},
): MovementSequence {
  const includeObservations = options.includeObservations ?? false;
  const includeTranscript = options.includeTranscript ?? false;
  const tokens = manifest.events
    .filter((event) => {
      if (event.kind === "action") return true;
      if (event.kind === "observation") return includeObservations;
      return includeTranscript;
    })
    .map(tokenizeEvent);
  return { id: manifest.sessionId, tokens };
}

export function buildMovementDataset(
  manifests: ReplayManifest[],
  options?: { includeObservations?: boolean; includeTranscript?: boolean },
): MovementTrainingDataset {
  return { sequences: manifests.map((manifest) => toMovementSequence(manifest, options)) };
}

function contextKey(k: number, suffix: MovementToken[]): string {
  return `${k}${KEY_SEP}${suffix.join(TOKEN_SEP)}`;
}

/**
 * Deterministic, dependency-free backoff n-gram backend.
 *
 * Learns transition counts for every context length `0..order`. Prediction
 * matches the longest available context suffix and backs off to shorter (more
 * general) contexts when the full-order context is unseen — this backoff is the
 * generalization mechanism: a novel prefix that shares a tail with training data
 * still yields a related prediction. Ties break by higher count, then
 * lexicographically, so results are fully reproducible across runs.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov";

  async train(dataset: MovementTrainingDataset, config: MovementTrainConfig = {}): Promise<TrainedMovementModel> {
    const order = Math.max(0, Math.floor(config.order ?? DEFAULT_ORDER));
    const transitions: Record<string, Record<MovementToken, number>> = {};
    const vocabulary = new Set<MovementToken>();
    let trainedTokenCount = 0;

    for (const sequence of dataset.sequences) {
      const tokens = sequence.tokens;
      for (let i = 0; i < tokens.length; i += 1) {
        const next = tokens[i];
        vocabulary.add(next);
        trainedTokenCount += 1;
        for (let k = 0; k <= order; k += 1) {
          if (i - k < 0) break;
          const suffix = tokens.slice(i - k, i);
          const key = contextKey(k, suffix);
          const bucket = (transitions[key] ??= {});
          bucket[next] = (bucket[next] ?? 0) + 1;
        }
      }
    }

    return {
      version: 1,
      backendId: this.id,
      order,
      vocabulary: [...vocabulary].sort(),
      transitions,
      trainedSequenceCount: dataset.sequences.length,
      trainedTokenCount,
    };
  }

  predict(model: TrainedMovementModel, context: MovementToken[]): MovementPrediction {
    const maxK = Math.min(model.order, context.length);
    for (let k = maxK; k >= 0; k -= 1) {
      const suffix = context.slice(context.length - k, context.length);
      const bucket = model.transitions[contextKey(k, suffix)];
      if (!bucket) continue;
      const candidates = rankCandidates(bucket);
      if (candidates.length === 0) continue;
      const best = candidates[0];
      return {
        token: best.token,
        probability: best.probability,
        contextOrderUsed: k,
        candidates,
      };
    }
    return { token: undefined, probability: 0, contextOrderUsed: -1, candidates: [] };
  }
}

function rankCandidates(bucket: Record<MovementToken, number>): MovementCandidate[] {
  const entries = Object.entries(bucket);
  const total = entries.reduce((sum, [, count]) => sum + count, 0);
  if (total === 0) return [];
  return entries
    .map(([token, count]) => ({ token, probability: count / total }))
    .sort((a, b) => {
      if (b.probability !== a.probability) return b.probability - a.probability;
      return a.token < b.token ? -1 : a.token > b.token ? 1 : 0;
    });
}

/**
 * Roll a trained model forward from a seed prefix, greedily appending the
 * predicted next movement. Reproduces recorded movements for seen prefixes and
 * generalizes (via backoff) for novel ones. Deterministic given the same model
 * and seed. A repeat-guard stops runaway loops of a single token, and
 * `minContextOrder` stops the rollout once the model can only back off below
 * that depth (the natural end of a learned workflow).
 */
export function generateMovements(
  backend: MovementModelBackend,
  model: TrainedMovementModel,
  seed: MovementToken[],
  options: { maxLength?: number; maxRepeat?: number; minContextOrder?: number } = {},
): MovementToken[] {
  const maxLength = options.maxLength ?? 32;
  const maxRepeat = options.maxRepeat ?? 4;
  const minContextOrder = options.minContextOrder ?? 0;
  const generated: MovementToken[] = [];
  const context = [...seed];
  let lastToken: MovementToken | undefined;
  let repeatCount = 0;

  while (generated.length < maxLength) {
    const prediction = backend.predict(model, context);
    if (prediction.token === undefined) break;
    // Stop at the natural end of a learned workflow: when the model can only
    // fall back below the requested context depth, there is no real conditional
    // evidence left and further tokens would be unigram noise.
    if (prediction.contextOrderUsed < minContextOrder) break;
    if (prediction.token === lastToken) {
      repeatCount += 1;
      if (repeatCount >= maxRepeat) break;
    } else {
      repeatCount = 0;
    }
    generated.push(prediction.token);
    context.push(prediction.token);
    lastToken = prediction.token;
  }

  return generated;
}

/**
 * Generalization eval harness: next-movement prediction accuracy over held-out
 * sequences (a sequence not seen during training). For each token that has a
 * preceding context, compare the model's top prediction to the actual next
 * movement. Returns overall accuracy plus a breakdown by the backoff level used,
 * which reveals how much the model relies on generalization vs. exact recall.
 */
export function evaluateReplayFidelity(
  backend: MovementModelBackend,
  model: TrainedMovementModel,
  heldOut: MovementSequence[],
): {
  total: number;
  correct: number;
  accuracy: number;
  byContextOrder: Record<number, { total: number; correct: number }>;
} {
  let total = 0;
  let correct = 0;
  const byContextOrder: Record<number, { total: number; correct: number }> = {};

  for (const sequence of heldOut) {
    for (let i = 0; i < sequence.tokens.length; i += 1) {
      const context = sequence.tokens.slice(0, i);
      const prediction = backend.predict(model, context);
      if (prediction.token === undefined) continue;
      total += 1;
      const level = prediction.contextOrderUsed;
      const stats = (byContextOrder[level] ??= { total: 0, correct: 0 });
      stats.total += 1;
      if (prediction.token === sequence.tokens[i]) {
        correct += 1;
        stats.correct += 1;
      }
    }
  }

  return { total, correct, accuracy: total === 0 ? 0 : correct / total, byContextOrder };
}

/**
 * Registry of movement-model backends — the pluggable seam. The deterministic
 * Markov backend is registered as the default; a real on-device model can be
 * registered under its own id and selected without changing call sites.
 */
export class MovementBackendRegistry {
  private readonly backends = new Map<string, MovementModelBackend>();
  private defaultId: string | undefined;

  register(backend: MovementModelBackend, options: { makeDefault?: boolean } = {}): this {
    this.backends.set(backend.id, backend);
    if (options.makeDefault || this.defaultId === undefined) {
      this.defaultId = backend.id;
    }
    return this;
  }

  get(id?: string): MovementModelBackend {
    const resolved = id ?? this.defaultId;
    if (resolved === undefined) {
      throw new Error("no movement backend registered");
    }
    const backend = this.backends.get(resolved);
    if (!backend) {
      throw new Error(`unknown movement backend: ${resolved}`);
    }
    return backend;
  }

  has(id: string): boolean {
    return this.backends.has(id);
  }

  list(): string[] {
    return [...this.backends.keys()].sort();
  }
}

/** A registry pre-populated with the default deterministic backend. */
export function createDefaultMovementBackendRegistry(): MovementBackendRegistry {
  return new MovementBackendRegistry().register(new MarkovMovementBackend(), { makeDefault: true });
}
