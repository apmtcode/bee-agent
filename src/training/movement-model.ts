import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";

/**
 * Local-movement learning: pluggable model backend.
 *
 * Standing objective #2(c)/#2(d): post-train a *local* model on recorded
 * movement trajectories so it can (a) repeat the recorded movements and
 * (b) generalize to new-but-related movements. bee-agent runs in the cloud
 * with no access to the user's real machine, so the concrete on-device
 * runtime (MLX / axolotl — see `runner.ts`) executes only when the user runs
 * bee-agent locally. This module is the *backend seam*: a deterministic,
 * dependency-free model that fully exercises the train -> infer -> generate
 * loop in tests, plus an interface any real on-device backend can implement.
 */

/** A single movement token. Derived from a replay action event (see
 * {@link buildMovementDataset}) or supplied directly by a caller. */
export type MovementToken = string;

/** Sentinel marking the start of a movement sequence (lets the model predict
 * the first move) and the end (lets `generate` know when to stop). */
export const MOVEMENT_START_TOKEN: MovementToken = "␂start";
export const MOVEMENT_END_TOKEN: MovementToken = "␃end";

export type MovementSequence = {
  /** Stable id — usually the source trajectory id. */
  id: string;
  tokens: MovementToken[];
};

export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

export type MovementPrediction = {
  /** Most likely next token. `MOVEMENT_END_TOKEN` means "stop". */
  token: MovementToken;
  /** Empirical probability of `token` within the backed-off context. */
  probability: number;
  /** Length of the context actually used after stupid-backoff. `order` means
   * the full context matched; 0 means it fell all the way back to the
   * unigram distribution — the signal that this is a *generalized* guess. */
  backoffOrder: number;
  /** Other candidates from the same context, most likely first. */
  alternatives: Array<{ token: MovementToken; probability: number }>;
};

export type MovementTrainOptions = {
  /** Max context length (n-gram order). Default 2. */
  order?: number;
};

export type MovementGenerateOptions = {
  /** Stop after this many generated tokens even without an END. Default 64. */
  maxTokens?: number;
};

/**
 * A model backend usable by the local-movement learning subsystem. Backends
 * are pluggable: the deterministic Markov backend below is the CI/cloud-safe
 * default; a real on-device small model implements the same interface and is
 * swapped in via {@link createMovementModelBackend}. `TModel` is the backend's
 * serializable trained-artifact shape.
 */
export interface MovementModelBackend<TModel = unknown> {
  readonly id: string;
  /** Train a fresh model artifact from a dataset. Pure/deterministic for the
   * mock backend so tests never flake. */
  train(dataset: MovementDataset, options?: MovementTrainOptions): TModel;
  /** Predict the single most likely next token given a raw context. */
  predictNext(model: TModel, context: MovementToken[]): MovementPrediction;
  /** Autoregressively roll out a full movement sequence from an optional seed. */
  generate(model: TModel, seed: MovementToken[], options?: MovementGenerateOptions): MovementToken[];
}

/** Serializable trained artifact for {@link DeterministicMarkovBackend}. */
export type MarkovMovementModel = {
  version: 1;
  backendId: "deterministic-markov";
  order: number;
  vocabulary: MovementToken[];
  /** context-key (""-joined tokens; "" = unigram) -> next-token counts. */
  transitions: Record<string, Record<MovementToken, number>>;
  sequenceCount: number;
  tokenCount: number;
};

const CONTEXT_SEPARATOR = "";

/**
 * Deterministic n-gram (Markov) backend with stupid-backoff.
 *
 * Training records, for every position in every sequence, the observed next
 * token under each context length 0..order. Inference tries the longest
 * available context and backs off to shorter ones — which is exactly what
 * lets it *generalize*: an unseen full context (a new-but-related movement)
 * falls back to the shorter context or the unigram prior instead of failing.
 * Fully deterministic (argmax with lexicographic tie-break, no RNG) so the
 * cloud/CI train->infer loop is reproducible.
 */
export class DeterministicMarkovBackend implements MovementModelBackend<MarkovMovementModel> {
  readonly id = "deterministic-markov";

  train(dataset: MovementDataset, options?: MovementTrainOptions): MarkovMovementModel {
    const order = Math.max(1, Math.floor(options?.order ?? 2));
    const transitions: Record<string, Record<MovementToken, number>> = {};
    const vocabulary = new Set<MovementToken>();
    let tokenCount = 0;

    for (const sequence of dataset.sequences) {
      // Pad with START*order so the first real move is predictable, and a
      // trailing END so the model learns when a movement completes.
      const padded = [
        ...Array<MovementToken>(order).fill(MOVEMENT_START_TOKEN),
        ...sequence.tokens,
        MOVEMENT_END_TOKEN,
      ];
      for (const token of sequence.tokens) {
        vocabulary.add(token);
      }
      tokenCount += sequence.tokens.length;

      for (let i = order; i < padded.length; i += 1) {
        const next = padded[i]!;
        for (let k = 0; k <= order; k += 1) {
          const context = padded.slice(i - k, i);
          const key = context.join(CONTEXT_SEPARATOR);
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
      sequenceCount: dataset.sequences.length,
      tokenCount,
    };
  }

  predictNext(model: MarkovMovementModel, context: MovementToken[]): MovementPrediction {
    const trimmed = context.slice(-model.order);
    // Stupid-backoff: longest matching context first, down to unigram ("").
    for (let k = Math.min(trimmed.length, model.order); k >= 0; k -= 1) {
      const key = trimmed.slice(trimmed.length - k).join(CONTEXT_SEPARATOR);
      const bucket = model.transitions[key];
      if (!bucket) {
        continue;
      }
      const ranked = rankBucket(bucket);
      if (ranked.length === 0) {
        continue;
      }
      const total = ranked.reduce((sum, entry) => sum + entry.count, 0);
      const [best, ...rest] = ranked;
      return {
        token: best!.token,
        probability: best!.count / total,
        backoffOrder: k,
        alternatives: rest.map((entry) => ({ token: entry.token, probability: entry.count / total })),
      };
    }
    // Empty model / unknown everything: signal termination rather than throw.
    return { token: MOVEMENT_END_TOKEN, probability: 0, backoffOrder: 0, alternatives: [] };
  }

  generate(model: MarkovMovementModel, seed: MovementToken[], options?: MovementGenerateOptions): MovementToken[] {
    const maxTokens = Math.max(1, Math.floor(options?.maxTokens ?? 64));
    const history = [...Array<MovementToken>(model.order).fill(MOVEMENT_START_TOKEN), ...seed];
    const produced: MovementToken[] = [...seed];
    for (let step = 0; step < maxTokens; step += 1) {
      const prediction = this.predictNext(model, history);
      if (prediction.token === MOVEMENT_END_TOKEN) {
        break;
      }
      produced.push(prediction.token);
      history.push(prediction.token);
    }
    return produced;
  }
}

function rankBucket(bucket: Record<MovementToken, number>): Array<{ token: MovementToken; count: number }> {
  return Object.entries(bucket)
    .map(([token, count]) => ({ token, count }))
    // Deterministic: higher count first, then lexicographic token order.
    .sort((a, b) => (b.count !== a.count ? b.count - a.count : a.token < b.token ? -1 : a.token > b.token ? 1 : 0));
}

export type MovementBackendKind = "deterministic-markov";

/**
 * Backend factory — the pluggable seam. Cloud/CI always resolves the
 * deterministic backend; a real on-device backend registers a new kind here
 * (or callers pass their own {@link MovementModelBackend} instance directly).
 */
export function createMovementModelBackend(kind: MovementBackendKind = "deterministic-markov"): MovementModelBackend {
  switch (kind) {
    case "deterministic-markov":
      return new DeterministicMarkovBackend();
    default: {
      const exhaustive: never = kind;
      throw new Error(`unknown movement model backend: ${String(exhaustive)}`);
    }
  }
}

/**
 * Turn a movement token into its normalized form. Slugs the free-text summary
 * so that "tapped Send button" and "tapped   Send  Button" collapse to the
 * same token — the raw text noise would otherwise fragment the vocabulary and
 * defeat generalization.
 */
export function normalizeMovementSummary(summary: string): string {
  return summary
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim()
    .replace(/\s+/g, "-");
}

/** Build a movement token from a replay action event. */
export function movementTokenFromAction(event: Extract<ReplayTimelineEvent, { kind: "action" }>): MovementToken {
  return `action:${event.tool}:${normalizeMovementSummary(event.summary)}`;
}

/**
 * Build a training dataset from reviewed replay manifests. Each source
 * trajectory becomes one movement sequence, ordered by timestamp, containing
 * only its *action* events (the movements the model must learn to repeat).
 * Trajectories with no actions are skipped.
 */
export function buildMovementDataset(replays: ReplayManifest[]): MovementDataset {
  const bySequence = new Map<string, Array<{ ts: number; token: MovementToken }>>();
  for (const replay of replays) {
    for (const event of replay.events) {
      if (event.kind !== "action") {
        continue;
      }
      const key = `${replay.sessionId}::${event.trajectoryId}`;
      const entries = bySequence.get(key) ?? [];
      entries.push({ ts: event.ts, token: movementTokenFromAction(event) });
      bySequence.set(key, entries);
    }
  }

  const sequences: MovementSequence[] = [];
  for (const [id, entries] of bySequence) {
    if (entries.length === 0) {
      continue;
    }
    entries.sort((a, b) => a.ts - b.ts);
    sequences.push({ id, tokens: entries.map((entry) => entry.token) });
  }
  // Deterministic dataset ordering regardless of Map insertion order.
  sequences.sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  return { version: 1, sequences };
}
