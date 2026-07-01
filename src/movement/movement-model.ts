import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Local-movement learning model (standing objective #2, parts c & d).
 *
 * This module defines a *pluggable* backend interface for post-training a small
 * local model on recorded movement trajectories so it can (c) repeat the
 * recorded movements and (d) generalize to new-but-related movements. The
 * default backend is a deterministic n-gram model with stupid-backoff — it needs
 * no native deps, trains in-process, and produces reproducible output, so it can
 * run in the cloud/CI where no real on-device model or OS input is available.
 *
 * A real on-device backend (e.g. an MLX/llama.cpp policy) can implement the same
 * {@link MovementModelBackend} interface and be swapped in via the registry
 * without touching call sites.
 */

/** A discrete movement token — a stable label for one recorded action. */
export type MovementToken = string;

/** An ordered sequence of movement tokens (one recorded/played trajectory). */
export type MovementSequence = MovementToken[];

/** Sentinel appended by the model to mark end-of-sequence during generation. */
export const MOVEMENT_END: MovementToken = "\u241Eend\u241E";

/** Internal separator used to key a context array as a single map key. */
const CONTEXT_SEP = "\u241F";

/** Normalize a (tool, summary) pair into a stable movement token. */
export function tokenizeAction(tool: string, summary: string): MovementToken {
  const normalizedTool = tool.trim().toLowerCase();
  const normalizedSummary = summary.trim().toLowerCase().replace(/\s+/g, " ");
  return `${normalizedTool}::${normalizedSummary}`;
}

/** Convert a recorded trajectory span's actions into a movement token sequence. */
export function tokenizeTrajectory(span: TrajectorySpan): MovementSequence {
  return [...span.actions]
    .sort((a, b) => a.ts - b.ts)
    .map((action) => tokenizeAction(action.tool, action.summary));
}

export type MovementPrediction = {
  token: MovementToken;
  /** Probability of this token within the ranked candidate set (0..1). */
  probability: number;
  /** Context length actually used to produce this prediction (backoff order). */
  order: number;
};

export type PredictMovementOptions = {
  /** Cap the number of ranked candidates returned (defaults to all). */
  topK?: number;
};

export type GenerateMovementOptions = {
  /** Hard cap on generated sequence length, including the seed (default 64). */
  maxSteps?: number;
};

export type TrainMovementOptions = {
  /** Max n-gram order (longest context length used). Defaults to 3. */
  order?: number;
};

export type SerializedMovementModel = {
  version: 1;
  backend: string;
  order: number;
  contexts: Array<{
    context: MovementToken[];
    next: Array<{ token: MovementToken; count: number }>;
  }>;
};

export interface MovementModel {
  readonly backend: string;
  readonly order: number;
  /** Rank the likely next tokens for a context prefix (most likely first). */
  predict(context: MovementSequence, options?: PredictMovementOptions): MovementPrediction[];
  /** Greedily roll out a full sequence from a seed prefix until END/maxSteps. */
  generate(seed: MovementSequence, options?: GenerateMovementOptions): MovementSequence;
  /** All tokens observed during training (excludes the END sentinel). */
  vocabulary(): MovementToken[];
  serialize(): SerializedMovementModel;
}

export interface MovementModelBackend {
  readonly name: string;
  train(sequences: MovementSequence[], options?: TrainMovementOptions): MovementModel;
  load(serialized: SerializedMovementModel): MovementModel;
}

/**
 * Deterministic n-gram model with stupid-backoff.
 *
 * Training records, for every context suffix of length 0..order-1 observed in
 * the data, the distribution of the following token (with {@link MOVEMENT_END}
 * appended to each sequence). Prediction tries the longest matching context and
 * backs off to shorter contexts until it finds evidence — which is exactly what
 * lets it both *repeat* a memorized trajectory and *generalize* a shared
 * sub-sequence onto a novel prefix.
 */
export class NGramMovementModel implements MovementModel {
  readonly backend = "ngram-backoff";

  constructor(
    readonly order: number,
    private readonly counts: Map<string, Map<MovementToken, number>>,
    private readonly vocab: Set<MovementToken>,
  ) {}

  static train(sequences: MovementSequence[], options: TrainMovementOptions = {}): NGramMovementModel {
    const order = Math.max(1, Math.floor(options.order ?? 3));
    const counts = new Map<string, Map<MovementToken, number>>();
    const vocab = new Set<MovementToken>();

    for (const rawSequence of sequences) {
      const sequence = [...rawSequence, MOVEMENT_END];
      for (let i = 0; i < sequence.length; i += 1) {
        const token = sequence[i]!;
        if (token !== MOVEMENT_END) {
          vocab.add(token);
        }
        // Record this token as the continuation of every context suffix that
        // ends just before it, for context lengths 0..order-1.
        const maxContext = Math.min(order - 1, i);
        for (let k = 0; k <= maxContext; k += 1) {
          const context = sequence.slice(i - k, i);
          const key = contextKey(context);
          let bucket = counts.get(key);
          if (!bucket) {
            bucket = new Map<MovementToken, number>();
            counts.set(key, bucket);
          }
          bucket.set(token, (bucket.get(token) ?? 0) + 1);
        }
      }
    }

    return new NGramMovementModel(order, counts, vocab);
  }

  predict(context: MovementSequence, options: PredictMovementOptions = {}): MovementPrediction[] {
    const maxContext = Math.min(this.order - 1, context.length);
    for (let k = maxContext; k >= 0; k -= 1) {
      const suffix = context.slice(context.length - k);
      const bucket = this.counts.get(contextKey(suffix));
      if (!bucket || bucket.size === 0) {
        continue;
      }
      const total = [...bucket.values()].reduce((sum, count) => sum + count, 0);
      const ranked = [...bucket.entries()]
        .map(([token, count]) => ({ token, probability: count / total, order: k }))
        .sort((a, b) => (b.probability - a.probability) || compareTokens(a.token, b.token));
      return options.topK !== undefined ? ranked.slice(0, Math.max(0, options.topK)) : ranked;
    }
    return [];
  }

  generate(seed: MovementSequence, options: GenerateMovementOptions = {}): MovementSequence {
    const maxSteps = Math.max(seed.length, Math.floor(options.maxSteps ?? 64));
    const output = [...seed];
    while (output.length < maxSteps) {
      const [best] = this.predict(output, { topK: 1 });
      if (!best || best.token === MOVEMENT_END) {
        break;
      }
      output.push(best.token);
    }
    return output;
  }

  vocabulary(): MovementToken[] {
    return [...this.vocab].sort(compareTokens);
  }

  serialize(): SerializedMovementModel {
    const contexts = [...this.counts.entries()].map(([key, bucket]) => ({
      context: key === "" ? [] : key.split(CONTEXT_SEP),
      next: [...bucket.entries()]
        .map(([token, count]) => ({ token, count }))
        .sort((a, b) => compareTokens(a.token, b.token)),
    }));
    contexts.sort((a, b) => compareTokens(contextKey(a.context), contextKey(b.context)));
    return { version: 1, backend: this.backend, order: this.order, contexts };
  }

  static load(serialized: SerializedMovementModel): NGramMovementModel {
    const counts = new Map<string, Map<MovementToken, number>>();
    const vocab = new Set<MovementToken>();
    for (const entry of serialized.contexts) {
      const bucket = new Map<MovementToken, number>();
      for (const { token, count } of entry.next) {
        bucket.set(token, count);
        if (token !== MOVEMENT_END) {
          vocab.add(token);
        }
      }
      counts.set(contextKey(entry.context), bucket);
    }
    return new NGramMovementModel(serialized.order, counts, vocab);
  }
}

export const ngramMovementBackend: MovementModelBackend = {
  name: "ngram-backoff",
  train: (sequences, options) => NGramMovementModel.train(sequences, options),
  load: (serialized) => NGramMovementModel.load(serialized),
};

const backendRegistry = new Map<string, MovementModelBackend>([[ngramMovementBackend.name, ngramMovementBackend]]);

/** Register a movement-model backend (e.g. a real on-device policy). */
export function registerMovementBackend(backend: MovementModelBackend): void {
  backendRegistry.set(backend.name, backend);
}

/** Look up a registered backend by name. Throws if it is not registered. */
export function getMovementBackend(name: string): MovementModelBackend {
  const backend = backendRegistry.get(name);
  if (!backend) {
    throw new Error(`unknown movement-model backend: ${name} (have: ${[...backendRegistry.keys()].join(", ")})`);
  }
  return backend;
}

/** Names of all registered backends. */
export function listMovementBackends(): string[] {
  return [...backendRegistry.keys()].sort(compareTokens);
}

function contextKey(context: MovementToken[]): string {
  return context.join(CONTEXT_SEP);
}

function compareTokens(a: MovementToken, b: MovementToken): number {
  return a < b ? -1 : a > b ? 1 : 0;
}
