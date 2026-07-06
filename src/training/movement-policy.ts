import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

/**
 * Local-movement learning: pluggable policy backend.
 *
 * The capture/replay/export pipeline turns recorded movements into reviewed
 * datasets, and `runner.ts` emits an on-device (Apple Silicon MLX/axolotl)
 * launch script for heavy fine-tuning. That path only *executes* on the user's
 * real machine, so it cannot be exercised in the cloud.
 *
 * This module adds the missing in-process seam: a `MovementPolicyBackend` that
 * trains a small model on a movement dataset and produces a `TrainedMovementPolicy`
 * capable of (c) repeating recorded movements and (d) generalizing to new but
 * related movements via context backoff. The backend is pluggable — the
 * deterministic `MarkovMovementBackend` below is the default/mock used for
 * cloud/CI validation; a real on-device backend can register under a new id and
 * satisfy the same interface.
 */

export const MOVEMENT_START_TOKEN = "<start>";
export const MOVEMENT_END_TOKEN = "<end>";

/** A single movement token (one recorded action, tokenized). */
export type MovementToken = string;

/** Maps a recorded action to a policy token. Default: the tool name. */
export type MovementTokenizer = (action: TrajectoryAction) => MovementToken;

export const defaultMovementTokenizer: MovementTokenizer = (action) => action.tool;

/** One trajectory rendered as an ordered token sequence (no start/end markers). */
export type MovementSequence = MovementToken[];

/** A single (context -> next) supervised example. */
export type MovementSample = {
  context: MovementToken[];
  next: MovementToken;
};

export type MovementDataset = {
  version: 1;
  /** Sorted, de-duplicated set of observed tokens (excludes start/end markers). */
  vocabulary: MovementToken[];
  /** One token sequence per source trajectory, in capture order. */
  sequences: MovementSequence[];
  /** Sliding-window supervised samples derived from `sequences`. */
  samples: MovementSample[];
};

export type BuildMovementDatasetOptions = {
  tokenizer?: MovementTokenizer;
  /** Max context length used when emitting supervised samples. Default 3. */
  order?: number;
};

/**
 * Turn reviewed/recorded trajectories into a movement dataset. Actions are read
 * in ascending timestamp order so replay and training see the same ordering.
 */
export function buildMovementDataset(
  trajectories: TrajectorySpan[],
  options: BuildMovementDatasetOptions = {},
): MovementDataset {
  const tokenizer = options.tokenizer ?? defaultMovementTokenizer;
  const order = Math.max(1, options.order ?? 3);
  const vocabulary = new Set<MovementToken>();
  const sequences: MovementSequence[] = [];
  const samples: MovementSample[] = [];

  for (const trajectory of trajectories) {
    const sequence = [...trajectory.actions]
      .sort((a, b) => a.ts - b.ts)
      .map((action) => tokenizer(action));
    if (sequence.length === 0) {
      continue;
    }
    for (const token of sequence) {
      vocabulary.add(token);
    }
    sequences.push(sequence);

    // Emit samples with start/end markers so the model learns how movements
    // begin and terminate, not just their interior transitions.
    const padded = [MOVEMENT_START_TOKEN, ...sequence, MOVEMENT_END_TOKEN];
    for (let i = 1; i < padded.length; i += 1) {
      const contextStart = Math.max(0, i - order);
      samples.push({ context: padded.slice(contextStart, i), next: padded[i]! });
    }
  }

  return {
    version: 1,
    vocabulary: [...vocabulary].sort(),
    sequences,
    samples,
  };
}

export type MovementPrediction = {
  token: MovementToken;
  confidence: number;
  /** Full next-token distribution at the backed-off context, descending. */
  distribution: Array<{ token: MovementToken; probability: number }>;
  /** Order of the context that actually produced the prediction after backoff. */
  backoffOrder: number;
};

/** A trained, serializable movement policy ready for inference. */
export interface TrainedMovementPolicy {
  readonly backendId: string;
  readonly order: number;
  /** Predict the next movement token given a (possibly short/unseen) context. */
  predictNext(context: MovementToken[]): MovementPrediction;
  /** Roll out a full movement sequence from a seed until <end> or maxSteps. */
  generate(seed: MovementToken[], maxSteps?: number): MovementSequence;
  /** Serialize to a plain JSON-safe object for persistence. */
  serialize(): MovementPolicyModel;
}

/** Pluggable backend that trains policies from movement datasets. */
export interface MovementPolicyBackend {
  readonly id: string;
  train(dataset: MovementDataset): Promise<TrainedMovementPolicy>;
}

/** JSON-safe persisted form of a trained policy. */
export type MovementPolicyModel = {
  version: 1;
  backendId: string;
  order: number;
  /** contextKey -> { token -> count }. Empty-context key is the START distribution. */
  transitions: Record<string, Record<MovementToken, number>>;
};

const CONTEXT_SEPARATOR = "";

function contextKey(context: MovementToken[]): string {
  return context.join(CONTEXT_SEPARATOR);
}

/**
 * Deterministic n-gram Markov backend. Learns variable-order transition counts
 * (orders 0..N) so inference can back off from the longest known context to
 * shorter ones — this is what lets it generalize to new-but-related movements
 * whose exact prefix was never recorded. Fully deterministic (lexicographic
 * tie-breaks), so cloud/CI tests are stable without any native dependency.
 */
export class MarkovMovementBackend implements MovementPolicyBackend {
  readonly id = "markov";

  constructor(private readonly order = 3) {}

  async train(dataset: MovementDataset): Promise<TrainedMovementPolicy> {
    const order = Math.max(1, this.order);
    const transitions = new Map<string, Map<MovementToken, number>>();

    const record = (context: MovementToken[], next: MovementToken): void => {
      const key = contextKey(context);
      let row = transitions.get(key);
      if (!row) {
        row = new Map<MovementToken, number>();
        transitions.set(key, row);
      }
      row.set(next, (row.get(next) ?? 0) + 1);
    };

    for (const sequence of dataset.sequences) {
      const padded = [MOVEMENT_START_TOKEN, ...sequence, MOVEMENT_END_TOKEN];
      for (let i = 1; i < padded.length; i += 1) {
        const next = padded[i]!;
        // Record this transition at every context order 0..order for backoff.
        for (let k = 0; k <= order; k += 1) {
          const start = Math.max(0, i - k);
          record(padded.slice(start, i), next);
        }
      }
    }

    return new MarkovMovementPolicy(this.id, order, transitions);
  }
}

class MarkovMovementPolicy implements TrainedMovementPolicy {
  constructor(
    readonly backendId: string,
    readonly order: number,
    private readonly transitions: Map<string, Map<MovementToken, number>>,
  ) {}

  predictNext(context: MovementToken[]): MovementPrediction {
    const maxOrder = Math.min(this.order, context.length);
    for (let k = maxOrder; k >= 0; k -= 1) {
      const key = contextKey(context.slice(context.length - k));
      const row = this.transitions.get(key);
      if (row && row.size > 0) {
        return { ...distributionFrom(row), backoffOrder: k };
      }
    }
    return {
      token: MOVEMENT_END_TOKEN,
      confidence: 0,
      distribution: [{ token: MOVEMENT_END_TOKEN, probability: 1 }],
      backoffOrder: -1,
    };
  }

  generate(seed: MovementToken[], maxSteps = 64): MovementSequence {
    const generated: MovementSequence = [];
    let context: MovementToken[] = [MOVEMENT_START_TOKEN, ...seed];
    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predictNext(context);
      if (prediction.token === MOVEMENT_END_TOKEN || prediction.backoffOrder < 0) {
        break;
      }
      generated.push(prediction.token);
      context = [...context, prediction.token].slice(-(this.order + 1));
    }
    return [...seed, ...generated];
  }

  serialize(): MovementPolicyModel {
    const transitions: Record<string, Record<MovementToken, number>> = {};
    for (const [key, row] of this.transitions) {
      const counts: Record<MovementToken, number> = {};
      for (const [token, count] of row) {
        counts[token] = count;
      }
      transitions[key] = counts;
    }
    return { version: 1, backendId: this.backendId, order: this.order, transitions };
  }
}

/** Reconstruct a trained policy from its serialized form for offline inference. */
export function deserializeMovementPolicy(model: MovementPolicyModel): TrainedMovementPolicy {
  const transitions = new Map<string, Map<MovementToken, number>>();
  for (const [key, counts] of Object.entries(model.transitions)) {
    const row = new Map<MovementToken, number>();
    for (const [token, count] of Object.entries(counts)) {
      row.set(token, count);
    }
    transitions.set(key, row);
  }
  return new MarkovMovementPolicy(model.backendId, model.order, transitions);
}

function distributionFrom(row: Map<MovementToken, number>): {
  token: MovementToken;
  confidence: number;
  distribution: Array<{ token: MovementToken; probability: number }>;
} {
  const total = [...row.values()].reduce((sum, count) => sum + count, 0);
  const distribution = [...row.entries()]
    .map(([token, count]) => ({ token, probability: count / total }))
    // Descending probability, then lexicographic token for deterministic ties.
    .sort((a, b) => (b.probability - a.probability) || (a.token < b.token ? -1 : a.token > b.token ? 1 : 0));
  const best = distribution[0]!;
  return { token: best.token, confidence: best.probability, distribution };
}

/**
 * Registry seam. The default/mock backend is deterministic and dependency-free.
 * A real on-device backend (e.g. wrapping the MLX runner) can be registered by
 * id and will satisfy the same `MovementPolicyBackend` contract.
 */
const BACKEND_FACTORIES = new Map<string, () => MovementPolicyBackend>([
  ["markov", () => new MarkovMovementBackend()],
  ["mock", () => new MarkovMovementBackend()],
]);

export function registerMovementPolicyBackend(id: string, factory: () => MovementPolicyBackend): void {
  BACKEND_FACTORIES.set(id, factory);
}

export function createMovementPolicyBackend(id = "markov"): MovementPolicyBackend {
  const factory = BACKEND_FACTORIES.get(id);
  if (!factory) {
    throw new Error(
      `Unknown movement policy backend "${id}". Registered: ${[...BACKEND_FACTORIES.keys()].join(", ")}`,
    );
  }
  return factory();
}
