import type { TrainingMode } from "./export-manifest.js";

/**
 * Local-movement learning: a pluggable backend that post-trains a small,
 * on-device movement model from reviewed movement trajectories, plus an
 * inference engine that (a) repeats the recorded movements and (b) generalizes
 * to new-but-related movements via backed-off n-gram decoding.
 *
 * This is the deterministic, cloud/CI-safe default backend. The
 * {@link MovementModelBackend} interface is the seam for swapping in a real
 * on-device learner (e.g. an MLX/axolotl small model) without changing callers.
 */

/** A single recorded movement — the atomic unit the model learns to produce. */
export type MovementStep = {
  /** Canonical movement/tool token, e.g. "mouse.click", "key.type". */
  tool: string;
  /** Human-readable payload for the movement (coordinates, text, target…). */
  summary: string;
};

/** An ordered run of movements captured within a single trajectory. */
export type MovementSequence = {
  trajectoryId: string;
  steps: MovementStep[];
};

export type MovementTrainingRequest = {
  jobId: string;
  mode: TrainingMode;
  sequences: MovementSequence[];
  /** Markov order (context window). Defaults to 2. Clamped to [1, 8]. */
  order?: number;
  /** Injected clock for deterministic tests; defaults to wall clock. */
  now?: () => Date;
};

/**
 * A fully serializable trained model. Plain JSON so it can be persisted next to
 * the training artifacts and reloaded for inference on any host.
 */
export type MovementModelArtifact = {
  version: 1;
  backendId: string;
  jobId: string;
  mode: TrainingMode;
  order: number;
  /** Distinct movement tokens observed (excludes the START/END sentinels). */
  vocabulary: string[];
  /**
   * Backed-off n-gram counts. Keyed by a context of length 0..order (0 = the
   * unigram/prior table), mapping each observed next-token to its count.
   */
  transitions: Record<string, Record<string, number>>;
  sequenceCount: number;
  stepCount: number;
  trainedAt: string;
};

export type MovementPrediction = {
  token: string;
  probability: number;
  /** Context length actually used after backoff (order..0). */
  order: number;
};

export type MovementFidelityReport = {
  sequences: number;
  steps: number;
  correct: number;
  /** Top-1 next-movement accuracy over all scored steps. */
  accuracy: number;
};

/** Pluggable backend seam. A real on-device learner implements the same shape. */
export interface MovementModelBackend {
  readonly id: string;
  train(request: MovementTrainingRequest): Promise<MovementModelArtifact>;
}

const START = "START";
export const MOVEMENT_END = "END";
const CONTEXT_SEPARATOR = "";

function clampOrder(order: number | undefined): number {
  if (order === undefined || !Number.isFinite(order)) {
    return 2;
  }
  return Math.min(8, Math.max(1, Math.floor(order)));
}

function contextKey(tokens: string[]): string {
  return tokens.join(CONTEXT_SEPARATOR);
}

function increment(table: Record<string, Record<string, number>>, ctx: string, token: string): void {
  const row = (table[ctx] ??= {});
  row[token] = (row[token] ?? 0) + 1;
}

/**
 * Deterministic default backend: a backed-off n-gram model over movement
 * tokens. Greedy decoding reproduces the dominant recorded movements (repeat);
 * unseen contexts fall back to shorter histories (generalize). No randomness,
 * no OS access — safe to train and evaluate in the cloud/CI.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov-ngram";

  constructor(private readonly defaultOrder = 2) {}

  async train(request: MovementTrainingRequest): Promise<MovementModelArtifact> {
    const order = clampOrder(request.order ?? this.defaultOrder);
    const transitions: Record<string, Record<string, number>> = {};
    const vocabulary = new Set<string>();
    let stepCount = 0;

    for (const sequence of request.sequences) {
      const tokens = sequence.steps.map((step) => step.tool);
      for (const token of tokens) {
        vocabulary.add(token);
      }
      stepCount += tokens.length;
      // Pad with `order` START sentinels and terminate with END so the model
      // learns how sequences begin and end.
      const padded = [...Array<string>(order).fill(START), ...tokens, MOVEMENT_END];
      for (let i = order; i < padded.length; i += 1) {
        const target = padded[i]!;
        const history = padded.slice(i - order, i);
        // Record every backoff order 0..order for this position.
        for (let k = 0; k <= order; k += 1) {
          const ctx = k === 0 ? "" : contextKey(history.slice(order - k));
          increment(transitions, ctx, target);
        }
      }
    }

    const now = request.now?.() ?? new Date();
    return {
      version: 1,
      backendId: this.id,
      jobId: request.jobId,
      mode: request.mode,
      order,
      vocabulary: [...vocabulary].sort(),
      transitions,
      sequenceCount: request.sequences.length,
      stepCount,
      trainedAt: now.toISOString(),
    };
  }
}

/**
 * Loads a {@link MovementModelArtifact} and generates movements from it. Pure,
 * deterministic decoding: identical seeds always yield identical output.
 */
export class MovementModelInference {
  constructor(private readonly artifact: MovementModelArtifact) {}

  get backendId(): string {
    return this.artifact.backendId;
  }

  /**
   * Predict the single most-likely next movement token for a context, backing
   * off from the full order down to the unigram prior. Ties break by token
   * string for determinism. Returns undefined only for an empty model.
   */
  predictNext(context: string[]): MovementPrediction | undefined {
    const { order, transitions } = this.artifact;
    // Left-pad with START so a short/empty context resolves against the same
    // padded n-grams the model was trained on (e.g. the first movement).
    const history = context.slice(-order);
    while (history.length < order) {
      history.unshift(START);
    }
    for (let k = order; k >= 0; k -= 1) {
      const ctx = k === 0 ? "" : contextKey(history.slice(history.length - k));
      const row = transitions[ctx];
      if (!row) {
        continue;
      }
      let total = 0;
      let bestToken: string | undefined;
      let bestCount = -1;
      for (const [token, count] of Object.entries(row)) {
        total += count;
        if (count > bestCount || (count === bestCount && (bestToken === undefined || token < bestToken))) {
          bestToken = token;
          bestCount = count;
        }
      }
      if (bestToken !== undefined && total > 0) {
        return { token: bestToken, probability: bestCount / total, order: k };
      }
    }
    return undefined;
  }

  /**
   * Greedily decode a full movement sequence from a seed context, stopping at
   * the END sentinel or after `maxSteps`. Reproduces recorded flows and, from
   * novel seeds, produces related flows via backoff.
   */
  generate(options: { seed?: string[]; maxSteps?: number } = {}): string[] {
    const maxSteps = options.maxSteps ?? 64;
    const context = [...(options.seed ?? [])];
    const output: string[] = [];
    for (let i = 0; i < maxSteps; i += 1) {
      const prediction = this.predictNext(context);
      if (!prediction || prediction.token === MOVEMENT_END) {
        break;
      }
      output.push(prediction.token);
      context.push(prediction.token);
    }
    return output;
  }

  /** Top-1 next-movement accuracy for one held-out sequence. */
  scoreSequence(sequence: MovementSequence): { correct: number; total: number } {
    const tokens = sequence.steps.map((step) => step.tool);
    let correct = 0;
    for (let i = 0; i < tokens.length; i += 1) {
      const prediction = this.predictNext(tokens.slice(0, i));
      if (prediction?.token === tokens[i]) {
        correct += 1;
      }
    }
    return { correct, total: tokens.length };
  }
}

/**
 * Generalization eval harness: measure how faithfully a trained model predicts
 * movements on held-out (but related) sequences. Fidelity = top-1 accuracy.
 */
export function evaluateMovementModel(
  inference: MovementModelInference,
  heldOut: MovementSequence[],
): MovementFidelityReport {
  let correct = 0;
  let steps = 0;
  for (const sequence of heldOut) {
    const score = inference.scoreSequence(sequence);
    correct += score.correct;
    steps += score.total;
  }
  return {
    sequences: heldOut.length,
    steps,
    correct,
    accuracy: steps === 0 ? 0 : correct / steps,
  };
}
