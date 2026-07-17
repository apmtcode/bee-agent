import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

/**
 * In-process, pluggable movement-model backend for the local-movement learning
 * subsystem (standing objective #2 (c) + (d)).
 *
 * The real on-device path (`LocalAppleSiliconTrainingRunner`) shells out to
 * mlx/axolotl and can only run on the user's machine. This module provides the
 * *cloud-testable* half: a movement-token schema, a dataset builder, a
 * `MovementModelBackend` seam with a deterministic mock backend that actually
 * learns from a dataset, and a generalization eval harness. Everything here is
 * pure and deterministic (no `Date`, no `Math.random`) so it runs identically
 * in CI and on-device, and so a real learned backend can be swapped in behind
 * the same interface later.
 */

/** A canonical, bounded string describing one movement/action. */
export type MovementToken = string;

/** An ordered movement sequence extracted from a single trajectory. */
export type MovementSequence = {
  trajectoryId: string;
  tokens: MovementToken[];
};

/** A training-ready dataset of movement sequences with a derived vocabulary. */
export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
  vocabulary: MovementToken[];
};

/** Result of asking a model to predict the next movement. */
export type MovementPrediction = {
  /** Predicted next token, or `undefined` when the model has no basis at all. */
  token: MovementToken | undefined;
  /** Fraction of observed continuations for the matched context (0..1). */
  confidence: number;
  /** Backoff order actually used (context length matched); -1 if none. */
  order: number;
};

/** A trained movement model that can predict and generate movements. */
export interface MovementModel {
  /** Maximum context order the model was trained with. */
  readonly order: number;
  /** Distinct tokens the model has seen. */
  readonly vocabulary: MovementToken[];
  /** Predict the single most-likely next movement given a context prefix. */
  predictNext(context: MovementToken[]): MovementPrediction;
  /** Autoregressively generate up to `steps` movements from a seed prefix. */
  generate(seed: MovementToken[], steps: number): MovementToken[];
}

export type MovementTrainingOptions = {
  /** Highest n-gram order to learn (context length). Defaults to 3. */
  order?: number;
};

/** Pluggable backend seam: train a dataset into a `MovementModel`. */
export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, options?: MovementTrainingOptions): MovementModel;
}

const TOKEN_FIELD_SEP = "/";
const CONTEXT_SEP = "";

/** Lowercase, delimiter-safe slug used for free-form token fields. */
export function slugMovementField(value: string): string {
  const slug = value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return slug.length > 0 ? slug : "unknown";
}

/**
 * Reduce a captured action to a canonical, bounded movement token. Uses the
 * structured gesture metadata (`gesture`/`direction`/`target`) when present so
 * the vocabulary stays small and generalizable; falls back to a slug of the
 * human summary otherwise.
 */
export function tokenizeMovement(action: TrajectoryAction): MovementToken {
  const parts: string[] = [slugMovementField(action.tool)];
  const metadata = action.metadata ?? {};
  const gesture = metadata["gesture"];
  const direction = metadata["direction"];
  const target = metadata["target"];
  let structured = false;
  if (typeof gesture === "string" && gesture.length > 0) {
    parts.push(slugMovementField(gesture));
    structured = true;
  }
  if (typeof direction === "string" && direction.length > 0) {
    parts.push(slugMovementField(direction));
    structured = true;
  }
  if (typeof target === "string" && target.length > 0) {
    parts.push(slugMovementField(target));
    structured = true;
  }
  if (!structured) {
    parts.push(slugMovementField(action.summary));
  }
  return parts.join(TOKEN_FIELD_SEP);
}

/** Build a training dataset of movement sequences from captured trajectories. */
export function buildMovementDataset(trajectories: TrajectorySpan[]): MovementDataset {
  const sequences: MovementSequence[] = [];
  const vocabulary = new Set<MovementToken>();
  for (const trajectory of trajectories) {
    const orderedActions = [...trajectory.actions].sort((a, b) => a.ts - b.ts);
    if (orderedActions.length === 0) {
      continue;
    }
    const tokens = orderedActions.map((action) => {
      const token = tokenizeMovement(action);
      vocabulary.add(token);
      return token;
    });
    sequences.push({ trajectoryId: trajectory.id, tokens });
  }
  return {
    version: 1,
    sequences,
    vocabulary: [...vocabulary].sort(),
  };
}

/**
 * Deterministic variable-order Markov backend with stupid-backoff.
 *
 * It memorizes exact recorded movement chains (high-order contexts) so replay
 * of seen skills is exact, and backs off to shorter-context statistics for
 * unseen prefixes so it *generalizes* to new-but-related movements — the two
 * behaviours objective #2 (c) and (d) require. Fully deterministic: ties break
 * by token order, so training the same dataset always yields the same model.
 */
export class DeterministicMovementBackend implements MovementModelBackend {
  readonly name = "deterministic-markov";

  train(dataset: MovementDataset, options: MovementTrainingOptions = {}): MovementModel {
    const order = Math.max(1, Math.floor(options.order ?? 3));
    // counts[k] maps a context of exactly k tokens -> (nextToken -> count).
    const counts: Array<Map<string, Map<MovementToken, number>>> = [];
    for (let k = 0; k <= order; k += 1) {
      counts.push(new Map());
    }
    for (const sequence of dataset.sequences) {
      const tokens = sequence.tokens;
      for (let i = 0; i < tokens.length; i += 1) {
        const next = tokens[i]!;
        const maxContext = Math.min(order, i);
        for (let k = 0; k <= maxContext; k += 1) {
          const context = tokens.slice(i - k, i);
          const key = context.join(CONTEXT_SEP);
          const bucket = counts[k]!;
          let continuations = bucket.get(key);
          if (!continuations) {
            continuations = new Map();
            bucket.set(key, continuations);
          }
          continuations.set(next, (continuations.get(next) ?? 0) + 1);
        }
      }
    }
    return new DeterministicMovementModel(order, [...dataset.vocabulary], counts);
  }
}

class DeterministicMovementModel implements MovementModel {
  constructor(
    readonly order: number,
    readonly vocabulary: MovementToken[],
    private readonly counts: Array<Map<string, Map<MovementToken, number>>>,
  ) {}

  predictNext(context: MovementToken[]): MovementPrediction {
    const maxK = Math.min(this.order, context.length);
    for (let k = maxK; k >= 0; k -= 1) {
      const key = context.slice(context.length - k).join(CONTEXT_SEP);
      const continuations = this.counts[k]?.get(key);
      if (!continuations || continuations.size === 0) {
        continue;
      }
      let total = 0;
      let best: MovementToken | undefined;
      let bestCount = -1;
      for (const [token, count] of continuations) {
        total += count;
        if (count > bestCount || (count === bestCount && best !== undefined && token < best)) {
          best = token;
          bestCount = count;
        }
      }
      if (best !== undefined) {
        return { token: best, confidence: total > 0 ? bestCount / total : 0, order: k };
      }
    }
    return { token: undefined, confidence: 0, order: -1 };
  }

  generate(seed: MovementToken[], steps: number): MovementToken[] {
    const generated: MovementToken[] = [];
    const context = [...seed];
    for (let i = 0; i < steps; i += 1) {
      const prediction = this.predictNext(context);
      if (prediction.token === undefined) {
        break;
      }
      generated.push(prediction.token);
      context.push(prediction.token);
    }
    return generated;
  }
}

/** Per-backoff-order accuracy breakdown. */
export type MovementEvalByOrder = Record<number, { predictions: number; correct: number }>;

export type MovementEvalResult = {
  totalPredictions: number;
  correct: number;
  accuracy: number;
  averageConfidence: number;
  /** Accuracy split by the backoff order the model used for each prediction. */
  byOrder: MovementEvalByOrder;
};

/**
 * Generalization eval harness: next-token top-1 accuracy over held-out
 * sequences. Feeding *held-out* (but related) trajectories measures whether the
 * model generalizes rather than merely memorizes its training set.
 */
export function evaluateMovementModel(
  model: MovementModel,
  heldOut: MovementSequence[],
): MovementEvalResult {
  let totalPredictions = 0;
  let correct = 0;
  let confidenceSum = 0;
  const byOrder: MovementEvalByOrder = {};
  for (const sequence of heldOut) {
    const tokens = sequence.tokens;
    for (let i = 1; i < tokens.length; i += 1) {
      const context = tokens.slice(0, i);
      const prediction = model.predictNext(context);
      totalPredictions += 1;
      confidenceSum += prediction.confidence;
      const bucket = byOrder[prediction.order] ?? { predictions: 0, correct: 0 };
      bucket.predictions += 1;
      if (prediction.token === tokens[i]) {
        correct += 1;
        bucket.correct += 1;
      }
      byOrder[prediction.order] = bucket;
    }
  }
  return {
    totalPredictions,
    correct,
    accuracy: totalPredictions > 0 ? correct / totalPredictions : 0,
    averageConfidence: totalPredictions > 0 ? confidenceSum / totalPredictions : 0,
    byOrder,
  };
}
