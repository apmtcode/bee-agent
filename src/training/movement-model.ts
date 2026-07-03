/**
 * Movement-model subsystem (local-movement learning objective, pieces c + d).
 *
 * The capture pipeline (`src/capture/`) records movement trajectories and the
 * exporter/runner (`src/training/`) turn reviewed trajectories into a dataset and
 * an external launch plan (mlx / axolotl) for on-device training. That external
 * training cannot run in the cloud, so this module provides an **in-process,
 * deterministic, pluggable model backend** that actually learns from recorded
 * movement sequences and can:
 *
 *   - **repeat** recorded movements (objective 2c) — an exact recorded context
 *     yields the recorded continuation, and
 *   - **generalize** to new-but-related movements (objective 2d) — an unseen
 *     context backs off to the longest matching suffix it has seen.
 *
 * The default {@link NGramMovementBackend} is a stupid-backoff Markov model: it
 * needs no native deps, is fully deterministic (argmax with lexical tie-break),
 * serializes to plain JSON, and is validated entirely with synthetic event
 * streams in the cloud. Real on-device backends (an mlx-trained small model,
 * etc.) implement the same {@link MovementModelBackend} seam and can be swapped
 * in via {@link registerMovementBackend} without touching call sites.
 */

import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

/** A single discrete movement, canonicalised to a stable slug token. */
export type MovementToken = string;

/** Sentinel appended to every training sequence so the model can learn to stop. */
export const MOVEMENT_END_TOKEN = "<end>";

/** Separator used to key multi-token contexts in the serialized transition table. */
const CONTEXT_SEPARATOR = "␟";

/** An ordered movement sequence extracted from one or more trajectories. */
export type MovementSequence = {
  /** Trajectory the sequence was derived from, when applicable. */
  sourceTrajectoryId?: string;
  tokens: MovementToken[];
};

export type MovementTrainOptions = {
  /** Highest context length the model conditions on (>= 1). Defaults to 3. */
  order?: number;
};

export type MovementPrediction = {
  token: MovementToken;
  /** Context length actually used after back-off (0 = unigram). */
  order: number;
  /** Empirical probability of the token within the backed-off context. */
  probability: number;
};

export type SerializedMovementModel = {
  version: 1;
  backend: string;
  order: number;
  vocabulary: MovementToken[];
  /** context-key -> { nextToken -> count }. The empty-string key is the unigram. */
  transitions: Record<string, Record<MovementToken, number>>;
};

/** A trained, queryable movement model. Deterministic given the same dataset. */
export interface TrainedMovementModel {
  readonly backend: string;
  readonly order: number;
  /** Most likely next token for a context, using stupid-backoff. */
  predictNext(context: MovementToken[]): MovementPrediction | undefined;
  /**
   * Autoregressively continue a prefix until the end sentinel is predicted or
   * `maxSteps` is reached. The returned tokens never include the sentinel.
   */
  generate(prefix: MovementToken[], maxSteps: number): MovementToken[];
  serialize(): SerializedMovementModel;
}

/** Pluggable training backend. Swap implementations via the registry. */
export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementSequence[], options?: MovementTrainOptions): TrainedMovementModel;
}

// --- Tokenisation -----------------------------------------------------------

function slug(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "") || "x";
}

function metaString(metadata: Record<string, unknown> | undefined, key: string): string | undefined {
  const value = metadata?.[key];
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

/**
 * Canonicalise one recorded action into a stable movement token. Prefers the
 * structured gesture metadata written by the capture adapters (gesture/target/
 * direction) and falls back to a slug of the human summary.
 */
export function tokenizeAction(action: TrajectoryAction): MovementToken {
  const metadata = action.metadata;
  const gesture = metaString(metadata, "gesture");
  const qualifier = metaString(metadata, "target") ?? metaString(metadata, "direction");
  const verb = gesture ?? action.summary;
  const parts = [slug(action.tool), slug(verb)];
  if (qualifier) {
    parts.push(slug(qualifier));
  }
  return parts.join(":");
}

/** Extract an ordered movement sequence from a trajectory span (actions only). */
export function tokenizeTrajectory(span: TrajectorySpan): MovementSequence {
  const tokens = [...span.actions]
    .sort((a, b) => a.ts - b.ts)
    .map((action) => tokenizeAction(action));
  return { sourceTrajectoryId: span.id, tokens };
}

// --- N-gram backend ---------------------------------------------------------

class NGramMovementModel implements TrainedMovementModel {
  constructor(
    readonly backend: string,
    readonly order: number,
    private readonly transitions: Map<string, Map<MovementToken, number>>,
    private readonly vocabulary: Set<MovementToken>,
  ) {}

  predictNext(context: MovementToken[]): MovementPrediction | undefined {
    const maxContext = Math.min(this.order, context.length);
    for (let k = maxContext; k >= 0; k -= 1) {
      const key = context.slice(context.length - k).join(CONTEXT_SEPARATOR);
      const counts = this.transitions.get(key);
      if (!counts) {
        continue;
      }
      const best = argmax(counts);
      if (best) {
        return { token: best.token, order: k, probability: best.count / best.total };
      }
    }
    return undefined;
  }

  generate(prefix: MovementToken[], maxSteps: number): MovementToken[] {
    const context = [...prefix];
    const produced: MovementToken[] = [];
    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predictNext(context);
      if (!prediction || prediction.token === MOVEMENT_END_TOKEN) {
        break;
      }
      produced.push(prediction.token);
      context.push(prediction.token);
    }
    return produced;
  }

  serialize(): SerializedMovementModel {
    const transitions: Record<string, Record<MovementToken, number>> = {};
    for (const [key, counts] of this.transitions) {
      const entry: Record<MovementToken, number> = {};
      for (const [token, count] of counts) {
        entry[token] = count;
      }
      transitions[key] = entry;
    }
    return {
      version: 1,
      backend: this.backend,
      order: this.order,
      vocabulary: [...this.vocabulary].sort(),
      transitions,
    };
  }
}

/**
 * Deterministic stupid-backoff Markov backend. Learns transition counts for
 * every context length in `[0, order]`; prediction uses the longest context
 * with observed continuations, backing off toward the unigram. Ties are broken
 * lexically so a given dataset always yields byte-identical models.
 */
export class NGramMovementBackend implements MovementModelBackend {
  readonly name = "ngram";

  train(dataset: MovementSequence[], options?: MovementTrainOptions): TrainedMovementModel {
    const order = Math.max(1, Math.floor(options?.order ?? 3));
    const transitions = new Map<string, Map<MovementToken, number>>();
    const vocabulary = new Set<MovementToken>();

    for (const sequence of dataset) {
      const tokens = [...sequence.tokens, MOVEMENT_END_TOKEN];
      for (const token of sequence.tokens) {
        vocabulary.add(token);
      }
      for (let i = 0; i < tokens.length; i += 1) {
        const next = tokens[i]!;
        for (let k = 0; k <= order; k += 1) {
          if (i - k < 0) {
            break;
          }
          const key = tokens.slice(i - k, i).join(CONTEXT_SEPARATOR);
          const counts = transitions.get(key) ?? new Map<MovementToken, number>();
          counts.set(next, (counts.get(next) ?? 0) + 1);
          transitions.set(key, counts);
        }
      }
    }

    return new NGramMovementModel(this.name, order, transitions, vocabulary);
  }
}

function argmax(counts: Map<MovementToken, number>): { token: MovementToken; count: number; total: number } | undefined {
  let bestToken: MovementToken | undefined;
  let bestCount = -1;
  let total = 0;
  for (const [token, count] of counts) {
    total += count;
    if (count > bestCount || (count === bestCount && bestToken !== undefined && token < bestToken)) {
      bestToken = token;
      bestCount = count;
    }
  }
  return bestToken === undefined ? undefined : { token: bestToken, count: bestCount, total };
}

/** Rehydrate a model previously produced by {@link TrainedMovementModel.serialize}. */
export function deserializeMovementModel(serialized: SerializedMovementModel): TrainedMovementModel {
  const transitions = new Map<string, Map<MovementToken, number>>();
  for (const [key, counts] of Object.entries(serialized.transitions)) {
    transitions.set(key, new Map(Object.entries(counts)));
  }
  return new NGramMovementModel(
    serialized.backend,
    serialized.order,
    transitions,
    new Set(serialized.vocabulary),
  );
}

// --- Backend registry (pluggable seam for real on-device models) ------------

const BACKENDS = new Map<string, MovementModelBackend>();

export function registerMovementBackend(backend: MovementModelBackend): void {
  BACKENDS.set(backend.name, backend);
}

export function createMovementBackend(name = "ngram"): MovementModelBackend {
  const backend = BACKENDS.get(name);
  if (!backend) {
    throw new Error(`unknown movement backend: ${name} (registered: ${[...BACKENDS.keys()].join(", ") || "none"})`);
  }
  return backend;
}

registerMovementBackend(new NGramMovementBackend());

// --- Evaluation harness (objective 2d: generalization fidelity) -------------

export type MovementEvalResult = {
  /** Held-out next-token predictions attempted (positions after the first). */
  predictions: number;
  /** Predictions matching the recorded next token (teacher-forced). */
  correct: number;
  /** correct / predictions, or 1 when there was nothing to predict. */
  accuracy: number;
};

/**
 * Teacher-forced next-token accuracy on held-out sequences: for each position
 * the model sees the true prefix and must predict the recorded next token
 * (including the end sentinel). This measures replay fidelity on the recorded
 * set and generalization on held-out-but-related sequences.
 */
export function evaluateMovementModel(model: TrainedMovementModel, heldOut: MovementSequence[]): MovementEvalResult {
  let predictions = 0;
  let correct = 0;
  for (const sequence of heldOut) {
    const tokens = [...sequence.tokens, MOVEMENT_END_TOKEN];
    for (let i = 0; i < tokens.length; i += 1) {
      predictions += 1;
      const prediction = model.predictNext(tokens.slice(0, i));
      if (prediction?.token === tokens[i]) {
        correct += 1;
      }
    }
  }
  return {
    predictions,
    correct,
    accuracy: predictions === 0 ? 1 : correct / predictions,
  };
}
