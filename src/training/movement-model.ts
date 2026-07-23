import type { ReplayTimelineEvent } from "../capture/replay.js";

/**
 * In-process movement-prediction model.
 *
 * The rest of the training subsystem (runner/execution-service) emits an
 * external `mlx`/`axolotl` launch plan that only executes on a real Apple
 * Silicon machine. This module provides the complementary piece the
 * self-evolution objective calls for: a *pluggable, local* model that can be
 * trained on a recorded movement dataset entirely in-process, so it can
 *
 *   (c) repeat recorded movements, and
 *   (d) generalize to new-but-related movements,
 *
 * and — critically — be validated in the cloud/CI with synthetic event
 * streams (no real OS input required). The default backend is a deterministic
 * n-gram (variable-order Markov) model with stupid-backoff, which is small,
 * fast, and needs no native deps. The `MovementModelBackend` interface is the
 * documented seam where a real on-device small model (e.g. an MLX-trained
 * policy) can be dropped in without changing callers.
 */

/** A single tokenized movement/action symbol in a trajectory. */
export type MovementToken = string;

/** An ordered sequence of movement tokens (one recorded trajectory). */
export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

/** Sentinel token appended to sequences so the model can learn to stop. */
export const MOVEMENT_BOUNDARY_TOKEN = "␂end";

const CONTEXT_SEPARATOR = "";
const EMPTY_CONTEXT_KEY = "";

/**
 * Reduce a replay timeline event to a stable discrete symbol. Movement/action
 * events (device gestures, tool calls) carry the signal; observations and
 * transcript turns are tokenized coarsely so they act as context, not noise.
 */
export function tokenizeMovementEvent(event: ReplayTimelineEvent): MovementToken {
  switch (event.kind) {
    case "action":
      return `act:${slug(event.tool)}:${slug(event.summary)}`;
    case "observation":
      return `obs:${slug(event.source)}:${slug(event.summary)}`;
    case "transcript":
      return `msg:${event.role}`;
  }
}

/** Build a movement sequence from a recorded replay timeline. */
export function buildMovementSequence(id: string, events: ReplayTimelineEvent[]): MovementSequence {
  return { id, tokens: events.map(tokenizeMovementEvent) };
}

export type TrainMovementModelOptions = {
  /** Maximum n-gram context length. Higher = more faithful replay, less generalization. Default 3. */
  order?: number;
  /** Append {@link MOVEMENT_BOUNDARY_TOKEN} to each sequence so rollouts learn to stop. Default true. */
  appendBoundaryToken?: boolean;
};

/**
 * Serializable trained model. `contexts[key]` maps a k-gram context (0 <= k <=
 * order) to a token->count distribution; keeping every context length is what
 * enables stupid-backoff at inference time.
 */
export type TrainedMovementModel = {
  version: 1;
  backend: string;
  order: number;
  boundaryToken: string | null;
  vocabulary: MovementToken[];
  contexts: Record<string, Record<MovementToken, number>>;
  sequenceCount: number;
  eventCount: number;
};

export type MovementPrediction = {
  /** Predicted next token, or null when the model has no information at all. */
  token: MovementToken | null;
  /** Length of the context suffix that actually produced the prediction. */
  contextLength: number;
  /** Estimated probability of {@link token} under the matched context. */
  probability: number;
  /** True when the full requested context was unseen and the model backed off. */
  backedOff: boolean;
};

/**
 * Pluggable movement-model backend. The default {@link NgramMovementBackend} is
 * deterministic and dependency-free; a real on-device backend implements the
 * same three members and can be swapped in via {@link MovementModelRegistry}.
 */
export interface MovementModelBackend {
  readonly name: string;
  train(sequences: MovementSequence[], options?: TrainMovementModelOptions): Promise<TrainedMovementModel>;
  predictNext(model: TrainedMovementModel, context: MovementToken[]): MovementPrediction;
}

/** Deterministic variable-order Markov model with stupid-backoff. */
export class NgramMovementBackend implements MovementModelBackend {
  readonly name = "ngram";

  async train(
    sequences: MovementSequence[],
    options: TrainMovementModelOptions = {},
  ): Promise<TrainedMovementModel> {
    const order = Math.max(1, Math.floor(options.order ?? 3));
    const appendBoundary = options.appendBoundaryToken ?? true;
    const boundaryToken = appendBoundary ? MOVEMENT_BOUNDARY_TOKEN : null;

    const contexts: Record<string, Record<MovementToken, number>> = {};
    const vocabulary = new Set<MovementToken>();
    let eventCount = 0;

    for (const sequence of sequences) {
      const tokens = boundaryToken ? [...sequence.tokens, boundaryToken] : [...sequence.tokens];
      for (let i = 0; i < tokens.length; i += 1) {
        const next = tokens[i]!;
        vocabulary.add(next);
        eventCount += 1;
        // Record next-token counts for every context length 0..order.
        const maxK = Math.min(order, i);
        for (let k = 0; k <= maxK; k += 1) {
          const context = tokens.slice(i - k, i);
          const key = contextKey(context);
          const distribution = (contexts[key] ??= {});
          distribution[next] = (distribution[next] ?? 0) + 1;
        }
      }
    }

    return {
      version: 1,
      backend: this.name,
      order,
      boundaryToken,
      vocabulary: [...vocabulary].sort(),
      contexts,
      sequenceCount: sequences.length,
      eventCount,
    };
  }

  predictNext(model: TrainedMovementModel, context: MovementToken[]): MovementPrediction {
    const requested = Math.min(context.length, model.order);
    for (let k = requested; k >= 0; k -= 1) {
      const suffix = context.slice(context.length - k);
      const distribution = model.contexts[contextKey(suffix)];
      if (!distribution) {
        continue;
      }
      const best = argmax(distribution);
      if (!best) {
        continue;
      }
      const total = Object.values(distribution).reduce((sum, count) => sum + count, 0);
      return {
        token: best.token,
        contextLength: k,
        probability: total > 0 ? best.count / total : 0,
        backedOff: k < requested,
      };
    }
    return { token: null, contextLength: 0, probability: 0, backedOff: requested > 0 };
  }
}

export type RolloutMovementParams = {
  /** Tokens already observed; the rollout continues from here. Default empty. */
  seed?: MovementToken[];
  /** Hard cap on generated tokens (excluding the seed). */
  maxLength: number;
  /** Stop (without emitting) when the model predicts the boundary token. Default true. */
  stopAtBoundary?: boolean;
};

/**
 * Autoregressively generate a movement sequence from a trained model. This is
 * the "repeat / generalize movements" capability: seeded with a recorded prefix
 * the model reproduces the recorded continuation; seeded with a novel-but-
 * related prefix it composes seen sub-sequences via backoff.
 */
export function rolloutMovements(
  backend: MovementModelBackend,
  model: TrainedMovementModel,
  params: RolloutMovementParams,
): MovementToken[] {
  const stopAtBoundary = params.stopAtBoundary ?? true;
  const history: MovementToken[] = [...(params.seed ?? [])];
  const generated: MovementToken[] = [];

  for (let step = 0; step < params.maxLength; step += 1) {
    const prediction = backend.predictNext(model, history);
    if (prediction.token === null) {
      break;
    }
    if (stopAtBoundary && model.boundaryToken !== null && prediction.token === model.boundaryToken) {
      break;
    }
    generated.push(prediction.token);
    history.push(prediction.token);
  }

  return generated;
}

export type MovementReplayEvaluation = {
  sequenceCount: number;
  /** Number of next-token predictions scored (teacher-forced). */
  predictions: number;
  correct: number;
  /** Fraction of next-token predictions that matched the recorded token. */
  accuracy: number;
  /** Predictions that required backing off to a shorter context. */
  backoffPredictions: number;
  backoffRate: number;
};

/**
 * Teacher-forced next-token fidelity over held-out sequences. Each position is
 * scored as "given the preceding recorded movements, predict the next one", so
 * scoring starts at the first position that actually has a preceding context
 * (position 0 has none and is a pure prior, not a replay task). On the training
 * set this measures replay faithfulness (1.0 for unambiguous flows); on held-out
 * but related sequences it measures generalization.
 */
export function evaluateMovementReplayFidelity(
  backend: MovementModelBackend,
  model: TrainedMovementModel,
  heldOut: MovementSequence[],
  options: { includeBoundary?: boolean } = {},
): MovementReplayEvaluation {
  const includeBoundary = options.includeBoundary ?? Boolean(model.boundaryToken);
  let predictions = 0;
  let correct = 0;
  let backoffPredictions = 0;

  for (const sequence of heldOut) {
    const tokens =
      includeBoundary && model.boundaryToken ? [...sequence.tokens, model.boundaryToken] : [...sequence.tokens];
    for (let i = 1; i < tokens.length; i += 1) {
      const prediction = backend.predictNext(model, tokens.slice(0, i));
      predictions += 1;
      if (prediction.token === tokens[i]) {
        correct += 1;
      }
      if (prediction.backedOff) {
        backoffPredictions += 1;
      }
    }
  }

  return {
    sequenceCount: heldOut.length,
    predictions,
    correct,
    accuracy: predictions > 0 ? correct / predictions : 0,
    backoffPredictions,
    backoffRate: predictions > 0 ? backoffPredictions / predictions : 0,
  };
}

export type MovementTemplate = {
  id: string;
  tokens: MovementToken[];
};

/**
 * Deterministically expand named templates into a dataset. Useful for
 * validating capture->train->replay round-trips and generalization without any
 * real OS input. `repeats` duplicates each template (with a suffixed id) so
 * frequency-based backends have mass to learn from.
 */
export function synthesizeMovementSequences(spec: {
  templates: MovementTemplate[];
  repeats?: number;
}): MovementSequence[] {
  const repeats = Math.max(1, Math.floor(spec.repeats ?? 1));
  const sequences: MovementSequence[] = [];
  for (const template of spec.templates) {
    for (let copy = 0; copy < repeats; copy += 1) {
      sequences.push({
        id: repeats === 1 ? template.id : `${template.id}#${copy}`,
        tokens: [...template.tokens],
      });
    }
  }
  return sequences;
}

/** Registry that makes the movement-model backend pluggable by name. */
export class MovementModelRegistry {
  private readonly backends = new Map<string, MovementModelBackend>();

  register(backend: MovementModelBackend): this {
    this.backends.set(backend.name, backend);
    return this;
  }

  has(name: string): boolean {
    return this.backends.has(name);
  }

  get(name: string): MovementModelBackend {
    const backend = this.backends.get(name);
    if (!backend) {
      throw new Error(`Unknown movement-model backend: ${name}`);
    }
    return backend;
  }

  list(): string[] {
    return [...this.backends.keys()].sort();
  }
}

/** Registry pre-seeded with the built-in deterministic n-gram backend. */
export function createDefaultMovementModelRegistry(): MovementModelRegistry {
  return new MovementModelRegistry().register(new NgramMovementBackend());
}

function slug(value: string): string {
  const normalized = value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return normalized.length > 0 ? normalized : "_";
}

function contextKey(context: MovementToken[]): string {
  return context.length === 0 ? EMPTY_CONTEXT_KEY : context.join(CONTEXT_SEPARATOR);
}

function argmax(distribution: Record<MovementToken, number>): { token: MovementToken; count: number } | undefined {
  let bestToken: MovementToken | undefined;
  let bestCount = -Infinity;
  for (const [token, count] of Object.entries(distribution)) {
    // Deterministic tie-break: higher count wins, then lexicographically smaller token.
    if (count > bestCount || (count === bestCount && (bestToken === undefined || token < bestToken))) {
      bestToken = token;
      bestCount = count;
    }
  }
  return bestToken === undefined ? undefined : { token: bestToken, count: bestCount };
}
