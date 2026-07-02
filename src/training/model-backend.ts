/**
 * Pluggable movement-model backend for the local-movement learning subsystem.
 *
 * Objective #2 of the self-evolution charter needs bee-agent to (c) post-train a
 * local model on a recorded-movement dataset so it can *repeat* the recorded
 * movements, and (d) *generalize* to new-but-related movements. The real
 * on-device training path (mlx / axolotl) is emitted by
 * {@link LocalAppleSiliconTrainingRunner} as a launch script, but that cannot
 * run in the cloud/CI where this engine evolves the code.
 *
 * This module provides the missing seam: a backend-agnostic interface plus a
 * fully in-process, dependency-free, deterministic backend so the whole
 * train -> infer -> generalize loop can be exercised and tested anywhere. A
 * real small on-device model can implement {@link MovementModelBackend} later
 * without touching callers.
 *
 * Everything here is pure (no fs / no clock / no randomness), so trained models
 * are plain JSON — persistable, diffable, and replayable.
 */

/** A single discrete movement primitive (e.g. `"device:swipe:down"`). */
export type MovementToken = string;

/** One recorded (or synthetic) ordered sequence of movements. */
export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
  sessionId?: string;
  outcome?: "success" | "failure" | "aborted";
};

/** A dataset the backend trains on. `vocabulary` is the sorted unique token set. */
export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
  vocabulary: MovementToken[];
};

/** A trained model. Plain JSON so it is persistable and backend-portable. */
export type TrainedMovementModel = {
  backend: string;
  version: 1;
  maxOrder: number;
  vocabulary: MovementToken[];
  /** key = `${order}␟${context.join("␟")}` -> { nextToken: count }. */
  transitions: Record<string, Record<MovementToken, number>>;
  /** Global token frequencies, used as the order-0 backoff distribution. */
  unigram: Record<MovementToken, number>;
  sequenceCount: number;
  tokenCount: number;
};

/** A ranked next-token candidate. */
export type MovementCandidate = {
  token: MovementToken;
  probability: number;
};

/**
 * The result of asking the model what comes next.
 * `order` reports which Markov order produced the answer (the exact-match order,
 * `0` for the unigram fallback, `-1` when the model is empty) — useful for
 * telling "repeated a recorded move" (high order) from "generalized" (backed off
 * to a lower order).
 */
export type MovementPrediction = {
  token: MovementToken | null;
  confidence: number;
  order: number;
  candidates: MovementCandidate[];
};

export type TrainMovementOptions = {
  /** Highest Markov order to learn. Defaults to 3. */
  maxOrder?: number;
};

export type GenerateMovementOptions = {
  /** Maximum tokens to emit. */
  maxSteps: number;
  /** Optional token that halts generation when produced. */
  stopToken?: MovementToken;
  /**
   * Minimum confidence required to keep generating. Once a prediction falls
   * below this, generation stops (prevents low-signal rambling). Default 0.
   */
  minConfidence?: number;
};

/**
 * The pluggable seam. A deterministic mock ({@link DeterministicMarkovBackend})
 * ships here; a real on-device small model can implement the same three methods.
 */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: TrainMovementOptions): Promise<TrainedMovementModel>;
  predictNext(model: TrainedMovementModel, context: MovementToken[]): MovementPrediction;
  generate(
    model: TrainedMovementModel,
    prompt: MovementToken[],
    options: GenerateMovementOptions,
  ): MovementToken[];
}

const CONTEXT_SEPARATOR = "␟";
const DEFAULT_MAX_ORDER = 3;

function transitionKey(order: number, context: MovementToken[]): string {
  return `${order}${CONTEXT_SEPARATOR}${context.join(CONTEXT_SEPARATOR)}`;
}

/**
 * Turn a frequency table into a ranked, normalized prediction. Ties are broken
 * deterministically (higher count first, then lexicographic token order) so the
 * same model always yields the same answer — a hard requirement for replay and
 * for reproducible tests.
 */
function toPrediction(distribution: Record<MovementToken, number>, order: number): MovementPrediction {
  const entries = Object.entries(distribution);
  const total = entries.reduce((sum, [, count]) => sum + count, 0);
  if (total === 0) {
    return { token: null, confidence: 0, order: -1, candidates: [] };
  }
  const candidates: MovementCandidate[] = entries
    .map(([token, count]) => ({ token, probability: count / total }))
    .sort((a, b) => {
      if (b.probability !== a.probability) {
        return b.probability - a.probability;
      }
      return a.token < b.token ? -1 : a.token > b.token ? 1 : 0;
    });
  return {
    token: candidates[0]?.token ?? null,
    confidence: candidates[0]?.probability ?? 0,
    order,
    candidates,
  };
}

/**
 * A variable-order Markov model with stupid-backoff. It learns next-token
 * frequencies for every context length up to `maxOrder`. At inference it uses
 * the longest context suffix it has seen and backs off to shorter contexts
 * (finally to the global unigram) when a longer context is novel.
 *
 * Why this satisfies the objective:
 *  - **Repeat**: a context that appeared verbatim in training resolves at the
 *    highest matching order to the recorded next move.
 *  - **Generalize**: a novel context built from familiar tokens still resolves,
 *    via backoff, to a plausible continuation — new-but-related movement.
 */
export class DeterministicMarkovBackend implements MovementModelBackend {
  readonly id = "deterministic-markov";

  async train(dataset: MovementDataset, options: TrainMovementOptions = {}): Promise<TrainedMovementModel> {
    const maxOrder = Math.max(1, options.maxOrder ?? DEFAULT_MAX_ORDER);
    const transitions: Record<string, Record<MovementToken, number>> = {};
    const unigram: Record<MovementToken, number> = {};
    let tokenCount = 0;

    for (const sequence of dataset.sequences) {
      const tokens = sequence.tokens;
      for (const token of tokens) {
        unigram[token] = (unigram[token] ?? 0) + 1;
        tokenCount += 1;
      }
      for (let order = 1; order <= maxOrder; order += 1) {
        for (let i = order; i < tokens.length; i += 1) {
          const context = tokens.slice(i - order, i);
          const next = tokens[i]!;
          const key = transitionKey(order, context);
          const distribution = (transitions[key] ??= {});
          distribution[next] = (distribution[next] ?? 0) + 1;
        }
      }
    }

    const vocabulary =
      dataset.vocabulary.length > 0 ? [...dataset.vocabulary] : computeVocabulary(dataset.sequences);

    return {
      backend: this.id,
      version: 1,
      maxOrder,
      vocabulary,
      transitions,
      unigram,
      sequenceCount: dataset.sequences.length,
      tokenCount,
    };
  }

  predictNext(model: TrainedMovementModel, context: MovementToken[]): MovementPrediction {
    const maxUsableOrder = Math.min(model.maxOrder, context.length);
    for (let order = maxUsableOrder; order >= 1; order -= 1) {
      const suffix = context.slice(context.length - order);
      const distribution = model.transitions[transitionKey(order, suffix)];
      if (distribution) {
        return toPrediction(distribution, order);
      }
    }
    if (Object.keys(model.unigram).length > 0) {
      return toPrediction(model.unigram, 0);
    }
    return { token: null, confidence: 0, order: -1, candidates: [] };
  }

  generate(
    model: TrainedMovementModel,
    prompt: MovementToken[],
    options: GenerateMovementOptions,
  ): MovementToken[] {
    const minConfidence = options.minConfidence ?? 0;
    const produced: MovementToken[] = [];
    const context = [...prompt];
    for (let step = 0; step < options.maxSteps; step += 1) {
      const prediction = this.predictNext(model, context);
      if (prediction.token === null || prediction.confidence < minConfidence) {
        break;
      }
      produced.push(prediction.token);
      context.push(prediction.token);
      if (options.stopToken !== undefined && prediction.token === options.stopToken) {
        break;
      }
    }
    return produced;
  }
}

function computeVocabulary(sequences: MovementSequence[]): MovementToken[] {
  const seen = new Set<MovementToken>();
  for (const sequence of sequences) {
    for (const token of sequence.tokens) {
      seen.add(token);
    }
  }
  return [...seen].sort();
}

// ---------------------------------------------------------------------------
// Tokenization + dataset construction from the existing capture/replay types.
// ---------------------------------------------------------------------------

/** Minimal action shape shared by TrajectoryAction and replay action events. */
type TokenizableAction = {
  tool: string;
  summary: string;
  metadata?: Record<string, unknown>;
};

/** Minimal observation shape shared by TrajectoryObservation and replay events. */
type TokenizableObservation = {
  source: string;
  summary: string;
  metadata?: Record<string, unknown>;
};

function normalizeVerb(summary: string): string {
  const word = summary.trim().toLowerCase().split(/\s+/)[0] ?? "";
  const cleaned = word.replace(/[^a-z0-9]/g, "");
  return cleaned.length > 0 ? cleaned : "event";
}

function metadataString(metadata: Record<string, unknown> | undefined, key: string): string | undefined {
  const value = metadata?.[key];
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

/**
 * Produce a stable movement token for an action. Prefers structured metadata
 * (gesture kind + direction) so semantically-identical movements collapse to the
 * same token regardless of free-text summary wording; falls back to the leading
 * verb of the summary.
 */
export function tokenizeAction(action: TokenizableAction): MovementToken {
  const tool = action.tool.trim().length > 0 ? action.tool.trim() : "action";
  const gesture = metadataString(action.metadata, "gesture");
  const primitive = gesture ?? normalizeVerb(action.summary);
  const direction = metadataString(action.metadata, "direction");
  return [tool, primitive, ...(direction ? [direction] : [])].join(":");
}

/** Produce a stable movement token for an observation (window/UI/OS event). */
export function tokenizeObservation(observation: TokenizableObservation): MovementToken {
  const source = observation.source.trim().length > 0 ? observation.source.trim() : "observation";
  const event = metadataString(observation.metadata, "event");
  const primitive = event ?? normalizeVerb(observation.summary);
  return [source, primitive].join(":");
}

/**
 * Build a training dataset from trajectory-like spans. Each span becomes one
 * sequence of action tokens (in recorded order). Redacted/reviewed actions are
 * used when present so training never sees anything the reviewer removed.
 */
export function buildMovementDatasetFromTrajectories(
  trajectories: Array<{
    id: string;
    sessionId?: string;
    actions: TokenizableAction[];
    outcome?: { status: "success" | "failure" | "aborted" };
    review?: { redactedActions?: TokenizableAction[] };
  }>,
): MovementDataset {
  const sequences: MovementSequence[] = trajectories.map((trajectory) => {
    const actions = trajectory.review?.redactedActions ?? trajectory.actions;
    return {
      id: trajectory.id,
      tokens: actions.map((action) => tokenizeAction(action)),
      ...(trajectory.sessionId ? { sessionId: trajectory.sessionId } : {}),
      ...(trajectory.outcome ? { outcome: trajectory.outcome.status } : {}),
    };
  });
  return { version: 1, sequences, vocabulary: computeVocabulary(sequences) };
}

/**
 * Build a single ordered movement sequence from a replay manifest's timeline.
 * Includes both `action` and `observation` events (the full movement/UI stream),
 * skipping `transcript` events which are not movements.
 */
export function buildMovementSequenceFromReplayEvents(
  sessionId: string,
  events: Array<
    | ({ kind: "action" } & TokenizableAction)
    | ({ kind: "observation" } & TokenizableObservation)
    | { kind: "transcript" }
  >,
): MovementSequence {
  const tokens: MovementToken[] = [];
  for (const event of events) {
    if (event.kind === "action") {
      tokens.push(tokenizeAction(event));
    } else if (event.kind === "observation") {
      tokens.push(tokenizeObservation(event));
    }
  }
  return { id: sessionId, tokens, sessionId };
}

// ---------------------------------------------------------------------------
// Generalization eval harness (objective 2d): held-out next-token fidelity.
// ---------------------------------------------------------------------------

export type MovementEvalResult = {
  /** Number of (context -> next) prediction points scored. */
  predictions: number;
  /** Correct top-1 next-token predictions. */
  correct: number;
  /** correct / predictions (0 when there were no predictions). */
  accuracy: number;
  /** Mean Markov order used across scored points (signal of repeat vs backoff). */
  meanOrder: number;
};

/**
 * Score a trained model on held-out sequences by walking each sequence and, at
 * every position, asking the model to predict the actual next token from the
 * prefix so far. This is the seam for measuring both faithful *repetition*
 * (contexts seen in training) and *generalization* (novel contexts resolved via
 * backoff) — the caller decides which sequences to feed.
 */
export function evaluateMovementModel(
  backend: MovementModelBackend,
  model: TrainedMovementModel,
  sequences: MovementSequence[],
): MovementEvalResult {
  let predictions = 0;
  let correct = 0;
  let orderSum = 0;
  for (const sequence of sequences) {
    for (let i = 1; i < sequence.tokens.length; i += 1) {
      const prediction = backend.predictNext(model, sequence.tokens.slice(0, i));
      predictions += 1;
      orderSum += Math.max(prediction.order, 0);
      if (prediction.token === sequence.tokens[i]) {
        correct += 1;
      }
    }
  }
  return {
    predictions,
    correct,
    accuracy: predictions === 0 ? 0 : correct / predictions,
    meanOrder: predictions === 0 ? 0 : orderSum / predictions,
  };
}
