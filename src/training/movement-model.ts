import type { ReplayManifest } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Local-movement learning subsystem — in-process model (objective #2 (c)+(d)).
 *
 * This module implements the "post-train a local model on the recorded dataset
 * to repeat the movements, and generalize to new but related movements" piece of
 * the movement subsystem. The training runner (`runner.ts`) shells out to an
 * external Apple-Silicon toolchain (mlx/axolotl) that cannot run in the cloud;
 * this model is a fully deterministic, dependency-free learner that DOES run in
 * the cloud, so the capture -> dataset -> train -> infer -> generalize pipeline
 * can be validated end-to-end with synthetic event streams and in CI.
 *
 * The backend is pluggable via {@link MovementModelBackend}: a real on-device
 * neural policy can implement the same interface and slot in behind it. The
 * default {@link NgramMovementModel} is an order-k Markov model with Katz-style
 * backoff — deterministic argmax decoding (no RNG), so training and inference
 * are reproducible and testable.
 */

/** A single learnable movement, distilled from a recorded action/gesture. */
export type MovementToken = {
  /** Gesture kind ("tap"/"swipe"/"scroll"/"type"/"shortcut") or tool name. */
  kind: string;
  /** UI element / target the movement acted on, if known. */
  target?: string;
  /** Directional component ("up"/"down"/"left"/"right"), if any. */
  direction?: string;
};

/** An ordered movement sequence, e.g. one recorded trajectory. */
export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

/** Serializable, persisted model — this is the "trained model" artifact. */
export type MovementModelState = {
  version: 1;
  /** Maximum context length (k) the model conditions on. */
  order: number;
  /** Sorted, de-duplicated token keys (the vocabulary). */
  vocabulary: string[];
  /** key -> canonical token, so states round-trip without ambiguity. */
  tokens: Record<string, MovementToken>;
  /** contextKey -> nextTokenKey -> count. Empty context key is the unigram. */
  transitions: Record<string, Record<string, number>>;
  /** firstTokenKey -> count, used to pick a deterministic generation seed. */
  starts: Record<string, number>;
  sequenceCount: number;
  tokenCount: number;
};

/** A single next-movement prediction with backoff provenance. */
export type MovementPrediction = {
  token: MovementToken;
  /** count / total at the backed-off context (a conditional probability). */
  probability: number;
  /** How many prior tokens were actually conditioned on (backoff depth). */
  contextOrder: number;
  /** Raw observation count supporting this prediction. */
  observationCount: number;
};

export type GenerateOptions = {
  /** Prefix to continue from. Defaults to the most common recorded start. */
  seed?: MovementToken[];
  /** Hard cap on total emitted tokens (including seed). Defaults to 64. */
  maxLength?: number;
  /** Stop once this token is emitted. */
  stopToken?: MovementToken;
};

/**
 * Pluggable backend contract. A real neural on-device model can implement this
 * and be swapped in wherever an {@link NgramMovementModel} is used today.
 */
export interface MovementModelBackend {
  readonly id: string;
  readonly order: number;
  train(sequences: MovementSequence[]): MovementModelState;
  predictNext(state: MovementModelState, context: MovementToken[]): MovementPrediction | undefined;
  generate(state: MovementModelState, options?: GenerateOptions): MovementToken[];
}

const CONTEXT_SEPARATOR = "";
const FIELD_SEPARATOR = "";

/** Stable, collision-resistant key for a token (identity for counting). */
export function movementTokenKey(token: MovementToken): string {
  return [token.kind, token.target ?? "", token.direction ?? ""].join(FIELD_SEPARATOR);
}

function buildContextKey(keys: string[]): string {
  return keys.join(CONTEXT_SEPARATOR);
}

function canonicalizeToken(token: MovementToken): MovementToken {
  const canonical: MovementToken = { kind: token.kind };
  if (token.target !== undefined) {
    canonical.target = token.target;
  }
  if (token.direction !== undefined) {
    canonical.direction = token.direction;
  }
  return canonical;
}

/**
 * Order-k Markov movement model with deterministic decoding and Katz backoff.
 *
 * - Repeat: seeding with a recorded prefix reproduces the dominant recorded path
 *   because decoding is argmax over transition counts.
 * - Generalize: a novel context that shares a suffix with training data still
 *   yields a prediction — the model backs off to the longest seen suffix, so
 *   related-but-unseen prefixes route to the learned continuation.
 */
export class NgramMovementModel implements MovementModelBackend {
  readonly id = "ngram-movement-model";
  readonly order: number;

  constructor(order = 2) {
    if (!Number.isInteger(order) || order < 1) {
      throw new Error(`NgramMovementModel order must be a positive integer, received ${order}`);
    }
    this.order = order;
  }

  train(sequences: MovementSequence[]): MovementModelState {
    const tokens: Record<string, MovementToken> = {};
    const transitions: Record<string, Record<string, number>> = {};
    const starts: Record<string, number> = {};
    const vocabulary = new Set<string>();
    let tokenCount = 0;
    let sequenceCount = 0;

    for (const sequence of sequences) {
      if (sequence.tokens.length === 0) {
        continue;
      }
      sequenceCount += 1;
      const keys = sequence.tokens.map((token) => {
        const key = movementTokenKey(token);
        tokens[key] = canonicalizeToken(token);
        vocabulary.add(key);
        return key;
      });

      const firstKey = keys[0]!;
      starts[firstKey] = (starts[firstKey] ?? 0) + 1;

      for (let i = 0; i < keys.length; i += 1) {
        tokenCount += 1;
        const nextKey = keys[i]!;
        const maxContext = Math.min(this.order, i);
        for (let length = 0; length <= maxContext; length += 1) {
          const contextKeys = keys.slice(i - length, i);
          const contextKey = buildContextKey(contextKeys);
          const bucket = (transitions[contextKey] ??= {});
          bucket[nextKey] = (bucket[nextKey] ?? 0) + 1;
        }
      }
    }

    return {
      version: 1,
      order: this.order,
      vocabulary: [...vocabulary].sort(),
      tokens,
      transitions,
      starts,
      sequenceCount,
      tokenCount,
    };
  }

  predictNext(state: MovementModelState, context: MovementToken[]): MovementPrediction | undefined {
    const contextKeys = context.map(movementTokenKey);
    const maxLength = Math.min(state.order, contextKeys.length);
    for (let length = maxLength; length >= 0; length -= 1) {
      const contextKey = buildContextKey(contextKeys.slice(contextKeys.length - length));
      const bucket = state.transitions[contextKey];
      if (!bucket) {
        continue;
      }
      const best = argmaxTransition(bucket);
      if (!best) {
        continue;
      }
      const token = state.tokens[best.key];
      if (!token) {
        continue;
      }
      return {
        token: canonicalizeToken(token),
        probability: best.count / best.total,
        contextOrder: length,
        observationCount: best.count,
      };
    }
    return undefined;
  }

  generate(state: MovementModelState, options: GenerateOptions = {}): MovementToken[] {
    const maxLength = options.maxLength ?? 64;
    const stopKey = options.stopToken ? movementTokenKey(options.stopToken) : undefined;
    const output: MovementToken[] = (options.seed ?? this.defaultSeed(state)).map(canonicalizeToken);

    while (output.length < maxLength) {
      const context = output.slice(Math.max(0, output.length - state.order));
      const prediction = this.predictNext(state, context);
      if (!prediction) {
        break;
      }
      output.push(prediction.token);
      if (stopKey && movementTokenKey(prediction.token) === stopKey) {
        break;
      }
    }
    return output;
  }

  private defaultSeed(state: MovementModelState): MovementToken[] {
    const best = argmaxTransition(state.starts);
    if (best) {
      const token = state.tokens[best.key];
      if (token) {
        return [canonicalizeToken(token)];
      }
    }
    return [];
  }
}

/** Deterministic argmax: highest count, ties broken by lexicographic key. */
function argmaxTransition(
  bucket: Record<string, number>,
): { key: string; count: number; total: number } | undefined {
  let bestKey: string | undefined;
  let bestCount = -1;
  let total = 0;
  for (const key of Object.keys(bucket).sort()) {
    const count = bucket[key]!;
    total += count;
    if (count > bestCount) {
      bestCount = count;
      bestKey = key;
    }
  }
  if (bestKey === undefined) {
    return undefined;
  }
  return { key: bestKey, count: bestCount, total };
}

/** Extract a movement token from a recorded trajectory action. */
function tokenFromActionMetadata(
  tool: string,
  metadata: Record<string, unknown> | undefined,
): MovementToken {
  const gesture = typeof metadata?.gesture === "string" ? metadata.gesture : undefined;
  const target = typeof metadata?.target === "string" ? metadata.target : undefined;
  const direction = typeof metadata?.direction === "string" ? metadata.direction : undefined;
  const token: MovementToken = { kind: gesture ?? tool };
  if (target !== undefined) {
    token.target = target;
  }
  if (direction !== undefined) {
    token.direction = direction;
  }
  return token;
}

/** Bridge: turn a recorded trajectory into a training sequence. */
export function tokenizeTrajectory(span: TrajectorySpan): MovementSequence {
  const tokens = [...span.actions]
    .sort((a, b) => a.ts - b.ts)
    .map((action) => tokenFromActionMetadata(action.tool, action.metadata));
  return { id: span.id, tokens };
}

/** Bridge: turn a replay manifest's action timeline into a training sequence. */
export function tokenizeReplayManifest(manifest: ReplayManifest): MovementSequence {
  const tokens = manifest.events
    .filter((event): event is Extract<typeof event, { kind: "action" }> => event.kind === "action")
    .map((event) => ({ kind: event.tool, target: event.summary }));
  return { id: manifest.sessionId, tokens };
}

export type SequenceFidelity = {
  matched: number;
  total: number;
  accuracy: number;
};

/** Position-wise exact-match fidelity of a generated vs. expected sequence. */
export function evaluateSequenceFidelity(
  predicted: MovementToken[],
  expected: MovementToken[],
): SequenceFidelity {
  const total = expected.length;
  if (total === 0) {
    return { matched: 0, total: 0, accuracy: 1 };
  }
  let matched = 0;
  const limit = Math.min(predicted.length, expected.length);
  for (let i = 0; i < limit; i += 1) {
    if (movementTokenKey(predicted[i]!) === movementTokenKey(expected[i]!)) {
      matched += 1;
    }
  }
  return { matched, total, accuracy: matched / total };
}

export type NextTokenAccuracy = {
  correct: number;
  evaluated: number;
  accuracy: number;
  backedOff: number;
};

/**
 * Generalization eval harness: measure next-token prediction accuracy on
 * held-out sequences. `backedOff` counts predictions that had to fall back to a
 * shorter context than available — the generalization signal.
 */
export function evaluateNextTokenAccuracy(
  backend: MovementModelBackend,
  state: MovementModelState,
  heldOut: MovementSequence[],
): NextTokenAccuracy {
  let correct = 0;
  let evaluated = 0;
  let backedOff = 0;
  for (const sequence of heldOut) {
    for (let i = 1; i < sequence.tokens.length; i += 1) {
      const context = sequence.tokens.slice(0, i);
      const prediction = backend.predictNext(state, context);
      evaluated += 1;
      if (!prediction) {
        continue;
      }
      if (prediction.contextOrder < Math.min(state.order, context.length)) {
        backedOff += 1;
      }
      if (movementTokenKey(prediction.token) === movementTokenKey(sequence.tokens[i]!)) {
        correct += 1;
      }
    }
  }
  return {
    correct,
    evaluated,
    accuracy: evaluated === 0 ? 1 : correct / evaluated,
    backedOff,
  };
}
