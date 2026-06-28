/**
 * Local-movement learning: pluggable model backend + a deterministic,
 * in-process reference implementation.
 *
 * Standing objective #2 (parts c & d): post-train a local model on a recorded
 * movement dataset to (c) *repeat* recorded movements and (d) *generalize* to
 * new-but-related movements. The real on-device training (mlx/axolotl) is
 * emitted by {@link ./runner.ts} as a launch script and only runs on the user's
 * machine. This module provides the missing piece that *can* run and be
 * validated in the cloud: a backend interface plus a deterministic
 * variable-order Markov ("n-gram with back-off") backend that genuinely learns
 * transition statistics from movement-token sequences, reproduces recorded
 * movements exactly, and generalizes to novel prefixes via context back-off.
 *
 * Everything here is dependency-free and deterministic (no Math.random / no
 * Date.now) so cloud/CI tests are reproducible.
 */

import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/** A single movement step, encoded as a stable string token. */
export type MovementToken = string;

/** Sentinel tokens framing every training sequence. Never emitted as output. */
export const MOVEMENT_START_TOKEN = "<start>";
export const MOVEMENT_END_TOKEN = "<end>";

/** Separator used to build a context key from a window of tokens. */
const CONTEXT_SEPARATOR = "";

export const DEFAULT_MOVEMENT_BACKEND_ID = "markov-backoff";
export const DEFAULT_MOVEMENT_MODEL_ORDER = 3;

export type MovementSequence = {
  /** Source identifier (trajectory id, session id, …) for traceability. */
  id: string;
  tokens: MovementToken[];
};

export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

export type MovementPrediction = {
  token: MovementToken;
  /** Conditional probability under the (backed-off) context. */
  probability: number;
  /** Observed count supporting this prediction. */
  count: number;
  /** Context order actually used after back-off (0 = unconditional unigram). */
  order: number;
};

/**
 * A fully serializable (plain-JSON) trained model. `transitions` is indexed by
 * order → context-key → next-token → count.
 */
export type MovementModelArtifact = {
  version: 1;
  backend: string;
  maxOrder: number;
  vocabulary: MovementToken[];
  sequenceCount: number;
  tokenCount: number;
  transitions: Record<string, Record<string, Record<MovementToken, number>>>;
  trainedAt?: string;
};

export type TrainMovementModelOptions = {
  maxOrder?: number;
  /** Injectable timestamp (cloud determinism — module never calls Date.now). */
  trainedAt?: string;
};

export type PredictMovementOptions = {
  /** Cap on returned predictions (ranked best-first). */
  topK?: number;
  /** Include the end-of-sequence sentinel among predictions. Default false. */
  includeEnd?: boolean;
};

export type GenerateMovementOptions = {
  /** Already-performed prefix to continue from. Default: from sequence start. */
  seed?: MovementToken[];
  /** Maximum number of tokens to emit before stopping. Default 256. */
  maxLength?: number;
};

/**
 * Pluggable backend seam. A real on-device small model would implement this
 * same interface (train/predict/generate over an opaque, JSON-able artifact);
 * the Markov backend below is the deterministic reference used in tests.
 */
export interface LocalMovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: TrainMovementModelOptions): MovementModelArtifact;
  predict(
    artifact: MovementModelArtifact,
    context: MovementToken[],
    options?: PredictMovementOptions,
  ): MovementPrediction[];
  generate(artifact: MovementModelArtifact, options?: GenerateMovementOptions): MovementToken[];
}

// ---------------------------------------------------------------------------
// Tokenization: bridge capture artifacts (replays/trajectories) → token seqs.
// ---------------------------------------------------------------------------

export type MovementTokenizerOptions = {
  /** Include `transcript` timeline events as `msg:<role>` tokens. Default false. */
  includeTranscript?: boolean;
  /** Include `observation` events as `observation:<source>` tokens. Default true. */
  includeObservations?: boolean;
};

/** Map one replay timeline event to a movement token, or `undefined` to drop. */
export function tokenizeReplayEvent(
  event: ReplayTimelineEvent,
  options: MovementTokenizerOptions = {},
): MovementToken | undefined {
  const includeObservations = options.includeObservations ?? true;
  switch (event.kind) {
    case "action":
      return `action:${event.tool}`;
    case "observation":
      return includeObservations ? `observation:${event.source}` : undefined;
    case "transcript":
      return options.includeTranscript ? `msg:${event.role}` : undefined;
  }
}

export function tokenizeReplayEvents(
  events: readonly ReplayTimelineEvent[],
  options?: MovementTokenizerOptions,
): MovementToken[] {
  const tokens: MovementToken[] = [];
  for (const event of events) {
    const token = tokenizeReplayEvent(event, options);
    if (token !== undefined) {
      tokens.push(token);
    }
  }
  return tokens;
}

/** Build a dataset from replay manifests (one sequence per manifest). */
export function movementDatasetFromReplays(
  replays: ReadonlyArray<Pick<ReplayManifest, "sessionId" | "events"> & { id?: string }>,
  options?: MovementTokenizerOptions,
): MovementDataset {
  return {
    version: 1,
    sequences: replays.map((replay, index) => ({
      id: replay.id ?? replay.sessionId ?? `replay-${index}`,
      tokens: tokenizeReplayEvents(replay.events, options),
    })),
  };
}

/** Build a dataset directly from trajectory spans (one sequence per span). */
export function movementDatasetFromTrajectories(
  trajectories: readonly TrajectorySpan[],
  options: MovementTokenizerOptions = {},
): MovementDataset {
  const includeObservations = options.includeObservations ?? true;
  return {
    version: 1,
    sequences: trajectories.map((trajectory) => {
      const merged = [
        ...trajectory.observations.map((observation) => ({ ts: observation.ts, token: `observation:${observation.source}` })),
        ...trajectory.actions.map((action) => ({ ts: action.ts, token: `action:${action.tool}` })),
      ]
        .filter((entry) => includeObservations || !entry.token.startsWith("observation:"))
        .sort((a, b) => a.ts - b.ts);
      return { id: trajectory.id, tokens: merged.map((entry) => entry.token) };
    }),
  };
}

// ---------------------------------------------------------------------------
// Markov back-off backend (deterministic reference implementation).
// ---------------------------------------------------------------------------

export class MarkovMovementBackend implements LocalMovementModelBackend {
  readonly id = DEFAULT_MOVEMENT_BACKEND_ID;

  train(dataset: MovementDataset, options: TrainMovementModelOptions = {}): MovementModelArtifact {
    const maxOrder = Math.max(0, options.maxOrder ?? DEFAULT_MOVEMENT_MODEL_ORDER);
    const transitions: MovementModelArtifact["transitions"] = {};
    const vocabulary = new Set<MovementToken>();
    let tokenCount = 0;

    for (const sequence of dataset.sequences) {
      const padded = [MOVEMENT_START_TOKEN, ...sequence.tokens, MOVEMENT_END_TOKEN];
      tokenCount += sequence.tokens.length;
      for (const token of sequence.tokens) {
        vocabulary.add(token);
      }
      for (let target = 1; target < padded.length; target += 1) {
        const nextToken = padded[target]!;
        const maxContext = Math.min(maxOrder, target);
        for (let order = 0; order <= maxContext; order += 1) {
          const contextKey = padded.slice(target - order, target).join(CONTEXT_SEPARATOR);
          const orderKey = String(order);
          const byContext = (transitions[orderKey] ??= {});
          const counts = (byContext[contextKey] ??= {});
          counts[nextToken] = (counts[nextToken] ?? 0) + 1;
        }
      }
    }

    return {
      version: 1,
      backend: this.id,
      maxOrder,
      vocabulary: [...vocabulary].sort(),
      sequenceCount: dataset.sequences.length,
      tokenCount,
      transitions,
      ...(options.trainedAt ? { trainedAt: options.trainedAt } : {}),
    };
  }

  predict(
    artifact: MovementModelArtifact,
    context: MovementToken[],
    options: PredictMovementOptions = {},
  ): MovementPrediction[] {
    const includeEnd = options.includeEnd ?? false;
    const padded = context.length === 0 ? [MOVEMENT_START_TOKEN] : context;
    const maxContext = Math.min(artifact.maxOrder, padded.length);

    for (let order = maxContext; order >= 0; order -= 1) {
      const orderKey = String(order);
      const byContext = artifact.transitions[orderKey];
      if (!byContext) {
        continue;
      }
      const contextKey = padded.slice(padded.length - order).join(CONTEXT_SEPARATOR);
      const counts = byContext[contextKey];
      if (!counts) {
        continue;
      }
      const entries = Object.entries(counts).filter(
        ([token]) => includeEnd || token !== MOVEMENT_END_TOKEN,
      );
      if (entries.length === 0) {
        continue;
      }
      const total = entries.reduce((sum, [, count]) => sum + count, 0);
      const ranked = entries
        .map(([token, count]) => ({ token, count, probability: count / total, order }))
        // Deterministic: highest count first, ties broken lexicographically.
        .sort((a, b) => (b.count !== a.count ? b.count - a.count : a.token < b.token ? -1 : 1));
      return options.topK === undefined ? ranked : ranked.slice(0, Math.max(0, options.topK));
    }

    return [];
  }

  generate(artifact: MovementModelArtifact, options: GenerateMovementOptions = {}): MovementToken[] {
    const maxLength = options.maxLength ?? 256;
    const seed = options.seed ?? [];
    const generated: MovementToken[] = [];

    while (generated.length < maxLength) {
      const context = [MOVEMENT_START_TOKEN, ...seed, ...generated];
      const [best] = this.predict(artifact, context, { topK: 1, includeEnd: true });
      if (!best || best.token === MOVEMENT_END_TOKEN) {
        break;
      }
      generated.push(best.token);
    }

    return generated;
  }
}

// ---------------------------------------------------------------------------
// Pluggable backend registry.
// ---------------------------------------------------------------------------

export type MovementBackendFactory = () => LocalMovementModelBackend;

const backendRegistry = new Map<string, MovementBackendFactory>();

export function registerMovementModelBackend(id: string, factory: MovementBackendFactory): void {
  backendRegistry.set(id, factory);
}

export function listMovementModelBackends(): string[] {
  return [...backendRegistry.keys()].sort();
}

export function createMovementModelBackend(
  id: string = DEFAULT_MOVEMENT_BACKEND_ID,
): LocalMovementModelBackend {
  const factory = backendRegistry.get(id);
  if (!factory) {
    throw new Error(
      `Unknown movement-model backend "${id}". Registered: ${listMovementModelBackends().join(", ") || "(none)"}`,
    );
  }
  return factory();
}

registerMovementModelBackend(DEFAULT_MOVEMENT_BACKEND_ID, () => new MarkovMovementBackend());

// ---------------------------------------------------------------------------
// Generalization eval harness: replay fidelity on held-out sequences.
// ---------------------------------------------------------------------------

export type MovementEvalResult = {
  sequences: number;
  predictions: number;
  correct: number;
  /** Top-1 next-token accuracy over all held-out positions. */
  accuracy: number;
  /** Fraction of held-out sequences reproduced exactly by free-running generate. */
  exactSequenceMatch: number;
};

/**
 * Measure how well a trained model predicts/reproduces held-out (but related)
 * movement sequences — the generalization signal for objective #2(d).
 *
 * Two metrics: teacher-forced top-1 next-token accuracy, and the fraction of
 * sequences reproduced verbatim when generating freely from start.
 */
export function evaluateMovementModel(
  backend: LocalMovementModelBackend,
  artifact: MovementModelArtifact,
  heldOut: readonly MovementSequence[],
): MovementEvalResult {
  let predictions = 0;
  let correct = 0;
  let exact = 0;

  for (const sequence of heldOut) {
    const target = [...sequence.tokens, MOVEMENT_END_TOKEN];
    for (let i = 0; i < target.length; i += 1) {
      const context = sequence.tokens.slice(0, i);
      const [best] = backend.predict(artifact, context, { topK: 1, includeEnd: true });
      predictions += 1;
      if (best && best.token === target[i]) {
        correct += 1;
      }
    }
    const produced = backend.generate(artifact, { seed: [] }).join(CONTEXT_SEPARATOR);
    if (produced === sequence.tokens.join(CONTEXT_SEPARATOR)) {
      exact += 1;
    }
  }

  return {
    sequences: heldOut.length,
    predictions,
    correct,
    accuracy: predictions === 0 ? 0 : correct / predictions,
    exactSequenceMatch: heldOut.length === 0 ? 0 : exact / heldOut.length,
  };
}
