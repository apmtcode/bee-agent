import type { ReplayManifest } from "../capture/replay.js";
import type { ReviewedTrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

/**
 * Pluggable local-model backend for the movement-learning subsystem
 * (standing objective #2d: post-train a local model on recorded movements so it
 * can repeat them and generalize to new-but-related movements).
 *
 * The real on-device path trains a small model via the plans produced by
 * {@link LocalAppleSiliconTrainingRunner}. That cannot run in the cloud, so this
 * module defines a backend-agnostic seam plus a fully deterministic reference
 * backend ({@link MarkovMovementBackend}) that trains and infers in-process.
 * The deterministic backend makes the whole capture -> dataset -> train ->
 * replay/generalize round-trip testable without touching a real machine, and
 * documents the interface a real on-device backend must satisfy.
 */

/** A single discrete movement step (e.g. a tool/action name). */
export type MovementToken = string;

/** One recorded movement sequence extracted from a trajectory or replay. */
export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

/** A structured, replayable dataset of movement sequences. */
export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

export type MovementTrainingOptions = {
  /**
   * Context length (how many preceding tokens condition the next-token
   * prediction). 1 = bigram, 2 = trigram, etc. Defaults to 1.
   */
  order?: number;
};

/** Serializable, backend-tagged trained model. Persist as JSON. */
export type MovementModelArtifact = {
  backendId: string;
  version: 1;
  order: number;
  /** Sorted set of tokens observed during training. */
  vocabulary: MovementToken[];
  /** Number of sequences the model was trained on. */
  sequenceCount: number;
  /** Backend-private parameters. Opaque to callers. */
  params: unknown;
};

export type MovementContext = {
  /** Tokens emitted so far; the backend conditions on the most recent ones. */
  history: MovementToken[];
};

export type MovementCandidate = {
  token: MovementToken;
  probability: number;
};

export type MovementPrediction = {
  /** Most likely next token, or undefined if the model has no opinion. */
  token: MovementToken | undefined;
  /** Probability of {@link token} (0..1). */
  confidence: number;
  /** Full next-token distribution, highest probability first. */
  distribution: MovementCandidate[];
};

/**
 * The seam every movement-model backend implements. A real on-device backend
 * (MLX, llama.cpp, ...) satisfies the same contract as the in-process mock, so
 * callers and tests are backend-agnostic.
 */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainingOptions): Promise<MovementModelArtifact>;
  predict(artifact: MovementModelArtifact, context: MovementContext): MovementPrediction;
}

/** Sentinel used to pad the start of a sequence up to the context order. */
const START_TOKEN = "<start>";
const CONTEXT_SEPARATOR = "";

type MarkovModelParams = {
  /** contextKey -> (token -> observation count). */
  transitions: Record<string, Record<MovementToken, number>>;
  /** token -> total observation count (the unigram fallback). */
  unigram: Record<MovementToken, number>;
};

function contextKey(history: MovementToken[], order: number): string {
  const window: MovementToken[] = [];
  for (let i = 0; i < order; i += 1) {
    const token = history[history.length - order + i];
    window.push(token ?? START_TOKEN);
  }
  return window.join(CONTEXT_SEPARATOR);
}

function distributionFromCounts(counts: Record<MovementToken, number>): MovementCandidate[] {
  const total = Object.values(counts).reduce((sum, count) => sum + count, 0);
  if (total === 0) {
    return [];
  }
  return Object.entries(counts)
    .map(([token, count]) => ({ token, probability: count / total }))
    .sort((a, b) => {
      if (b.probability !== a.probability) {
        return b.probability - a.probability;
      }
      // Deterministic tie-break so predictions never depend on insertion order.
      return a.token < b.token ? -1 : a.token > b.token ? 1 : 0;
    });
}

/**
 * A deterministic order-k Markov (n-gram) movement model. It exactly reproduces
 * recorded sequences it was trained on and generalizes to unseen prefixes by
 * falling back through shorter contexts down to the global unigram. No RNG, no
 * wall-clock — identical inputs always yield identical outputs, so it is safe to
 * assert on in tests.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov";

  async train(dataset: MovementDataset, options: MovementTrainingOptions = {}): Promise<MovementModelArtifact> {
    const order = Math.max(1, Math.floor(options.order ?? 1));
    const transitions: Record<string, Record<MovementToken, number>> = {};
    const unigram: Record<MovementToken, number> = {};
    const vocabulary = new Set<MovementToken>();

    for (const sequence of dataset.sequences) {
      const padded = [...Array<MovementToken>(order).fill(START_TOKEN), ...sequence.tokens];
      for (let i = order; i < padded.length; i += 1) {
        const token = padded[i]!;
        vocabulary.add(token);
        unigram[token] = (unigram[token] ?? 0) + 1;
        // Record every context length from `order` down to 1 for graceful backoff.
        for (let contextLength = order; contextLength >= 1; contextLength -= 1) {
          const key = `${contextLength}${CONTEXT_SEPARATOR}${contextKey(padded.slice(0, i), contextLength)}`;
          const bucket = (transitions[key] ??= {});
          bucket[token] = (bucket[token] ?? 0) + 1;
        }
      }
    }

    const params: MarkovModelParams = { transitions, unigram };
    return {
      backendId: this.id,
      version: 1,
      order,
      vocabulary: [...vocabulary].sort(),
      sequenceCount: dataset.sequences.length,
      params,
    };
  }

  predict(artifact: MovementModelArtifact, context: MovementContext): MovementPrediction {
    if (artifact.backendId !== this.id) {
      throw new Error(`MarkovMovementBackend cannot read artifact from backend "${artifact.backendId}"`);
    }
    const params = artifact.params as MarkovModelParams;

    // Back off from the longest available context to the unigram fallback.
    for (let contextLength = artifact.order; contextLength >= 1; contextLength -= 1) {
      const key = `${contextLength}${CONTEXT_SEPARATOR}${contextKey(context.history, contextLength)}`;
      const counts = params.transitions[key];
      if (counts && Object.keys(counts).length > 0) {
        return finalizePrediction(distributionFromCounts(counts));
      }
    }

    return finalizePrediction(distributionFromCounts(params.unigram));
  }
}

function finalizePrediction(distribution: MovementCandidate[]): MovementPrediction {
  const top = distribution[0];
  return {
    token: top?.token,
    confidence: top?.probability ?? 0,
    distribution,
  };
}

export type MovementTokenizer = (action: ReviewedTrajectoryAction) => MovementToken;

const defaultTokenizer: MovementTokenizer = (action) => action.tool;

/**
 * Build a movement dataset from replay manifests. Each manifest becomes one
 * sequence of its ordered `action` events' tools.
 */
export function datasetFromReplayManifests(manifests: ReplayManifest[]): MovementDataset {
  const sequences = manifests.map<MovementSequence>((manifest) => ({
    id: manifest.sessionId,
    tokens: manifest.events.flatMap((event) => (event.kind === "action" ? [event.tool] : [])),
  }));
  return { version: 1, sequences };
}

/**
 * Build a movement dataset from trajectory spans. Prefers the reviewed
 * (redacted) actions when present; otherwise uses raw actions. Each span becomes
 * one sequence.
 */
export function datasetFromTrajectories(
  trajectories: TrajectorySpan[],
  tokenizer: MovementTokenizer = defaultTokenizer,
): MovementDataset {
  const sequences = trajectories.map<MovementSequence>((trajectory) => {
    const reviewed = trajectory.review?.redactedActions;
    const actions: ReviewedTrajectoryAction[] = reviewed
      ? reviewed
      : trajectory.actions.map((action) => ({ ts: action.ts, tool: action.tool, summary: action.summary }));
    const ordered = [...actions].sort((a, b) => a.ts - b.ts);
    return { id: trajectory.id, tokens: ordered.map((action) => tokenizer(action)) };
  });
  return { version: 1, sequences };
}

export type MovementRolloutParams = {
  /** Prefix tokens to seed the rollout with (also emitted in the output). */
  seed?: MovementToken[];
  /** Maximum number of tokens to generate (including the seed). */
  maxSteps: number;
  /** Stop generating once this token is produced (it is still emitted). */
  stopToken?: MovementToken;
};

/**
 * Greedily roll out a movement sequence from a trained model. Deterministic:
 * always follows the argmax next token. Used to *repeat* a recorded movement
 * (seed with its opening steps) or to *generalize* to a new-but-related movement
 * (seed with a related prefix). Stops when the model has no next token, the
 * stop token is emitted, or maxSteps is reached.
 */
export function rolloutMovements(
  backend: MovementModelBackend,
  artifact: MovementModelArtifact,
  params: MovementRolloutParams,
): MovementToken[] {
  const history: MovementToken[] = [...(params.seed ?? [])];
  if (history.length >= params.maxSteps) {
    return history.slice(0, params.maxSteps);
  }
  if (params.stopToken !== undefined && history.at(-1) === params.stopToken) {
    return history;
  }
  while (history.length < params.maxSteps) {
    const prediction = backend.predict(artifact, { history });
    if (prediction.token === undefined) {
      break;
    }
    history.push(prediction.token);
    if (params.stopToken !== undefined && prediction.token === params.stopToken) {
      break;
    }
  }
  return history;
}

export type MovementFidelityReport = {
  /** Total next-token predictions scored. */
  total: number;
  /** How many matched the recorded next token. */
  correct: number;
  /** correct / total (0 when total is 0). */
  accuracy: number;
  perSequence: Array<{ id: string; total: number; correct: number; accuracy: number }>;
};

/**
 * Measure next-token prediction fidelity of a trained model against a set of
 * reference sequences. Scoring the *training* sequences measures memorization
 * (how faithfully recorded movements are repeated); scoring *held-out but
 * related* sequences measures generalization. This is the generalization-eval
 * harness the roadmap calls for.
 */
export function evaluateMovementFidelity(
  backend: MovementModelBackend,
  artifact: MovementModelArtifact,
  sequences: MovementSequence[],
): MovementFidelityReport {
  let total = 0;
  let correct = 0;
  const perSequence = sequences.map((sequence) => {
    let seqTotal = 0;
    let seqCorrect = 0;
    for (let i = 0; i < sequence.tokens.length; i += 1) {
      const prediction = backend.predict(artifact, { history: sequence.tokens.slice(0, i) });
      seqTotal += 1;
      if (prediction.token === sequence.tokens[i]) {
        seqCorrect += 1;
      }
    }
    total += seqTotal;
    correct += seqCorrect;
    return {
      id: sequence.id,
      total: seqTotal,
      correct: seqCorrect,
      accuracy: seqTotal === 0 ? 0 : seqCorrect / seqTotal,
    };
  });

  return {
    total,
    correct,
    accuracy: total === 0 ? 0 : correct / total,
    perSequence,
  };
}
