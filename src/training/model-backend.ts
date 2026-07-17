import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";

/**
 * Movement-learning model backend contract.
 *
 * The local-movement learning subsystem records user movements/actions into
 * {@link ReplayManifest}s, then post-trains a small local model to (c) repeat
 * the recorded movements and (d) generalize to new-but-related movements. The
 * real on-device training executes when the user runs bee-agent locally
 * (see `runner.ts`, which emits mlx/axolotl launch scripts). This module
 * provides the *pluggable seam* those backends implement plus a deterministic
 * in-process backend (`markov-backend.ts`) so the pipeline is testable in the
 * cloud with synthetic event streams and no OS/GPU access.
 */

/** A stable, comparable token describing a single movement/action. */
export type MovementToken = string;

/** One ordered sequence of movement tokens extracted from a trajectory. */
export type MovementSample = {
  /** The trajectory this sequence was derived from, when known. */
  trajectoryId?: string;
  /** Ordered movement tokens, earliest first. */
  tokens: MovementToken[];
};

/** A training dataset: many movement sequences plus the observed vocabulary. */
export type MovementDataset = {
  samples: MovementSample[];
  /** Sorted, de-duplicated set of every token seen across all samples. */
  vocabulary: MovementToken[];
};

/** A single candidate continuation with its estimated probability. */
export type MovementCandidate = {
  token: MovementToken;
  probability: number;
};

/** The model's prediction for the next movement given a context. */
export type MovementPrediction = {
  /** Best next token, or `null` when the model has no basis to predict. */
  token: MovementToken | null;
  /** Probability mass assigned to {@link token} (0 when `token` is null). */
  confidence: number;
  /**
   * Length of the context suffix the prediction actually matched. A higher
   * order means a more specific (less generalized) match; `0` means the model
   * fell back to its unconditional prior.
   */
  matchedOrder: number;
  /** All candidates, most probable first (deterministically tie-broken). */
  candidates: MovementCandidate[];
};

/** Opaque, JSON-serializable model weights produced by a backend. */
export type SerializedMovementModel = {
  backendId: string;
  version: 1;
  [key: string]: unknown;
};

/** Options accepted by every backend's `train`. */
export type MovementTrainOptions = {
  /** Maximum context length the model may condition on. */
  maxOrder?: number;
};

/** A trained model that can predict and generate movement continuations. */
export interface TrainedMovementModel {
  readonly backendId: string;
  /** Predict the next movement given prior movement tokens (context). */
  predict(context: MovementToken[]): MovementPrediction;
  /**
   * Greedily generate up to `steps` continuation tokens starting from `seed`.
   * Generation stops early if the model can no longer predict a next token.
   */
  generate(seed: MovementToken[], steps: number): MovementToken[];
  /** Export weights so the model can be persisted and later restored. */
  serialize(): SerializedMovementModel;
}

/** A pluggable movement-model training backend. */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainOptions): Promise<TrainedMovementModel>;
  /** Rebuild a trained model from previously serialized weights. */
  restore(serialized: SerializedMovementModel): TrainedMovementModel;
}

/**
 * Tokenize a single replay action event into a stable movement token. Only
 * `action` events represent movements; observations/transcript are context.
 */
export function movementActionToken(event: Extract<ReplayTimelineEvent, { kind: "action" }>): MovementToken {
  return `${event.tool}:${normalizeSummary(event.summary)}`;
}

/**
 * Extract the ordered movement-token sequence from a single replay manifest.
 * Events are assumed already time-sorted by `buildReplayManifest`.
 */
export function extractMovementSequence(replay: ReplayManifest): MovementToken[] {
  return replay.events
    .filter((event): event is Extract<ReplayTimelineEvent, { kind: "action" }> => event.kind === "action")
    .map(movementActionToken);
}

/**
 * Build a {@link MovementDataset} from recorded replay manifests. Sequences
 * with no movement actions are dropped; the vocabulary is sorted for stable,
 * deterministic downstream training.
 */
export function buildMovementDataset(replays: ReplayManifest[]): MovementDataset {
  const vocabulary = new Set<MovementToken>();
  const samples: MovementSample[] = [];

  for (const replay of replays) {
    const tokens = extractMovementSequence(replay);
    if (tokens.length === 0) {
      continue;
    }
    for (const token of tokens) {
      vocabulary.add(token);
    }
    samples.push({
      ...(replay.trajectoryIds[0] ? { trajectoryId: replay.trajectoryIds[0] } : {}),
      tokens,
    });
  }

  return {
    samples,
    vocabulary: [...vocabulary].sort(),
  };
}

function normalizeSummary(summary: string): string {
  return summary.trim().replace(/\s+/g, " ").toLowerCase();
}
