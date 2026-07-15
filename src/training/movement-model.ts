import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Local-movement learning: model backend contract.
 *
 * A recorded session is treated as an ordered stream of discrete *action
 * tokens* (mouse/keyboard/window/tool events reduced to a stable key). A
 * movement model learns to predict the next token given the preceding context,
 * which lets it (a) *repeat* a recorded movement exactly and (b) *generalize*
 * to new-but-related movements by reusing shorter shared context.
 *
 * The backend is pluggable: the deterministic {@link MovementModelBackend}
 * implemented in `markov-backend.ts` runs entirely in-process (so cloud/CI
 * tests pass), while the same interface is the documented seam for a real
 * on-device small model that the user trains when running bee-agent locally.
 */

/** A single tokenized movement step. */
export type MovementToken = string;

/** An ordered movement sequence derived from one trajectory / session. */
export type MovementSequence = {
  /** Trajectory or session id this sequence came from. */
  sourceId: string;
  /** Ordered action tokens. */
  tokens: MovementToken[];
  /** Optional terminal reward for reward-aware backends. */
  reward?: number;
};

/** A dataset of movement sequences ready for training. */
export type MovementDataset = {
  version: 1;
  createdAt?: string;
  sequences: MovementSequence[];
};

/** Reduces a trajectory action to a single, stable movement token. */
export type MovementTokenizer = (action: {
  tool: string;
  summary: string;
  metadata?: Record<string, unknown>;
}) => MovementToken;

/** Default tokenizer: use the action's tool name as the token. */
export const defaultMovementTokenizer: MovementTokenizer = (action) => action.tool;

/** Training knobs shared across backends (backend may ignore unknown fields). */
export type MovementTrainingConfig = {
  /** Maximum context order (n-gram length) to learn. */
  maxOrder?: number;
  /** Additive smoothing applied to observed counts. */
  smoothing?: number;
};

/** Summary metrics attached to a trained model. */
export type MovementTrainingMetadata = {
  sequenceCount: number;
  tokenCount: number;
  vocabularySize: number;
  [key: string]: unknown;
};

/**
 * A trained model artifact. It is plain JSON, so it round-trips through
 * `JSON.stringify`/`parse` and can be persisted with the existing atomic-write
 * helpers without a bespoke serializer.
 */
export type MovementModelArtifact = {
  version: 1;
  backend: string;
  createdAt?: string;
  /** Backend-specific serialized parameters. */
  parameters: unknown;
  metadata: MovementTrainingMetadata;
};

/** The result of a single next-token prediction. */
export type MovementPrediction = {
  /** Predicted next token, or `null` when the model has no basis to predict. */
  token: MovementToken | null;
  /** Confidence in [0,1] assigned to the predicted token. */
  confidence: number;
  /** Context order actually used (backoff depth); -1 when nothing matched. */
  contextOrder: number;
  /** Ranked candidate distribution, highest probability first. */
  distribution: Array<{ token: MovementToken; probability: number }>;
};

/** Pluggable local-movement model backend. */
export interface MovementModelBackend {
  /** Stable backend identifier, stamped onto produced artifacts. */
  readonly id: string;
  /** Post-train a model on a movement dataset. */
  train(dataset: MovementDataset, config?: MovementTrainingConfig): MovementModelArtifact;
  /** Predict the next token given a context of preceding tokens. */
  predict(model: MovementModelArtifact, context: MovementToken[]): MovementPrediction;
  /** Deterministically roll out a sequence from a seed context. */
  generate(model: MovementModelArtifact, seed: MovementToken[], maxSteps: number): MovementToken[];
}

/**
 * Build a training dataset from captured trajectories. Actions are ordered by
 * timestamp and tokenized; redacted (reviewed) actions take precedence over raw
 * actions so exported datasets never leak un-reviewed capture.
 */
export function buildMovementDatasetFromTrajectories(
  trajectories: TrajectorySpan[],
  options?: { tokenizer?: MovementTokenizer; requireApproved?: boolean; createdAt?: string },
): MovementDataset {
  const tokenizer = options?.tokenizer ?? defaultMovementTokenizer;
  const requireApproved = options?.requireApproved ?? false;
  const sequences: MovementSequence[] = [];

  for (const trajectory of trajectories) {
    if (requireApproved && trajectory.review?.status !== "approved") {
      continue;
    }
    const actions = trajectory.review?.redactedActions
      ? trajectory.review.redactedActions.map((action) => ({ ts: action.ts, tool: action.tool, summary: action.summary }))
      : trajectory.actions.map((action) => ({
          ts: action.ts,
          tool: action.tool,
          summary: action.summary,
          metadata: action.metadata,
        }));

    const tokens = [...actions]
      .sort((a, b) => a.ts - b.ts)
      .map((action) => tokenizer(action));

    if (tokens.length === 0) {
      continue;
    }
    sequences.push({
      sourceId: trajectory.id,
      tokens,
      ...(trajectory.outcome?.reward !== undefined ? { reward: trajectory.outcome.reward } : {}),
    });
  }

  return {
    version: 1,
    ...(options?.createdAt ? { createdAt: options.createdAt } : {}),
    sequences,
  };
}

/** Extract a movement sequence from a replay manifest's action events. */
export function buildMovementSequenceFromReplay(replay: ReplayManifest): MovementSequence {
  const tokens = replay.events
    .filter((event): event is Extract<ReplayTimelineEvent, { kind: "action" }> => event.kind === "action")
    .map((event) => event.tool);
  return { sourceId: replay.sessionId, tokens };
}

/** Aggregate accuracy of a model measured over held-out sequences. */
export type MovementEvalResult = {
  totalPredictions: number;
  correct: number;
  accuracy: number;
  averageConfidence: number;
  /** How often each backoff order was used to make a prediction. */
  backoffOrderHistogram: Record<number, number>;
};

/**
 * Generalization eval harness: for each sequence, replay it token-by-token and
 * measure how often the model's next-token prediction matches the recorded
 * next token. Run on held-out sequences to measure generalization; run on the
 * training sequences to confirm exact-repeat fidelity.
 */
export function evaluateMovementModel(
  backend: MovementModelBackend,
  model: MovementModelArtifact,
  sequences: MovementSequence[],
): MovementEvalResult {
  let total = 0;
  let correct = 0;
  let confidenceSum = 0;
  const backoffOrderHistogram: Record<number, number> = {};

  for (const sequence of sequences) {
    for (let index = 1; index < sequence.tokens.length; index += 1) {
      const context = sequence.tokens.slice(0, index);
      const prediction = backend.predict(model, context);
      total += 1;
      confidenceSum += prediction.confidence;
      backoffOrderHistogram[prediction.contextOrder] =
        (backoffOrderHistogram[prediction.contextOrder] ?? 0) + 1;
      if (prediction.token === sequence.tokens[index]) {
        correct += 1;
      }
    }
  }

  return {
    totalPredictions: total,
    correct,
    accuracy: total === 0 ? 0 : correct / total,
    averageConfidence: total === 0 ? 0 : confidenceSum / total,
    backoffOrderHistogram,
  };
}
