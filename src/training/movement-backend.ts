import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";

/**
 * Pluggable local-model backend for the movement-learning subsystem.
 *
 * bee-agent records local movements (mouse/keyboard/window actions) into
 * trajectories, which are flattened into {@link ReplayManifest} timelines. To
 * "post-train a local model that repeats the recorded movements and generalizes
 * to new but related movements" (self-evolution objective #2 c/d) we need a
 * training backend. The real on-device backend trains a small local model
 * (e.g. MLX on Apple Silicon); it cannot run in the cloud/CI. This interface is
 * the seam: any backend that can {@link MovementTrainingBackend.train} a
 * serializable model and {@link MovementTrainingBackend.predictNext} the next
 * movement satisfies it. {@link MarkovMovementBackend} is the deterministic,
 * dependency-free default used for cloud/CI validation of the whole
 * capture → dataset → train → infer → replay loop.
 */

/** A single recorded movement, normalized from an action timeline event. */
export type MovementStep = {
  /** Canonical token used for sequence modeling (see {@link canonicalMovementToken}). */
  token: string;
  /** Original tool/device that produced the movement (e.g. "mouse", "keyboard"). */
  tool: string;
  /** Human-readable description of the movement. */
  summary: string;
  /** Optional structured payload preserved for faithful replay. */
  metadata?: Record<string, unknown>;
};

/** An ordered sequence of movements captured within one trajectory. */
export type MovementSequence = MovementStep[];

/** Config passed to a backend's training call. Backends may ignore fields they don't support. */
export type MovementTrainConfig = {
  /** Highest Markov context order (number of preceding steps) to model. Default 2. */
  order?: number;
};

/**
 * A trained, serializable movement model artifact. Deliberately plain JSON so it
 * can be persisted alongside on-device artifacts (analogous to `model.gguf`).
 */
export type TrainedMovementModel = {
  version: 1;
  backendId: string;
  order: number;
  /** Distinct tokens observed in training, sorted for determinism. */
  vocabulary: string[];
  /**
   * Context → next-token counts. The key is the joined context (`"" ` for the
   * order-0/unigram distribution); the value maps a next token to its count.
   */
  transitions: Record<string, Record<string, number>>;
  /** One representative step per token, used to reconstruct movements on rollout. */
  representatives: Record<string, MovementStep>;
  trainedSequenceCount: number;
  trainedStepCount: number;
};

/** A single next-movement prediction. */
export type MovementPrediction = {
  /** The predicted token. */
  token: string;
  /** A concrete movement to perform (representative for the token). */
  step: MovementStep;
  /** Probability of this token given the (possibly backed-off) context. */
  probability: number;
  /**
   * How many context steps were actually used. Lower than the requested order
   * means the backend generalized via backoff — the exact context was unseen but
   * a related shorter context matched. `0` means a pure unigram fallback.
   */
  contextOrderUsed: number;
  /** Full ranked candidate distribution for the resolved context. */
  candidates: Array<{ token: string; probability: number }>;
};

/** The backend seam. Implementations must be deterministic given identical input. */
export interface MovementTrainingBackend {
  /** Stable identifier recorded into the trained model artifact. */
  readonly id: string;
  /** Train a model from recorded movement sequences. */
  train(sequences: MovementSequence[], config?: MovementTrainConfig): Promise<TrainedMovementModel>;
  /** Predict the next movement given a context window, or `undefined` if none. */
  predictNext(model: TrainedMovementModel, context: MovementStep[]): MovementPrediction | undefined;
}

const CONTEXT_SEPARATOR = "␟"; // unit separator, cannot appear in a normal token

/**
 * Normalize a movement into a canonical token. Tokens intentionally collapse the
 * free-text summary to its leading keyword so that structurally similar
 * movements (e.g. two different "click" targets) share a token — this is what
 * lets the model generalize across related-but-novel movements.
 */
export function canonicalMovementToken(tool: string, summary: string): string {
  const normalizedTool = tool.trim().toLowerCase().replace(/\s+/g, "-") || "unknown";
  const keyword = summary
    .trim()
    .toLowerCase()
    .split(/[\s:>/-]+/)
    .find((part) => part.length > 0);
  return keyword ? `${normalizedTool}:${keyword}` : normalizedTool;
}

/**
 * Flatten a {@link ReplayManifest} into the ordered movement sequence used for
 * training. Only `action` events are movements; observation/transcript events
 * are conditioning context and are dropped here. Events are already ts-sorted in
 * the manifest, but we re-sort defensively to keep the derivation total.
 */
export function deriveMovementSequence(replay: ReplayManifest): MovementSequence {
  return [...replay.events]
    .filter((event): event is Extract<ReplayTimelineEvent, { kind: "action" }> => event.kind === "action")
    .sort((a, b) => a.ts - b.ts)
    .map((event) => ({
      token: canonicalMovementToken(event.tool, event.summary),
      tool: event.tool,
      summary: event.summary,
      metadata: { trajectoryId: event.trajectoryId, ts: event.ts },
    }));
}

/**
 * Roll a trained model forward from a seed context, greedily selecting the most
 * probable next movement at each step. This is the model-driven replay engine:
 * seed it with a recorded prefix to reproduce movements, or with a held-out but
 * related prefix to observe generalization. Stops at `maxSteps` or when no
 * prediction is available. `stopToken` (if given) ends the rollout when emitted.
 */
export function rolloutMovements(
  backend: MovementTrainingBackend,
  model: TrainedMovementModel,
  seed: MovementStep[],
  options: { maxSteps?: number; stopToken?: string } = {},
): MovementStep[] {
  const maxSteps = options.maxSteps ?? 32;
  const generated: MovementStep[] = [];
  const context = [...seed];
  for (let i = 0; i < maxSteps; i += 1) {
    const prediction = backend.predictNext(model, context);
    if (!prediction) {
      break;
    }
    generated.push(prediction.step);
    context.push(prediction.step);
    if (options.stopToken && prediction.token === options.stopToken) {
      break;
    }
  }
  return generated;
}

/** Internal: build the joined transition key for a context slice. */
export function movementContextKey(context: MovementStep[]): string {
  return context.map((step) => step.token).join(CONTEXT_SEPARATOR);
}

export { CONTEXT_SEPARATOR as MOVEMENT_CONTEXT_SEPARATOR };
