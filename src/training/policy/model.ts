/**
 * Movement-policy model surface.
 *
 * This is the in-process, backend-agnostic seam for standing objective #2 pieces
 * (c) post-train a local model on recorded movements and (d) generalize to new but
 * related movements. The default backend (see `ngram-backend.ts`) runs fully
 * in-process so the train -> predict -> generalize loop is validated in the cloud
 * with synthetic data; a real on-device small model (mlx/gguf) can implement the
 * same {@link MovementModelBackend} interface without changing any call sites.
 */

/** A situation the agent was in: an order-independent bag of feature tokens. */
export type MovementContext = {
  /** Canonical feature tokens describing recent observations + the prior action. */
  tokens: string[];
};

/** A normalized, replayable movement/action label (the model's prediction target). */
export type MovementActionLabel = {
  /** Channel that performed the action, e.g. "device", "os", "browser". */
  tool: string;
  /** Discriminating descriptor, e.g. "tap:compose" or "swipe:down". */
  descriptor: string;
};

/** One supervised step: the context, and the action taken next. */
export type MovementTrainingExample = {
  context: MovementContext;
  action: MovementActionLabel;
  /** Per-example weight (reward-shaped). Defaults to 1 when omitted. */
  weight?: number;
  sourceTrajectoryId?: string;
};

/** A structured, replayable training dataset derived from recorded trajectories. */
export type MovementDataset = {
  version: 1;
  examples: MovementTrainingExample[];
  /** Distinct action keys present, sorted — the model's action vocabulary. */
  actionVocabulary: string[];
};

export type MovementPredictionMethod = "exact" | "generalized" | "prior" | "none";

export type MovementScoredAction = {
  action: MovementActionLabel;
  confidence: number;
};

export type MovementPrediction = {
  /** Best action, or undefined when the model has learned nothing applicable. */
  action: MovementActionLabel | undefined;
  confidence: number;
  method: MovementPredictionMethod;
  /** Ranked alternatives (best first) for beam replay / debugging. */
  alternatives: MovementScoredAction[];
};

/**
 * A trained model artifact. Opaque to callers; each backend defines its own
 * `state` shape. Serializable to a portable JSON string via the backend.
 */
export type TrainedMovementModel<TState = unknown> = {
  version: 1;
  backend: string;
  /** ISO timestamp, or null for a deterministic/reproducible artifact. */
  trainedAt: string | null;
  exampleCount: number;
  actionVocabulary: string[];
  state: TState;
};

export type MovementTrainOptions = {
  /** ISO timestamp to stamp the model; omit for a deterministic artifact. */
  trainedAt?: string;
};

/**
 * Pluggable local-model backend. Implementations must be deterministic given the
 * same dataset + options so cloud/CI runs are reproducible.
 */
export interface MovementModelBackend<TState = unknown> {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainOptions): TrainedMovementModel<TState>;
  predict(model: TrainedMovementModel<TState>, context: MovementContext): MovementPrediction;
  serialize(model: TrainedMovementModel<TState>): string;
  deserialize(serialized: string): TrainedMovementModel<TState>;
}

/** Canonical, stable key for an action label. */
export function actionKey(action: MovementActionLabel): string {
  return `${action.tool} ${action.descriptor}`;
}

/** Parse an {@link actionKey} back into a label. */
export function parseActionKey(key: string): MovementActionLabel {
  const separator = key.indexOf(" ");
  if (separator < 0) {
    return { tool: key, descriptor: "" };
  }
  return { tool: key.slice(0, separator), descriptor: key.slice(separator + 1) };
}

/** Deduplicate + sort context tokens so equivalent situations share one key. */
export function normalizeTokens(tokens: Iterable<string>): string[] {
  const seen = new Set<string>();
  for (const token of tokens) {
    const trimmed = token.trim();
    if (trimmed.length > 0) {
      seen.add(trimmed);
    }
  }
  return [...seen].sort();
}

/** Stable key for a context (order-independent). */
export function contextKey(context: MovementContext): string {
  return normalizeTokens(context.tokens).join("|");
}
