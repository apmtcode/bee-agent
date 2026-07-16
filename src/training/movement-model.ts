// Local-movement learning subsystem — model backend contract.
//
// This module defines the *pluggable* seam between bee-agent's recorded
// movement trajectories and whatever local model learns to repeat and
// generalize them. It is intentionally backend-agnostic: the deterministic
// mock backend (`NgramMovementBackend`) ships in-repo so the whole
// capture -> dataset -> train -> infer loop is exercisable in the cloud with
// synthetic streams, while a real on-device backend (e.g. an MLX/axolotl
// policy) can implement the same `MovementModelBackend` interface and slot in
// via `MovementModelRegistry` with zero call-site changes.

export type MovementActor = "mouse" | "keyboard" | "window" | "application" | "touch";

export type MovementDirection = "up" | "down" | "left" | "right";

/**
 * A single low-level movement/action on the local computer. Deliberately flat
 * and JSON-serializable so it can be recorded, replayed, and tokenized without
 * losing information across the capture -> dataset -> train boundary.
 */
export type MovementEvent = {
  ts: number;
  actor: MovementActor;
  action: string;
  target?: string;
  x?: number;
  y?: number;
  key?: string;
  direction?: MovementDirection;
  value?: string;
};

/** What the recorded movements were trying to accomplish. Drives generalization. */
export type MovementContext = {
  goal: string;
  appId?: string;
  platform?: string;
  labels?: string[];
};

export type MovementTrajectory = {
  id: string;
  context: MovementContext;
  events: MovementEvent[];
};

export type MovementTrainingDataset = {
  trajectories: MovementTrajectory[];
};

export type MovementPredictionSource =
  | "context" // exact (appId + goal) context match
  | "app" // same app, different goal — related-movement generalization
  | "global"; // cross-context backoff — broadest generalization

export type MovementPrediction = {
  /** The predicted next event, or `undefined` when the model predicts the sequence ends. */
  event?: MovementEvent;
  token: string;
  /** Probability of this token under the matched distribution, in [0, 1]. */
  confidence: number;
  source: MovementPredictionSource;
  /** True when the model predicts the trajectory is complete. */
  end: boolean;
};

export type MovementModelTrainingConfig = {
  /** Markov order (how many prior tokens condition the next). Default 2. */
  order?: number;
};

export type MovementRolloutOptions = {
  maxSteps?: number;
  startTs?: number;
};

export type SerializedMovementModel = {
  version: 1;
  backendId: string;
  order: number;
  modelVersion: number;
  /** bucketKey -> "len|gram" -> nextToken -> count */
  transitions: Record<string, Record<string, Record<string, number>>>;
  /** token -> representative event template (ts stripped) + inter-event delta */
  templates: Record<string, { event: Omit<MovementEvent, "ts">; dt: number }>;
};

/** A trained, queryable movement policy. */
export interface TrainedMovementModel {
  readonly backendId: string;
  readonly order: number;
  readonly version: number;
  /** Predict the single next event given a context and the prefix so far. */
  predictNext(context: MovementContext, prefix: MovementEvent[]): MovementPrediction;
  /** Greedily roll out a full trajectory for a context (deterministic argmax). */
  rollout(context: MovementContext, options?: MovementRolloutOptions): MovementEvent[];
  serialize(): SerializedMovementModel;
}

/** A pluggable training backend. Mock and real backends both implement this. */
export interface MovementModelBackend {
  readonly id: string;
  train(
    dataset: MovementTrainingDataset,
    config?: MovementModelTrainingConfig,
  ): Promise<TrainedMovementModel>;
  load(serialized: SerializedMovementModel): TrainedMovementModel;
}

export const MOVEMENT_START_TOKEN = "<s>";
export const MOVEMENT_END_TOKEN = "</s>";

const COORD_BUCKET = 24;

function bucket(value: number): number {
  return Math.floor(value / COORD_BUCKET);
}

/**
 * Map an event to a discrete token. Coordinates are coarsely bucketed so
 * nearby positions share a token (this is what lets the model generalize a
 * "click near here" movement rather than memorizing exact pixels).
 */
export function movementEventToken(event: MovementEvent): string {
  const parts: string[] = [event.actor, event.action];
  if (event.target) parts.push(`@${event.target}`);
  if (event.direction) parts.push(`>${event.direction}`);
  if (event.key) parts.push(`k:${event.key}`);
  if (event.value !== undefined) parts.push(`v:${event.value}`);
  if (event.x !== undefined && event.y !== undefined) {
    parts.push(`p:${bucket(event.x)},${bucket(event.y)}`);
  }
  return parts.join("|");
}

/** Exact-context bucket key: same app AND same goal. */
export function movementContextKey(context: MovementContext): string {
  return `ctx:${context.appId ?? "*"}::${context.goal}`;
}

/** App-level bucket key: same app, any goal (related-movement generalization). */
export function movementAppKey(context: MovementContext): string {
  return `app:${context.appId ?? "*"}`;
}

export const MOVEMENT_GLOBAL_KEY = "global";

/** Registry that makes the model backend pluggable at runtime. */
export class MovementModelRegistry {
  private readonly backends = new Map<string, MovementModelBackend>();

  register(backend: MovementModelBackend): this {
    if (this.backends.has(backend.id)) {
      throw new Error(`movement model backend already registered: ${backend.id}`);
    }
    this.backends.set(backend.id, backend);
    return this;
  }

  has(id: string): boolean {
    return this.backends.has(id);
  }

  get(id: string): MovementModelBackend {
    const backend = this.backends.get(id);
    if (!backend) {
      throw new Error(`unknown movement model backend: ${id}`);
    }
    return backend;
  }

  list(): string[] {
    return [...this.backends.keys()].sort();
  }

  /** Rehydrate a serialized model using its declared backend. */
  load(serialized: SerializedMovementModel): TrainedMovementModel {
    return this.get(serialized.backendId).load(serialized);
  }
}
