import type { ReplayManifest } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Movement-model subsystem (standing objective #2c/#2d).
 *
 * This module defines the *pluggable, backend-agnostic* contract for a local
 * model that learns from recorded movement/action sequences and can both
 * (a) repeat the recorded movements and (b) generalize to new-but-related
 * movements. Concrete learners (a deterministic n-gram model for cloud/CI, a
 * real on-device small model when the user runs bee-agent locally) implement
 * {@link MovementModelBackend}. Everything here is pure/in-process so it runs
 * and is testable in the cloud against synthetic event streams.
 */

/**
 * A single discrete movement, encoded as a stable string token (e.g.
 * `device:tapped Submit`). Tokens are the alphabet the model learns over.
 */
export type MovementToken = string;

/** An ordered run of movements captured from one trajectory/session. */
export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

/** The training corpus: a set of recorded movement sequences. */
export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

/** Backend-tunable knobs. Backends ignore fields they do not use. */
export type MovementTrainingConfig = {
  /** Maximum context length (n-gram order) the backend should learn. */
  order?: number;
  /** Additive (Laplace) smoothing mass for unseen continuations. */
  smoothing?: number;
};

/**
 * A trained model. The envelope is uniform so models can be persisted/loaded
 * and routed by `backendId`; `data` is backend-private state.
 */
export type MovementModel = {
  backendId: string;
  version: 1;
  order: number;
  vocabulary: MovementToken[];
  data: Record<string, unknown>;
  metadata: {
    sequenceCount: number;
    tokenCount: number;
    smoothing: number;
  };
};

/** A scored continuation candidate. */
export type MovementCandidate = {
  token: MovementToken;
  probability: number;
};

/** The model's belief about the next movement given some context. */
export type MovementPrediction = {
  /** Highest-probability next token, or `undefined` if the model is empty. */
  token: MovementToken | undefined;
  probability: number;
  /** Candidates ranked by probability (descending, deterministic tie-break). */
  ranked: MovementCandidate[];
  /** The context length the prediction was actually drawn from after back-off. */
  backoffOrder: number;
};

/**
 * The pluggable learner contract. `train` is async to leave room for real
 * on-device backends; the rest are synchronous and deterministic so replay and
 * evaluation are reproducible.
 */
export interface MovementModelBackend {
  /** Stable identifier stamped onto every model this backend produces. */
  readonly id: string;
  train(dataset: MovementDataset, config?: MovementTrainingConfig): Promise<MovementModel>;
  predict(model: MovementModel, context: MovementToken[]): MovementPrediction;
  /** Greedily roll out up to `steps` movements from `seed`. */
  generate(model: MovementModel, seed: MovementToken[], steps: number): MovementToken[];
}

/** Registry so call sites can resolve a backend by id at runtime. */
export class MovementModelBackendRegistry {
  private readonly backends = new Map<string, MovementModelBackend>();

  register(backend: MovementModelBackend): this {
    this.backends.set(backend.id, backend);
    return this;
  }

  get(id: string): MovementModelBackend | undefined {
    return this.backends.get(id);
  }

  require(id: string): MovementModelBackend {
    const backend = this.backends.get(id);
    if (!backend) {
      throw new Error(`unknown movement-model backend: ${id}`);
    }
    return backend;
  }

  list(): MovementModelBackend[] {
    return [...this.backends.values()];
  }
}

/**
 * Encode one captured action as a stable token. Uses the tool plus a
 * normalized summary so semantically-identical movements collapse to the same
 * token (which is what lets the model generalize across sessions).
 */
export function encodeMovementToken(tool: string, summary: string): MovementToken {
  return `${normalizeTokenPart(tool)}:${normalizeTokenPart(summary)}`;
}

function normalizeTokenPart(value: string): string {
  return value.trim().replace(/\s+/g, " ").toLowerCase();
}

/** Project a trajectory's ordered actions into a movement sequence. */
export function tokenizeTrajectory(trajectory: TrajectorySpan): MovementSequence {
  const actions = trajectory.review?.redactedActions
    ? trajectory.review.redactedActions.map((action) => ({ tool: action.tool, summary: action.summary, ts: action.ts }))
    : trajectory.actions.map((action) => ({ tool: action.tool, summary: action.summary, ts: action.ts }));
  const ordered = [...actions].sort((a, b) => a.ts - b.ts);
  return {
    id: trajectory.id,
    tokens: ordered.map((action) => encodeMovementToken(action.tool, action.summary)),
  };
}

/** Project a replay manifest's action timeline into per-trajectory sequences. */
export function tokenizeReplayManifest(manifest: ReplayManifest): MovementSequence[] {
  const byTrajectory = new Map<string, { ts: number; token: MovementToken }[]>();
  for (const event of manifest.events) {
    if (event.kind !== "action") {
      continue;
    }
    const tokens = byTrajectory.get(event.trajectoryId) ?? [];
    tokens.push({ ts: event.ts, token: encodeMovementToken(event.tool, event.summary) });
    byTrajectory.set(event.trajectoryId, tokens);
  }
  return [...byTrajectory.entries()].map(([id, entries]) => ({
    id,
    tokens: [...entries].sort((a, b) => a.ts - b.ts).map((entry) => entry.token),
  }));
}

/** Build a dataset directly from captured trajectories (drops empty runs). */
export function buildMovementDataset(trajectories: TrajectorySpan[]): MovementDataset {
  return {
    version: 1,
    sequences: trajectories.map(tokenizeTrajectory).filter((sequence) => sequence.tokens.length > 0),
  };
}

/** Aggregate report from {@link evaluateMovementModel}. */
export type MovementEvaluation = {
  /** Teacher-forced next-token accuracy over all eligible positions. */
  nextTokenAccuracy: number;
  /** Positions evaluated (sequence length minus one, summed). */
  predictions: number;
  /** Correct top-1 predictions. */
  correct: number;
  /** Fraction of predictions that required backing off below full order. */
  backoffRate: number;
};

/**
 * Generalization eval harness: teacher-force the model through held-out
 * sequences and measure how often its top-1 continuation matches reality.
 * Works on sequences the model never trained on, which is exactly the
 * "generalize to new-but-related movements" signal.
 */
export function evaluateMovementModel(
  backend: MovementModelBackend,
  model: MovementModel,
  sequences: MovementSequence[],
): MovementEvaluation {
  let predictions = 0;
  let correct = 0;
  let backedOff = 0;
  for (const sequence of sequences) {
    for (let index = 0; index < sequence.tokens.length - 1; index += 1) {
      const context = sequence.tokens.slice(0, index + 1);
      const expected = sequence.tokens[index + 1];
      const prediction = backend.predict(model, context);
      predictions += 1;
      if (prediction.token === expected) {
        correct += 1;
      }
      if (prediction.backoffOrder < Math.min(model.order, context.length)) {
        backedOff += 1;
      }
    }
  }
  return {
    nextTokenAccuracy: predictions === 0 ? 0 : correct / predictions,
    predictions,
    correct,
    backoffRate: predictions === 0 ? 0 : backedOff / predictions,
  };
}
