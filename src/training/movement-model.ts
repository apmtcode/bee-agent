import type { ReplayManifest } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Local-movement learning model layer.
 *
 * This is the pluggable seam objective #2(c)/(d) asks for: post-train a *local*
 * model on recorded movements and generalize to new-but-related movements. It
 * sits alongside the Apple-Silicon subprocess runner (mlx/axolotl) — that runner
 * is the "real on-device heavy model" seam; this layer is the lightweight,
 * fully in-process seam that trains and infers with zero external processes, so
 * the whole capture -> dataset -> train -> infer -> generalize loop is exercised
 * (and validated) in the cloud/CI with simulated event streams.
 *
 * A movement is reduced to a stable string *token* (e.g. `device:tap:submit`).
 * Backends learn over token sequences and predict the next movement given a
 * context prefix; higher-order context with backoff is what yields
 * generalization to unseen prefixes.
 */

export type MovementToken = string;

/** One recorded movement stream (the ordered movements within a trajectory). */
export type MovementSequence = {
  /** Stable identifier of the source (trajectory id, replay session id, ...). */
  id: string;
  tokens: MovementToken[];
};

export type MovementDataset = {
  sequences: MovementSequence[];
};

export type MovementModelConfig = {
  /** Max Markov order (context length). Effective order is clamped to >= 1. */
  order: number;
};

export const DEFAULT_MOVEMENT_MODEL_CONFIG: MovementModelConfig = { order: 2 };

/** A ranked next-movement prediction. `probability` is within [0, 1]. */
export type MovementPrediction = {
  token: MovementToken;
  probability: number;
  /** How many context tokens actually matched (0 = unigram/backoff to prior). */
  matchedOrder: number;
};

/**
 * A serializable, backend-tagged trained model. `spec` is opaque JSON owned by
 * the backend that produced it; only the producing backend can `load` it.
 */
export type MovementModelArtifact = {
  backendId: string;
  version: 1;
  config: MovementModelConfig;
  vocabulary: MovementToken[];
  sequenceCount: number;
  tokenCount: number;
  spec: unknown;
};

/** An inference handle over a trained model. */
export interface MovementModel {
  readonly backendId: string;
  readonly vocabulary: ReadonlyArray<MovementToken>;
  /** Ranked predictions for the movement following `context` (most-recent last). */
  predictNext(context: MovementToken[]): MovementPrediction[];
  /**
   * Deterministically roll out a movement sequence from a seed context.
   * Stops after `steps` tokens or when a `stop` token is emitted.
   */
  generate(params: { seed?: MovementToken[]; steps: number; stop?: MovementToken[] }): MovementToken[];
}

/** A pluggable local-model backend (mock/simulated now; real on-device later). */
export interface LocalMovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, config?: MovementModelConfig): Promise<MovementModelArtifact>;
  load(artifact: MovementModelArtifact): MovementModel;
}

/**
 * Registry of movement backends so the training layer can select one by id and
 * new backends (a real on-device small model) can be dropped in without
 * touching call sites.
 */
export class MovementBackendRegistry {
  private readonly backends = new Map<string, LocalMovementModelBackend>();

  register(backend: LocalMovementModelBackend): this {
    this.backends.set(backend.id, backend);
    return this;
  }

  get(id: string): LocalMovementModelBackend {
    const backend = this.backends.get(id);
    if (!backend) {
      throw new Error(`unknown movement backend: ${id}`);
    }
    return backend;
  }

  has(id: string): boolean {
    return this.backends.has(id);
  }

  list(): string[] {
    return [...this.backends.keys()].sort();
  }

  load(artifact: MovementModelArtifact): MovementModel {
    return this.get(artifact.backendId).load(artifact);
  }
}

/** Reduce a recorded gesture/action into a stable, low-cardinality token. */
export function movementActionToken(params: {
  tool: string;
  gesture?: string;
  target?: string;
  direction?: string;
}): MovementToken {
  const parts = [normalizeTokenPart(params.tool)];
  if (params.gesture) {
    parts.push(normalizeTokenPart(params.gesture));
  }
  const qualifier = params.target ?? params.direction;
  if (qualifier) {
    parts.push(normalizeTokenPart(qualifier));
  }
  return parts.filter((part) => part.length > 0).join(":");
}

function normalizeTokenPart(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

/**
 * Build a movement dataset from reviewed trajectories. Prefers redacted
 * (reviewed-approved) actions so raw capture never leaks into training; falls
 * back to the span's own actions when no review is attached.
 */
export function extractMovementSequences(trajectories: TrajectorySpan[]): MovementDataset {
  const sequences: MovementSequence[] = [];
  for (const trajectory of trajectories) {
    const reviewed = trajectory.review?.redactedActions;
    const source = reviewed && reviewed.length > 0
      ? reviewed.map((action) => ({ tool: action.tool, summary: action.summary, metadata: undefined as Record<string, unknown> | undefined }))
      : trajectory.actions.map((action) => ({ tool: action.tool, summary: action.summary, metadata: action.metadata }));
    const tokens = source
      .map((action) => tokenFromAction(action))
      .filter((token): token is MovementToken => token.length > 0);
    if (tokens.length > 0) {
      sequences.push({ id: trajectory.id, tokens });
    }
  }
  return { sequences };
}

/** Build a movement dataset from a replay manifest's action events. */
export function extractMovementSequencesFromReplay(manifest: ReplayManifest): MovementDataset {
  const byTrajectory = new Map<string, MovementToken[]>();
  for (const event of manifest.events) {
    if (event.kind !== "action") {
      continue;
    }
    const token = movementActionToken({ tool: event.tool, gesture: gestureFromSummary(event.summary) });
    if (token.length === 0) {
      continue;
    }
    const existing = byTrajectory.get(event.trajectoryId);
    if (existing) {
      existing.push(token);
    } else {
      byTrajectory.set(event.trajectoryId, [token]);
    }
  }
  return {
    sequences: [...byTrajectory.entries()].map(([id, tokens]) => ({ id, tokens })),
  };
}

function tokenFromAction(action: { tool: string; summary: string; metadata?: Record<string, unknown> }): MovementToken {
  const gesture = typeof action.metadata?.gesture === "string" ? action.metadata.gesture : gestureFromSummary(action.summary);
  const target = typeof action.metadata?.target === "string" ? action.metadata.target : undefined;
  const direction = typeof action.metadata?.direction === "string" ? action.metadata.direction : undefined;
  return movementActionToken({ tool: action.tool, gesture, target, direction });
}

function gestureFromSummary(summary: string): string | undefined {
  const first = summary.trim().split(/\s+/)[0];
  return first && first.length > 0 ? first : undefined;
}

/**
 * Held-out next-token evaluation used to measure generalization: for every
 * position in each held-out sequence, ask the model to predict the next token
 * from the preceding context and score top-1 / top-k hits.
 */
export function evaluateNextTokenAccuracy(
  model: MovementModel,
  heldOut: MovementDataset,
  options: { topK?: number } = {},
): { total: number; top1: number; topK: number; top1Accuracy: number; topKAccuracy: number } {
  const k = Math.max(1, options.topK ?? 3);
  let total = 0;
  let top1 = 0;
  let topK = 0;
  for (const sequence of heldOut.sequences) {
    for (let i = 1; i < sequence.tokens.length; i += 1) {
      const context = sequence.tokens.slice(0, i);
      const expected = sequence.tokens[i];
      const predictions = model.predictNext(context);
      if (predictions.length === 0) {
        total += 1;
        continue;
      }
      total += 1;
      if (predictions[0]?.token === expected) {
        top1 += 1;
      }
      if (predictions.slice(0, k).some((prediction) => prediction.token === expected)) {
        topK += 1;
      }
    }
  }
  return {
    total,
    top1,
    topK,
    top1Accuracy: total === 0 ? 0 : top1 / total,
    topKAccuracy: total === 0 ? 0 : topK / total,
  };
}
