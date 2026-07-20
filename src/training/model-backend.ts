import type { ReplayManifest } from "../capture/replay.js";
import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

/**
 * Pluggable local-model backend for the movement-learning subsystem.
 *
 * Standing objective #2 (c/d): post-train a local model on recorded movements
 * so it can (c) repeat the recorded movements and (d) generalize to new but
 * related movements. This module defines the *seam* — a backend-agnostic
 * interface plus the tokenization/dataset builders that turn captured
 * trajectories and replay manifests into training data.
 *
 * The interface is deliberately synchronous for inference and async for
 * training so a real on-device backend (MLX / a small local model) can spawn
 * a training process while the deterministic in-process backend used by tests
 * resolves immediately. Everything here is fully deterministic — no clocks, no
 * randomness — so it validates in the cloud/CI against synthetic event streams.
 */

/** Structural boundary tokens wrapped around every training sequence. */
export const MOVEMENT_START_TOKEN = "<s>";
export const MOVEMENT_END_TOKEN = "</s>";

/** A single tokenized movement trajectory ready for training. */
export type MovementSequence = {
  id: string;
  tokens: string[];
};

/** A collection of movement sequences — the dataset a backend trains on. */
export type MovementDataset = {
  sequences: MovementSequence[];
};

export type MovementTokenizeOptions = {
  /**
   * When true, fold a slug of the action summary into the token so the model
   * distinguishes e.g. `click#ok` from `click#cancel`. Defaults to false —
   * tool-level tokens generalize better across related movements.
   */
  includeSummary?: boolean;
  /** Cap on summary words folded into the token (default 3). */
  maxSummaryWords?: number;
};

/** Backend-agnostic, JSON-serializable trained model artifact. */
export type TrainedMovementModel = {
  backend: string;
  version: 1;
  order: number;
  vocabulary: string[];
  /** Backend-specific weights; opaque to callers, serialized as plain JSON. */
  weights: unknown;
  stats: {
    sequenceCount: number;
    tokenCount: number;
  };
};

export type MovementTrainingConfig = {
  /**
   * Maximum context length the model conditions on (the `n-1` of an n-gram).
   * Higher order = more faithful replay, lower order = more generalization.
   * Defaults to 2.
   */
  order?: number;
};

export type MovementPrediction = {
  /** Highest-probability next token, or undefined when the model is empty. */
  token: string | undefined;
  /** Full next-token distribution, sorted by probability desc then token asc. */
  distribution: Array<{ token: string; probability: number }>;
  /** Which back-off context length produced the prediction (<= model.order). */
  contextOrderUsed: number;
};

export type MovementGenerateParams = {
  /** Starting context tokens (excluding the implicit START token). */
  seed?: readonly string[];
  /** Hard cap on generated steps (default 64). */
  maxSteps?: number;
  /** Stop when the END token is predicted; END is not included (default true). */
  stopAtEnd?: boolean;
};

export interface LocalModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, config?: MovementTrainingConfig): Promise<TrainedMovementModel>;
  predict(model: TrainedMovementModel, context: readonly string[]): MovementPrediction;
  generate(model: TrainedMovementModel, params?: MovementGenerateParams): string[];
}

// --------------------------------------------------------------------------
// Tokenization + dataset builders (capture -> dataset bridge)
// --------------------------------------------------------------------------

function slugSummary(summary: string, maxWords: number): string {
  return summary
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim()
    .split(/\s+/)
    .filter((word) => word.length > 0)
    .slice(0, maxWords)
    .join("-");
}

export function tokenizeMovementAction(
  action: { tool: string; summary?: string },
  options: MovementTokenizeOptions = {},
): string {
  const tool = action.tool.trim() || "unknown";
  if (!options.includeSummary || !action.summary) {
    return tool;
  }
  const slug = slugSummary(action.summary, options.maxSummaryWords ?? 3);
  return slug ? `${tool}#${slug}` : tool;
}

export function movementSequenceFromTrajectory(
  trajectory: TrajectorySpan,
  options: MovementTokenizeOptions = {},
): MovementSequence {
  const actions = trajectory.review?.redactedActions
    ? trajectory.review.redactedActions.map<TrajectoryAction>((action) => ({
        kind: "action",
        tool: action.tool,
        summary: action.summary,
        ts: action.ts,
      }))
    : trajectory.actions;
  const tokens = [...actions]
    .sort((a, b) => a.ts - b.ts)
    .map((action) => tokenizeMovementAction(action, options));
  return { id: trajectory.id, tokens };
}

export function movementDatasetFromTrajectories(
  trajectories: readonly TrajectorySpan[],
  options: MovementTokenizeOptions = {},
): MovementDataset {
  return {
    sequences: trajectories
      .map((trajectory) => movementSequenceFromTrajectory(trajectory, options))
      .filter((sequence) => sequence.tokens.length > 0),
  };
}

export function movementSequenceFromReplay(
  replay: Pick<ReplayManifest, "sessionId" | "events">,
  options: MovementTokenizeOptions = {},
): MovementSequence {
  const tokens = replay.events
    .filter((event): event is Extract<typeof event, { kind: "action" }> => event.kind === "action")
    .slice()
    .sort((a, b) => a.ts - b.ts)
    .map((event) => tokenizeMovementAction(event, options));
  return { id: replay.sessionId, tokens };
}

export function movementDatasetFromReplays(
  replays: ReadonlyArray<Pick<ReplayManifest, "sessionId" | "events">>,
  options: MovementTokenizeOptions = {},
): MovementDataset {
  return {
    sequences: replays
      .map((replay) => movementSequenceFromReplay(replay, options))
      .filter((sequence) => sequence.tokens.length > 0),
  };
}

// --------------------------------------------------------------------------
// Backend registry (pluggable seam)
// --------------------------------------------------------------------------

export type ModelBackendFactory = () => LocalModelBackend;

const backendRegistry = new Map<string, ModelBackendFactory>();

export function registerModelBackend(name: string, factory: ModelBackendFactory): void {
  backendRegistry.set(name, factory);
}

export function listModelBackends(): string[] {
  return [...backendRegistry.keys()].sort();
}

export function createModelBackend(name: string): LocalModelBackend {
  const factory = backendRegistry.get(name);
  if (!factory) {
    throw new Error(
      `unknown model backend "${name}"; registered backends: ${listModelBackends().join(", ") || "(none)"}`,
    );
  }
  return factory();
}

export const DEFAULT_MODEL_BACKEND = "markov";

export function defaultModelBackend(): LocalModelBackend {
  return createModelBackend(DEFAULT_MODEL_BACKEND);
}
