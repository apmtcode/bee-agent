// Local-movement learning: model schema + pluggable backend contract.
//
// This module defines the *shape* of a learned movement policy and the seam a
// backend must implement. The default backend (`ngram-backend.ts`) is a
// deterministic, in-process statistical model so post-training and inference
// can be validated in the cloud with synthetic data (the engine has no access
// to a real machine). A real on-device backend (e.g. an MLX/torch small model)
// implements the same `MovementModelBackend` interface and produces a
// serializable `MovementModel` artifact, so callers never depend on which
// backend trained the model.

import type { DevicePlatform } from "../capture/device-adapter.js";

/** The situation in which a movement occurs. Used as the conditioning key. */
export type MovementContext = {
  platform: DevicePlatform;
  appId: string;
  /** Screen / window / route bucket. Optional — models back off without it. */
  screen?: string;
};

/** One recorded or predicted movement primitive. */
export type MovementStep = {
  /**
   * Canonical action token, e.g. `tap`, `swipe:down`, `type`, `shortcut:save`.
   * Direction/modifier is folded into the token so the model conditions on it.
   */
  action: string;
  /** UI element / control the action targeted, when known. */
  target?: string;
  /** Milliseconds since epoch (relative ordering only; not learned). */
  ts: number;
};

/** A single demonstrated movement sequence within one context. */
export type MovementTrajectory = {
  id: string;
  context: MovementContext;
  steps: MovementStep[];
};

/** The replayable training dataset: many demonstrations across contexts. */
export type MovementDataset = {
  version: 1;
  trajectories: MovementTrajectory[];
};

/**
 * Serializable, backend-tagged learned artifact. Plain JSON so it can be
 * persisted to disk and reloaded on-device without the training code present.
 * The internal shape is backend-defined; callers only ever pass it back to the
 * same backend's `predict`.
 */
export type MovementModel = {
  backend: string;
  version: 1;
  trainedTrajectories: number;
  trainedSteps: number;
  /** Backend-private learned parameters. */
  weights: unknown;
};

export type TrainOptions = {
  /** History length the model conditions on (n-gram order). Default 2. */
  order?: number;
};

export type PredictRequest = {
  context: MovementContext;
  /** Steps already performed this run; the model continues from them. */
  prefix?: MovementStep[];
  /** Hard cap on emitted steps. Default 32. */
  maxSteps?: number;
};

export type PredictedMovementStep = {
  action: string;
  target?: string;
  /** Normalized transition mass for the chosen action, 0..1. */
  confidence: number;
  /**
   * How much the model had to generalize: 0 = exact context match,
   * higher = backed off to app / platform / global statistics. Lets callers
   * (and tests) observe that a novel-but-related context was handled by
   * generalization rather than memorization.
   */
  backoffLevel: number;
};

export type MovementPrediction = {
  backend: string;
  steps: PredictedMovementStep[];
  /** Max backoff level used across the emitted steps. */
  maxBackoffLevel: number;
};

/**
 * The pluggable seam. `train` turns a dataset into a serializable model;
 * `predict` runs inference. Both are async so real backends can shell out to a
 * training runtime or load a model file without changing the contract.
 */
export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, options?: TrainOptions): Promise<MovementModel>;
  predict(model: MovementModel, request: PredictRequest): Promise<MovementPrediction>;
}

/** Marker token the model emits/observes to represent end-of-sequence. */
export const MOVEMENT_END_TOKEN = "<end>";

/**
 * Context specificity levels, most specific first. Backends learn statistics at
 * each level and back off down the list so an unseen-but-related context (new
 * screen in a known app) is served by app-level statistics — the generalization
 * mechanism.
 */
export function movementContextKeys(context: MovementContext): string[] {
  const app = context.appId.trim().toLowerCase();
  const platform = context.platform;
  const screen = context.screen?.trim().toLowerCase();
  return [
    screen ? `${platform}|${app}|${screen}` : `${platform}|${app}|`,
    `${platform}|${app}`,
    `${platform}`,
    "*",
  ];
}

const registry = new Map<string, () => MovementModelBackend>();

/** Register a backend factory so callers can select one by name. */
export function registerMovementModelBackend(name: string, factory: () => MovementModelBackend): void {
  registry.set(name, factory);
}

/** Instantiate a registered backend by name. Throws if unknown. */
export function createMovementModelBackend(name: string): MovementModelBackend {
  const factory = registry.get(name);
  if (!factory) {
    throw new Error(`unknown movement model backend: ${name} (registered: ${[...registry.keys()].join(", ") || "none"})`);
  }
  return factory();
}

/** Names of all registered backends. */
export function listMovementModelBackends(): string[] {
  return [...registry.keys()].sort();
}
