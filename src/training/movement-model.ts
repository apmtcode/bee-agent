// Local-movement learning subsystem — model layer.
//
// This module defines the atomic movement schema, a discrete tokenizer, a
// dataset format, and the *pluggable* model-backend interfaces the training
// runner can drive. The concrete deterministic reference backend lives in
// `ngram-backend.ts`; a real on-device small-model backend (e.g. an MLX/torch
// policy) can implement the same `MovementModelBackend` interface later without
// touching callers — that is the documented on-device seam.
//
// Everything here is dependency-free and runs in-process, so the capture →
// dataset → train → infer loop is fully exercisable in the cloud/CI against
// synthetic event streams (see `synthetic-movements.ts`).

import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

/** The recognised classes of local movement/action a model learns to repeat. */
export const MOVEMENT_KINDS = [
  "tap",
  "swipe",
  "scroll",
  "type",
  "shortcut",
  "click",
  "keypress",
  "move",
  "focus",
  "open",
] as const;

export type MovementKind = (typeof MOVEMENT_KINDS)[number];

export const MOVEMENT_DIRECTIONS = ["up", "down", "left", "right"] as const;

export type MovementDirection = (typeof MOVEMENT_DIRECTIONS)[number];

/** The atomic unit of the learnable movement stream. */
export type MovementEvent = {
  ts: number;
  kind: MovementKind;
  /** UI element / region / app the movement acted on. */
  target?: string;
  direction?: MovementDirection;
  /** Free-text summary of a typed value / key name — not part of token identity. */
  value?: string;
};

export type MovementSequence = {
  id: string;
  events: MovementEvent[];
};

export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

/** Sentinel tokens marking sequence boundaries in the model's symbol stream. */
export const MOVEMENT_START_TOKEN = "START";
export const MOVEMENT_END_TOKEN = "END";

const TOKEN_FIELD_SEPARATOR = "|";

const MOVEMENT_KIND_SET = new Set<string>(MOVEMENT_KINDS);
const MOVEMENT_DIRECTION_SET = new Set<string>(MOVEMENT_DIRECTIONS);

/**
 * Reduce a movement to a discrete symbol. `value` is intentionally excluded so
 * that "type into the username field" collapses to one symbol regardless of the
 * exact text typed — this is what lets the model generalise across runs.
 */
export function movementToken(event: MovementEvent): string {
  return [event.kind, event.target ?? "*", event.direction ?? "*"].join(TOKEN_FIELD_SEPARATOR);
}

/** Best-effort reconstruction of an event skeleton from a token (for generation). */
export function movementFromToken(token: string, ts: number): MovementEvent | undefined {
  if (token === MOVEMENT_START_TOKEN || token === MOVEMENT_END_TOKEN) {
    return undefined;
  }
  const [kind, target, direction] = token.split(TOKEN_FIELD_SEPARATOR);
  if (!kind || !MOVEMENT_KIND_SET.has(kind)) {
    return undefined;
  }
  const event: MovementEvent = { ts, kind: kind as MovementKind };
  if (target && target !== "*") {
    event.target = target;
  }
  if (direction && direction !== "*" && MOVEMENT_DIRECTION_SET.has(direction)) {
    event.direction = direction as MovementDirection;
  }
  return event;
}

function normalizeMovementKind(value: string | undefined): MovementKind | undefined {
  if (!value) {
    return undefined;
  }
  const normalized = value.trim().toLowerCase();
  if (MOVEMENT_KIND_SET.has(normalized)) {
    return normalized as MovementKind;
  }
  // Map a few common synonyms onto the canonical vocabulary.
  switch (normalized) {
    case "key":
    case "hotkey":
      return "keypress";
    case "press":
    case "button":
      return "click";
    case "text":
    case "input":
      return "type";
    default:
      return undefined;
  }
}

function stringOrUndefined(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function directionOrUndefined(value: unknown): MovementDirection | undefined {
  return typeof value === "string" && MOVEMENT_DIRECTION_SET.has(value)
    ? (value as MovementDirection)
    : undefined;
}

/**
 * Derive a `MovementEvent` from a recorded trajectory action. Reads the rich
 * `metadata` the capture adapters attach (gesture/target/direction/valueSummary)
 * and falls back to the action's `tool` when no explicit gesture is present.
 * Returns `undefined` for actions that do not map to a physical movement.
 */
export function movementEventFromAction(action: TrajectoryAction): MovementEvent | undefined {
  const metadata = action.metadata ?? {};
  const gesture = typeof metadata.gesture === "string" ? metadata.gesture : undefined;
  const kind = normalizeMovementKind(gesture) ?? normalizeMovementKind(action.tool);
  if (!kind) {
    return undefined;
  }
  const event: MovementEvent = { ts: action.ts, kind };
  const target = stringOrUndefined(metadata.target);
  if (target) {
    event.target = target;
  }
  const direction = directionOrUndefined(metadata.direction);
  if (direction) {
    event.direction = direction;
  }
  const value = stringOrUndefined(metadata.valueSummary) ?? stringOrUndefined(metadata.value);
  if (value) {
    event.value = value;
  }
  return event;
}

/**
 * Build a training dataset from recorded trajectory spans. Each span becomes one
 * movement sequence (its actions, in timestamp order); spans that yield no
 * physical movements are dropped.
 */
export function buildMovementDataset(trajectories: TrajectorySpan[]): MovementDataset {
  const sequences: MovementSequence[] = [];
  for (const trajectory of trajectories) {
    const events = trajectory.actions
      .map((action) => movementEventFromAction(action))
      .filter((event): event is MovementEvent => event !== undefined)
      .sort((a, b) => a.ts - b.ts);
    if (events.length > 0) {
      sequences.push({ id: trajectory.id, events });
    }
  }
  return { version: 1, sequences };
}

export function countDatasetEvents(dataset: MovementDataset): number {
  return dataset.sequences.reduce((total, sequence) => total + sequence.events.length, 0);
}

// ---------------------------------------------------------------------------
// Pluggable model backend interfaces
// ---------------------------------------------------------------------------

export type MovementModelArtifact = {
  version: 1;
  backend: string;
  vocabulary: string[];
  sequenceCount: number;
  eventCount: number;
  trainedAt?: string;
  /** Backend-specific serialized parameters (e.g. transition tables, weights). */
  payload: unknown;
};

export type MovementPredictionAlternative = {
  token: string;
  probability: number;
};

export type MovementPrediction = {
  token: string;
  /** Reconstructed event, or `undefined` when the model predicts sequence end. */
  event?: MovementEvent;
  probability: number;
  /** How many context tokens actually matched — surfaces backoff for eval/debug. */
  contextOrderUsed: number;
  alternatives: MovementPredictionAlternative[];
};

export type MovementGenerateOptions = {
  maxSteps?: number;
};

/** A trained, in-process model that can repeat and generalise movements. */
export interface MovementModel {
  readonly backend: string;
  /** Predict the movement most likely to follow `context`. */
  predictNext(context: MovementEvent[]): MovementPrediction | undefined;
  /** Roll out a full movement sequence starting from `seed`. */
  generate(seed: MovementEvent[], options?: MovementGenerateOptions): MovementEvent[];
  toArtifact(): MovementModelArtifact;
}

export type MovementTrainOptions = {
  /** Maximum context length (n-gram order) the backend should model. */
  order?: number;
};

/** A pluggable training backend. Swap this to move from mock → real on-device. */
export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, options?: MovementTrainOptions): MovementModel;
  load(artifact: MovementModelArtifact): MovementModel;
}
