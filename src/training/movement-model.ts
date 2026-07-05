// Local-movement learning subsystem — model layer.
//
// This module defines the *pluggable* movement-model interface plus the shared
// data structures used to (a) turn recorded movement/action events into a
// canonical token stream, (b) assemble a replayable training dataset, and
// (c) train/infer with a backend. The concrete deterministic in-process backend
// lives in `markov-backend.ts`; a real on-device small model can be dropped in
// by implementing `MovementModelBackend` and registering it in
// `createMovementBackend`.
//
// Everything here is pure and OS-free so it runs and is fully testable in the
// cloud against synthetic/simulated event streams. The actual on-device
// recording and training happens when the user runs bee-agent locally.

import type { TrajectorySpan } from "../capture/trajectory.js";
import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import type { ExportedReplayManifest } from "./export-manifest.js";
import { MarkovMovementBackend } from "./markov-backend.js";

/** A single canonical movement token, e.g. `device:tap:button`. */
export type MovementToken = string;

/** Sentinel marking the start of a movement sequence (never predicted). */
export const MOVEMENT_START_TOKEN: MovementToken = "<start>";
/** Sentinel marking the end of a movement sequence (a valid prediction). */
export const MOVEMENT_END_TOKEN: MovementToken = "<end>";

/** One recorded movement sequence: an ordered list of action tokens. */
export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
  /** Optional provenance for auditability (session/trajectory it came from). */
  source?: string;
};

/** A replayable training dataset: sequences plus the derived vocabulary. */
export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
  vocabulary: MovementToken[];
};

/** Minimal shape needed to derive a movement token from an action. */
export type TokenizableAction = {
  tool: string;
  summary: string;
  metadata?: Record<string, unknown>;
};

export type MovementTokenizerOptions = {
  /** Include a coarse target class in the token (default true). */
  includeTarget?: boolean;
};

/** A single-token prediction with its probability and ranked alternatives. */
export type MovementPrediction = {
  token: MovementToken;
  probability: number;
  alternatives: Array<{ token: MovementToken; probability: number }>;
};

export type MovementGenerateParams = {
  /** Priming context; defaults to `[MOVEMENT_START_TOKEN]`. */
  seed?: MovementToken[];
  /** Hard cap on generated tokens (default 64) — prevents runaway cycles. */
  maxSteps?: number;
  /** Stop (and omit) when the end sentinel is produced (default true). */
  stopAtEnd?: boolean;
};

/** A trained, ready-to-infer movement model. Deterministic by contract. */
export interface TrainedMovementModel {
  /** Identifier of the backend that produced this model. */
  readonly backend: string;
  /** Argmax next-token prediction for `context`; undefined if untrained. */
  predictNext(context: MovementToken[]): MovementPrediction | undefined;
  /** Greedily generate a movement sequence (excludes seed and end sentinel). */
  generate(params?: MovementGenerateParams): MovementToken[];
  /** Mean log-probability the model assigns to `tokens` (higher = better fit). */
  scoreSequence(tokens: MovementToken[]): number;
  /** Serialize to a plain JSON value for persistence. */
  toJSON(): unknown;
}

/** A pluggable movement-model backend (train + load). */
export interface MovementModelBackend {
  readonly kind: string;
  train(dataset: MovementDataset): TrainedMovementModel;
  load(serialized: unknown): TrainedMovementModel;
}

const HEX_OR_ID_SUFFIX = /[-_]?(?:[0-9]+|[0-9a-f]{6,})$/i;

/**
 * Collapse a raw target/direction into a coarse, generalization-friendly class:
 * lowercase, first word only, and strip trailing numeric / hex id suffixes so
 * `field-3` and `field-7` both become `field`.
 */
export function normalizeTargetClass(raw: string | undefined): string | undefined {
  if (!raw) {
    return undefined;
  }
  const firstWord = raw.trim().toLowerCase().split(/\s+/)[0];
  if (!firstWord) {
    return undefined;
  }
  const stripped = firstWord.replace(HEX_OR_ID_SUFFIX, "");
  return stripped.length > 0 ? stripped : firstWord;
}

/**
 * Derive a stable, coarse movement token from an action event. The token
 * captures the tool, the gesture/verb, and (optionally) a normalized target
 * class — deliberately dropping specific ids so the model generalizes across
 * concrete instances of the same movement.
 */
export function movementTokenFromAction(
  action: TokenizableAction,
  options: MovementTokenizerOptions = {},
): MovementToken {
  const includeTarget = options.includeTarget ?? true;
  const tool = normalizeSlug(action.tool) || "tool";
  const metadata = action.metadata ?? {};
  const gesture = typeof metadata.gesture === "string" ? metadata.gesture : undefined;
  const verb = normalizeSlug(gesture ?? firstWord(action.summary)) || "act";

  const parts: string[] = [tool, verb];
  if (includeTarget) {
    const rawTarget =
      pickString(metadata.target) ??
      pickString(metadata.direction) ??
      pickString(metadata.valueSummary) ??
      targetFromSummary(action.summary);
    const targetClass = normalizeTargetClass(rawTarget);
    if (targetClass) {
      parts.push(targetClass);
    }
  }
  return parts.join(":");
}

function pickString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim().length > 0 ? value : undefined;
}

function firstWord(text: string): string {
  return text.trim().split(/\s+/)[0] ?? "";
}

/** Best-effort target extraction from a free-text summary ("tapped Submit"). */
function targetFromSummary(summary: string): string | undefined {
  const words = summary.trim().split(/\s+/);
  return words.length > 1 ? words.slice(1).join(" ") : undefined;
}

function normalizeSlug(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function sortedVocabulary(sequences: MovementSequence[]): MovementToken[] {
  const set = new Set<MovementToken>();
  for (const sequence of sequences) {
    for (const token of sequence.tokens) {
      set.add(token);
    }
  }
  set.add(MOVEMENT_END_TOKEN);
  return [...set].sort();
}

/** Build a dataset from reviewed trajectory spans (one sequence per span). */
export function buildMovementDatasetFromTrajectories(
  trajectories: TrajectorySpan[],
  options: MovementTokenizerOptions = {},
): MovementDataset {
  const sequences: MovementSequence[] = trajectories.map((trajectory) => {
    const tokens = [...trajectory.actions]
      .sort((a, b) => a.ts - b.ts)
      .map((action) => movementTokenFromAction(action, options));
    return { id: trajectory.id, tokens, source: `trajectory:${trajectory.id}` };
  });
  return { version: 1, sequences, vocabulary: sortedVocabulary(sequences) };
}

type AnyReplayManifest = ReplayManifest | ExportedReplayManifest;

/** Build a dataset from replay manifests (one sequence per replay's actions). */
export function buildMovementDatasetFromReplays(
  replays: AnyReplayManifest[],
  options: MovementTokenizerOptions = {},
): MovementDataset {
  const sequences: MovementSequence[] = replays.map((replay, index) => {
    const actions = replay.events.filter(
      (event): event is Extract<ReplayTimelineEvent, { kind: "action" }> => event.kind === "action",
    );
    const tokens = [...actions]
      .sort((a, b) => a.ts - b.ts)
      .map((action) => movementTokenFromAction({ tool: action.tool, summary: action.summary }, options));
    return {
      id: `${replay.sessionId}:${index}`,
      tokens,
      source: `session:${replay.sessionId}`,
    };
  });
  return { version: 1, sequences, vocabulary: sortedVocabulary(sequences) };
}

/**
 * Deterministically split a dataset by index (no randomness, so cloud/CI runs
 * are reproducible): every `holdoutEvery`-th sequence goes to `holdout`.
 */
export function splitMovementDataset(
  dataset: MovementDataset,
  holdoutEvery = 3,
): { train: MovementDataset; holdout: MovementDataset } {
  const trainSeq: MovementSequence[] = [];
  const holdoutSeq: MovementSequence[] = [];
  dataset.sequences.forEach((sequence, index) => {
    if (holdoutEvery > 0 && (index + 1) % holdoutEvery === 0) {
      holdoutSeq.push(sequence);
    } else {
      trainSeq.push(sequence);
    }
  });
  return {
    train: { version: 1, sequences: trainSeq, vocabulary: sortedVocabulary(trainSeq) },
    holdout: { version: 1, sequences: holdoutSeq, vocabulary: sortedVocabulary(holdoutSeq) },
  };
}

// --- Pluggable backend registry -------------------------------------------

export type MovementBackendKind = "markov";

export type CreateMovementBackendOptions = {
  /** Markov order (context length). Default 2. */
  order?: number;
  /** Additive (Laplace) smoothing weight. Default 0.1. */
  alpha?: number;
};

/**
 * Instantiate a movement-model backend by kind. This is the seam for dropping
 * in a real on-device small model later — add a case here that returns your
 * `MovementModelBackend` implementation.
 */
export function createMovementBackend(
  kind: MovementBackendKind = "markov",
  options: CreateMovementBackendOptions = {},
): MovementModelBackend {
  switch (kind) {
    case "markov":
      return new MarkovMovementBackend({ order: options.order, alpha: options.alpha });
    default: {
      const exhaustive: never = kind;
      throw new Error(`unknown movement backend: ${String(exhaustive)}`);
    }
  }
}
