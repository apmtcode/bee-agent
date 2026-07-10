import type { ReplayTimelineEvent } from "../capture/replay.js";
import type { ExportedReplayManifest, ReviewedExportManifest } from "./export-manifest.js";

/**
 * Movement dataset schema.
 *
 * This is the bridge between the *capture* side of the local-movement learning
 * subsystem (replay timelines of transcript/observation/action events) and the
 * *training* side (a model that learns to reproduce, and generalize, recorded
 * movements).
 *
 * A replay timeline is a chronologically ordered stream of events. For learning
 * "what to do next", we reshape each replay into a set of supervised
 * transitions: given the bounded window of events that preceded an action, the
 * label is that action. A movement model conditions on the context window and
 * predicts the next action token — exactly the shape a small on-device
 * next-action model would consume.
 */

export type MovementActionToken = {
  tool: string;
  summary: string;
};

/** A normalized, order-preserving fingerprint of the events preceding an action. */
export type MovementContextSignature = string;

export type MovementTransition = {
  /** Sequence this transition belongs to. */
  sequenceId: string;
  /** Zero-based position of the action within its sequence's action list. */
  index: number;
  /** The full context signature (bounded window of preceding events). */
  context: MovementContextSignature;
  /** A coarser signature used for backoff/generalization to related contexts. */
  backoffContext: MovementContextSignature;
  /** The action taken in this context — the supervised label. */
  action: MovementActionToken;
};

export type MovementSequence = {
  id: string;
  sessionId: string;
  events: ReplayTimelineEvent[];
  transitions: MovementTransition[];
};

export type MovementDataset = {
  version: 1;
  contextWindow: number;
  sequences: MovementSequence[];
  transitionCount: number;
};

export type BuildMovementDatasetOptions = {
  /**
   * Number of immediately-preceding non-action events (observations/transcript)
   * to fold into an action's context signature. Defaults to 2.
   */
  contextWindow?: number;
};

const DEFAULT_CONTEXT_WINDOW = 2;

/** Stable key for an action token — used for vocabulary + argmax tie-breaking. */
export function movementActionKey(action: MovementActionToken): string {
  return `${action.tool}::${action.summary}`;
}

function normalizeToken(value: string): string {
  return value.trim().toLowerCase().replace(/\s+/g, " ");
}

function describeEvent(event: ReplayTimelineEvent): string {
  switch (event.kind) {
    case "transcript":
      return `msg:${event.role}:${normalizeToken(event.content)}`;
    case "observation":
      return `obs:${normalizeToken(event.source)}:${normalizeToken(event.summary)}`;
    case "action":
      return `act:${normalizeToken(event.tool)}`;
  }
}

/**
 * Build the context signatures for the action at `events[actionIndex]`.
 *
 * - `context` folds in the last `contextWindow` preceding events verbatim.
 * - `backoffContext` keeps only the single most recent event, so that a novel
 *   full-window context can still fall back to a coarser learned distribution.
 *   This is the mechanism by which the model generalizes to related-but-unseen
 *   situations (classic n-gram backoff).
 */
function contextSignaturesFor(
  events: ReplayTimelineEvent[],
  actionIndex: number,
  contextWindow: number,
): { context: MovementContextSignature; backoffContext: MovementContextSignature } {
  const preceding = events.slice(0, actionIndex);
  const windowed = preceding.slice(Math.max(0, preceding.length - contextWindow));
  const descriptors = windowed.map(describeEvent);
  const context = descriptors.length > 0 ? descriptors.join(" | ") : "<start>";
  const last = descriptors.at(-1);
  const backoffContext = last ?? "<start>";
  return { context, backoffContext };
}

function buildSequence(
  id: string,
  sessionId: string,
  events: ReplayTimelineEvent[],
  contextWindow: number,
): MovementSequence {
  const ordered = [...events].sort((a, b) => a.ts - b.ts);
  const transitions: MovementTransition[] = [];
  let actionIndex = 0;
  ordered.forEach((event, position) => {
    if (event.kind !== "action") {
      return;
    }
    const { context, backoffContext } = contextSignaturesFor(ordered, position, contextWindow);
    transitions.push({
      sequenceId: id,
      index: actionIndex,
      context,
      backoffContext,
      action: { tool: event.tool, summary: event.summary },
    });
    actionIndex += 1;
  });
  return { id, sessionId, events: ordered, transitions };
}

/** Build a movement dataset from raw replay manifests (the capture-side output). */
export function buildMovementDataset(
  replays: Array<Pick<ExportedReplayManifest, "sessionId" | "trajectoryIds" | "events">>,
  options: BuildMovementDatasetOptions = {},
): MovementDataset {
  const contextWindow = Math.max(1, options.contextWindow ?? DEFAULT_CONTEXT_WINDOW);
  const sequences = replays.map((replay, replayIndex) => {
    const id = replay.trajectoryIds[0] ?? `${replay.sessionId}#${replayIndex}`;
    return buildSequence(id, replay.sessionId, replay.events, contextWindow);
  });
  const transitionCount = sequences.reduce((total, sequence) => total + sequence.transitions.length, 0);
  return { version: 1, contextWindow, sequences, transitionCount };
}

/** Convenience: build a dataset directly from a reviewed export manifest. */
export function movementDatasetFromExport(
  manifest: Pick<ReviewedExportManifest, "replays">,
  options: BuildMovementDatasetOptions = {},
): MovementDataset {
  return buildMovementDataset(manifest.replays, options);
}

/**
 * Deterministically split a dataset into train/holdout partitions. Every
 * `holdoutEvery`-th sequence (1-based) is routed to the holdout set, so the
 * split is reproducible with no randomness (cloud/CI safe). Used by the
 * generalization eval to measure fidelity on sequences the model never trained
 * on.
 */
export function splitMovementDataset(
  dataset: MovementDataset,
  holdoutEvery = 3,
): { train: MovementDataset; holdout: MovementDataset } {
  const divisor = Math.max(2, Math.floor(holdoutEvery));
  const trainSequences: MovementSequence[] = [];
  const holdoutSequences: MovementSequence[] = [];
  dataset.sequences.forEach((sequence, index) => {
    if ((index + 1) % divisor === 0) {
      holdoutSequences.push(sequence);
    } else {
      trainSequences.push(sequence);
    }
  });
  return {
    train: toDataset(dataset.contextWindow, trainSequences),
    holdout: toDataset(dataset.contextWindow, holdoutSequences),
  };
}

function toDataset(contextWindow: number, sequences: MovementSequence[]): MovementDataset {
  return {
    version: 1,
    contextWindow,
    sequences,
    transitionCount: sequences.reduce((total, sequence) => total + sequence.transitions.length, 0),
  };
}
