import type { ReplayTimelineEvent } from "../../capture/replay.js";

/**
 * A movement dataset is the training-ready view of reviewed replay manifests: the
 * ordered sequences of *action* tokens (one sequence per trajectory) that a local
 * movement-policy model learns to reproduce and generalize from. It is the bridge
 * between the capture/replay layer and the pluggable policy backends in this
 * directory, and is fully derivable from a replay manifest — no raw capture.
 */

/** A normalized, stable string identifying a single recorded action. */
export type MovementActionToken = string;

export type MovementStep = {
  /** Stable token used by the policy model (normalized `tool::summary`). */
  token: MovementActionToken;
  tool: string;
  summary: string;
  ts: number;
  /** Most recent observation summary seen before this action, if any. */
  observation?: string;
};

export type MovementSequence = {
  trajectoryId: string;
  sessionId: string;
  steps: MovementStep[];
};

export type MovementDataset = {
  version: 1;
  /** Sorted, de-duplicated action tokens observed across all sequences. */
  vocabulary: MovementActionToken[];
  sequences: MovementSequence[];
  stepCount: number;
};

/** Minimal structural shape shared by `ReplayManifest` and `ExportedReplayManifest`. */
export type MovementReplaySource = {
  sessionId: string;
  events: readonly ReplayTimelineEvent[];
};

function normalizeText(value: string): string {
  return value.trim().toLowerCase().replace(/\s+/g, " ");
}

/**
 * Tokenize a recorded action into a stable, comparable token. Two actions that
 * differ only in whitespace/case collapse to the same token so the policy model
 * treats them as the same movement.
 */
export function tokenizeMovementAction(tool: string, summary: string): MovementActionToken {
  return `${normalizeText(tool)}::${normalizeText(summary)}`;
}

/**
 * Build a movement dataset from one or more replay manifests. Events are grouped
 * into per-trajectory sequences; each action step is annotated with the most
 * recent preceding observation summary for that trajectory. Input event order is
 * respected (replay manifests are already sorted by timestamp), and sequences with
 * no actions are dropped.
 */
export function buildMovementDataset(replays: readonly MovementReplaySource[]): MovementDataset {
  const sequencesById = new Map<string, MovementSequence>();
  const lastObservationByTrajectory = new Map<string, string>();
  const order: string[] = [];

  for (const replay of replays) {
    for (const event of replay.events) {
      if (event.kind === "observation") {
        lastObservationByTrajectory.set(event.trajectoryId, event.summary);
        continue;
      }
      if (event.kind !== "action") {
        continue;
      }
      const key = `${replay.sessionId}${event.trajectoryId}`;
      let sequence = sequencesById.get(key);
      if (!sequence) {
        sequence = { trajectoryId: event.trajectoryId, sessionId: replay.sessionId, steps: [] };
        sequencesById.set(key, sequence);
        order.push(key);
      }
      const observation = lastObservationByTrajectory.get(event.trajectoryId);
      sequence.steps.push({
        token: tokenizeMovementAction(event.tool, event.summary),
        tool: event.tool,
        summary: event.summary,
        ts: event.ts,
        ...(observation ? { observation } : {}),
      });
    }
  }

  const sequences = order
    .map((key) => sequencesById.get(key))
    .filter((sequence): sequence is MovementSequence => sequence !== undefined && sequence.steps.length > 0);

  const vocabulary = [...new Set(sequences.flatMap((sequence) => sequence.steps.map((step) => step.token)))].sort();
  const stepCount = sequences.reduce((total, sequence) => total + sequence.steps.length, 0);

  return { version: 1, vocabulary, sequences, stepCount };
}
