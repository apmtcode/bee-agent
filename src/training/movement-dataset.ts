/**
 * Turns recorded capture artifacts into {@link MovementDataset}s that the
 * movement model backends can train on.
 *
 * Two sources are supported today:
 *  - {@link TrajectorySpan}s straight from the capture store (rich metadata).
 *  - {@link ReplayManifest}s produced by the replay builder (post-review,
 *    export-safe timelines).
 *
 * The tokenizer is intentionally low-cardinality and overridable: a movement is
 * reduced to `<tool>:<descriptor>` so the n-gram model sees repeated structure
 * (e.g. `device:tap:submit`) rather than free-form summaries.
 */
import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";
import type { ReplayManifest } from "../capture/replay.js";
import type { MovementDataset, MovementSequence, MovementToken } from "./movement-model.js";

export type MovementTokenSource = {
  tool: string;
  summary: string;
  metadata?: Record<string, unknown>;
};

export type MovementTokenizer = (source: MovementTokenSource) => MovementToken;

function readString(metadata: Record<string, unknown> | undefined, key: string): string | undefined {
  const value = metadata?.[key];
  return typeof value === "string" && value.trim().length > 0 ? value.trim() : undefined;
}

function firstWord(summary: string): string {
  const word = summary.trim().split(/\s+/u)[0];
  return (word ?? "action").toLowerCase();
}

/**
 * Default movement tokenizer. Prefers structured gesture metadata
 * (`gesture` + `direction`/`target`) and falls back to the leading verb of the
 * human summary, keeping token cardinality low enough for n-gram learning.
 */
export function defaultMovementTokenizer(source: MovementTokenSource): MovementToken {
  const gesture = readString(source.metadata, "gesture");
  const direction = readString(source.metadata, "direction");
  const target = readString(source.metadata, "target");
  const descriptor = gesture
    ? [gesture, direction ?? target].filter((part): part is string => Boolean(part)).join(":")
    : firstWord(source.summary);
  const tool = source.tool.trim() || "action";
  return `${tool}:${descriptor}`;
}

export function tokenizeAction(action: TrajectoryAction, tokenizer: MovementTokenizer = defaultMovementTokenizer): MovementToken {
  return tokenizer({ tool: action.tool, summary: action.summary, metadata: action.metadata });
}

export function movementSequenceFromTrajectory(
  trajectory: TrajectorySpan,
  tokenizer: MovementTokenizer = defaultMovementTokenizer,
): MovementSequence {
  const tokens = [...trajectory.actions]
    .sort((a, b) => a.ts - b.ts)
    .map((action) => tokenizeAction(action, tokenizer));
  return { id: trajectory.id, tokens };
}

export function movementDatasetFromTrajectories(
  trajectories: TrajectorySpan[],
  tokenizer: MovementTokenizer = defaultMovementTokenizer,
): MovementDataset {
  return {
    version: 1,
    sequences: trajectories
      .map((trajectory) => movementSequenceFromTrajectory(trajectory, tokenizer))
      .filter((sequence) => sequence.tokens.length > 0),
  };
}

export function movementDatasetFromReplays(
  replays: ReplayManifest[],
  tokenizer: MovementTokenizer = defaultMovementTokenizer,
): MovementDataset {
  const sequences: MovementSequence[] = [];
  for (const replay of replays) {
    const grouped = new Map<string, { ts: number; token: MovementToken }[]>();
    for (const event of replay.events) {
      if (event.kind !== "action") {
        continue;
      }
      const token = tokenizer({ tool: event.tool, summary: event.summary });
      const bucket = grouped.get(event.trajectoryId);
      if (bucket) {
        bucket.push({ ts: event.ts, token });
      } else {
        grouped.set(event.trajectoryId, [{ ts: event.ts, token }]);
      }
    }
    for (const [trajectoryId, entries] of grouped) {
      const tokens = entries.sort((a, b) => a.ts - b.ts).map((entry) => entry.token);
      if (tokens.length > 0) {
        sequences.push({ id: `${replay.sessionId}:${trajectoryId}`, tokens });
      }
    }
  }
  return { version: 1, sequences };
}
