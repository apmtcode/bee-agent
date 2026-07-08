import type { TrajectoryAction, TrajectorySpan } from "../../capture/trajectory.js";
import type { ReplayTimelineEvent } from "../../capture/replay.js";
import type { MovementDataset, MovementSequence, MovementStep, MovementToken } from "./types.js";

/**
 * Derive a discrete, learnable token from a captured action. The token is
 * intentionally coarser than the free-text summary: it keys on the stable,
 * structural parts of a movement (tool, gesture kind, direction) so that
 * distinct-but-related movements over the same UI vocabulary collapse onto the
 * same symbol. This is what lets a trained model generalize across sessions
 * rather than memorizing verbatim summaries.
 */
export function movementTokenFromAction(action: TrajectoryAction): MovementToken {
  const metadata = action.metadata ?? {};
  const gesture = stringField(metadata.gesture);
  const direction = stringField(metadata.direction);
  const event = stringField(metadata.event);

  const parts: string[] = [normalizeSegment(action.tool)];
  if (gesture) {
    parts.push(normalizeSegment(gesture));
  } else if (event) {
    parts.push(normalizeSegment(event));
  }
  if (direction) {
    parts.push(normalizeSegment(direction));
  }
  return parts.filter(Boolean).join(".") || "action";
}

/** Build a full movement step (token + replayable structure) from an action. */
export function movementStepFromAction(action: TrajectoryAction): MovementStep {
  const metadata = action.metadata ?? {};
  const gesture = stringField(metadata.gesture) ?? stringField(metadata.event);
  const target = stringField(metadata.target) ?? stringField(metadata.filePath);
  const direction = stringField(metadata.direction);
  return {
    token: movementTokenFromAction(action),
    tool: action.tool,
    ts: action.ts,
    ...(gesture ? { gesture } : {}),
    ...(target ? { target } : {}),
    ...(direction ? { direction } : {}),
  };
}

/** Convert a captured trajectory span into an ordered movement sequence. */
export function movementSequenceFromTrajectory(trajectory: TrajectorySpan): MovementSequence {
  const steps = [...trajectory.actions]
    .sort((a, b) => a.ts - b.ts)
    .map(movementStepFromAction);
  return { id: trajectory.id, steps };
}

/**
 * Convert the action events of a replay manifest timeline into a movement
 * sequence. Non-action events (transcript, observation) are ignored — the model
 * learns the *movement* stream.
 */
export function movementSequenceFromReplayEvents(
  id: string,
  events: readonly ReplayTimelineEvent[],
): MovementSequence {
  const steps = events
    .filter((event): event is Extract<ReplayTimelineEvent, { kind: "action" }> => event.kind === "action")
    .slice()
    .sort((a, b) => a.ts - b.ts)
    .map<MovementStep>((event) => ({
      token: normalizeSegment(event.tool),
      tool: event.tool,
      ts: event.ts,
    }));
  return { id, steps };
}

/** Assemble a dataset from captured trajectories, dropping empty sequences. */
export function buildMovementDataset(trajectories: readonly TrajectorySpan[]): MovementDataset {
  const sequences = trajectories
    .map(movementSequenceFromTrajectory)
    .filter((sequence) => sequence.steps.length > 0);
  return { version: 1, sequences };
}

/** Distinct tokens present in a dataset, sorted for deterministic output. */
export function movementVocabulary(dataset: MovementDataset): MovementToken[] {
  const tokens = new Set<MovementToken>();
  for (const sequence of dataset.sequences) {
    for (const step of sequence.steps) {
      tokens.add(step.token);
    }
  }
  return [...tokens].sort();
}

function stringField(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function normalizeSegment(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}
