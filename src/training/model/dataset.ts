/**
 * Builds {@link MovementDataset}s from captured trajectories and replay manifests so
 * the movement-model backends can train on the same reviewed data the training
 * exporter already produces. This is the bridge between the capture pipeline
 * (objective #2 a/b) and the model layer (objective #2 c/d).
 */
import type { TrajectorySpan } from "../../capture/trajectory.js";
import type { ReplayManifest } from "../../capture/replay.js";
import type { MovementDataset, MovementEvent, MovementSequence } from "./movement-model.js";

/** Convert one trajectory span (using reviewed/redacted views when present) to a sequence. */
export function trajectoryToMovementSequence(trajectory: TrajectorySpan): MovementSequence {
  const observations = trajectory.review?.redactedObservations
    ? trajectory.review.redactedObservations.map<MovementEvent>((observation) => ({
        kind: "observation",
        source: observation.source,
        summary: observation.summary,
        ts: observation.ts,
      }))
    : trajectory.observations.map<MovementEvent>((observation) => ({
        kind: "observation",
        source: observation.source,
        summary: observation.summary,
        ts: observation.ts,
      }));

  const actions = trajectory.review?.redactedActions
    ? trajectory.review.redactedActions.map<MovementEvent>((action) => ({
        kind: "action",
        tool: action.tool,
        summary: action.summary,
        ts: action.ts,
      }))
    : trajectory.actions.map<MovementEvent>((action) => ({
        kind: "action",
        tool: action.tool,
        summary: action.summary,
        ts: action.ts,
      }));

  const events = [...observations, ...actions].sort((a, b) => a.ts - b.ts);

  return {
    trajectoryId: trajectory.id,
    events,
    ...(trajectory.outcome ? { outcome: trajectory.outcome.status } : {}),
    ...(trajectory.outcome?.reward !== undefined ? { reward: trajectory.outcome.reward } : {}),
  };
}

/** Build a dataset from a set of trajectory spans (skips empty sequences). */
export function buildMovementDataset(trajectories: TrajectorySpan[]): MovementDataset {
  const sequences = trajectories
    .map(trajectoryToMovementSequence)
    .filter((sequence) => sequence.events.length > 0);
  return { version: 1, sequences };
}

/**
 * Build a dataset from replay manifests (the exporter's reviewed-export format).
 * Transcript events are ignored — only observation/action movements are modelled.
 */
export function buildMovementDatasetFromReplays(replays: ReplayManifest[]): MovementDataset {
  const byTrajectory = new Map<string, MovementEvent[]>();
  for (const replay of replays) {
    for (const event of replay.events) {
      if (event.kind === "transcript") {
        continue;
      }
      const list = byTrajectory.get(event.trajectoryId) ?? [];
      if (event.kind === "action") {
        list.push({ kind: "action", tool: event.tool, summary: event.summary, ts: event.ts });
      } else {
        list.push({ kind: "observation", source: event.source, summary: event.summary, ts: event.ts });
      }
      byTrajectory.set(event.trajectoryId, list);
    }
  }

  const sequences: MovementSequence[] = [];
  for (const [trajectoryId, events] of byTrajectory) {
    if (events.length === 0) {
      continue;
    }
    events.sort((a, b) => a.ts - b.ts);
    sequences.push({ trajectoryId, events });
  }
  sequences.sort((a, b) => a.trajectoryId.localeCompare(b.trajectoryId));

  return { version: 1, sequences };
}
