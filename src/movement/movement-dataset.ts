/**
 * Dataset builders for the local-movement learning subsystem.
 *
 * Converts recorded artifacts (replay manifests, trajectory spans, device
 * gestures) into the normalized `MovementDataset` the model backends train on,
 * and provides a deterministic synthetic event-stream generator so the
 * capture → dataset → train → replay round-trip is validatable without real OS
 * input (standing objective #2, cloud constraint).
 */

import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";
import type { MovementDataset, MovementSequence, MovementStep } from "./movement-model.js";

/**
 * Derive a normalized `MovementStep` from an `action` timeline event. The
 * replay event only carries `tool` + `summary`, so the summary is parsed into a
 * canonical action verb + target; richer sources (device gestures) attach
 * structured metadata via `movementStepFromMetadata`.
 */
export function movementStepFromActionEvent(
  event: Extract<ReplayTimelineEvent, { kind: "action" }>,
): MovementStep {
  const parsed = parseSummary(event.summary);
  return { tool: event.tool, action: parsed.action, ...(parsed.target ? { target: parsed.target } : {}) };
}

/** Build a movement step from a trajectory action, preferring structured metadata. */
export function movementStepFromTrajectoryAction(action: TrajectorySpan["actions"][number]): MovementStep {
  const metadata = action.metadata ?? {};
  const gesture = typeof metadata.gesture === "string" ? metadata.gesture : undefined;
  const parsed = parseSummary(action.summary);
  const step: MovementStep = {
    tool: action.tool,
    action: gesture ?? parsed.action,
  };
  const target = pickString(metadata.target) ?? parsed.target;
  if (target) {
    step.target = target;
  }
  const direction = pickDirection(metadata.direction);
  if (direction) {
    step.direction = direction;
  }
  const value = pickString(metadata.valueSummary);
  if (value) {
    step.value = value;
  }
  return step;
}

/** One dataset sequence per replay manifest (its ordered action events). */
export function movementSequenceFromReplay(replay: ReplayManifest): MovementSequence {
  const steps = replay.events
    .filter((event): event is Extract<ReplayTimelineEvent, { kind: "action" }> => event.kind === "action")
    .map(movementStepFromActionEvent);
  return { id: replay.sessionId, steps };
}

/** One dataset sequence per trajectory span (its ordered actions). */
export function movementSequenceFromTrajectory(trajectory: TrajectorySpan): MovementSequence {
  const steps = [...trajectory.actions]
    .sort((a, b) => a.ts - b.ts)
    .map(movementStepFromTrajectoryAction);
  return { id: trajectory.id, steps };
}

export function buildMovementDatasetFromReplays(replays: ReplayManifest[]): MovementDataset {
  return { sequences: replays.map(movementSequenceFromReplay).filter((sequence) => sequence.steps.length > 0) };
}

export function buildMovementDatasetFromTrajectories(trajectories: TrajectorySpan[]): MovementDataset {
  return {
    sequences: trajectories.map(movementSequenceFromTrajectory).filter((sequence) => sequence.steps.length > 0),
  };
}

/**
 * Deterministic synthetic movement generator. Emits repeatable, structured
 * sequences drawn from a small vocabulary of "flows" (e.g. open → navigate →
 * submit), seeded by an integer so runs are reproducible with no clock/RNG.
 * Used to validate the pipeline in the cloud and to build held-out eval splits.
 */
export function generateSyntheticMovementDataset(params: {
  sequenceCount: number;
  seed?: number;
  flows?: MovementStep[][];
}): MovementDataset {
  const flows = params.flows ?? DEFAULT_SYNTHETIC_FLOWS;
  const seed = Math.floor(params.seed ?? 1);
  const sequences: MovementSequence[] = [];
  for (let i = 0; i < Math.max(0, Math.floor(params.sequenceCount)); i += 1) {
    // Deterministic index mixing — no Math.random(), so cloud/CI is reproducible.
    const flowIndex = (seed * 2654435761 + i * 40503) % flows.length;
    const flow = flows[Math.abs(flowIndex)] ?? flows[0]!;
    sequences.push({ id: `synthetic-${seed}-${i}`, steps: flow.map((step) => ({ ...step })) });
  }
  return { sequences };
}

export const DEFAULT_SYNTHETIC_FLOWS: MovementStep[][] = [
  [
    { tool: "mouse", action: "click", target: "app-launcher" },
    { tool: "mouse", action: "click", target: "search-box" },
    { tool: "keyboard", action: "type", target: "search-box", value: "query" },
    { tool: "keyboard", action: "shortcut", target: "enter" },
    { tool: "mouse", action: "click", target: "first-result" },
  ],
  [
    { tool: "mouse", action: "click", target: "app-launcher" },
    { tool: "mouse", action: "click", target: "compose-button" },
    { tool: "keyboard", action: "type", target: "body", value: "message" },
    { tool: "mouse", action: "click", target: "send-button" },
  ],
  [
    { tool: "device", action: "tap", target: "app-launcher" },
    { tool: "device", action: "swipe", direction: "up" },
    { tool: "device", action: "tap", target: "settings" },
    { tool: "device", action: "swipe", direction: "down" },
  ],
];

function parseSummary(summary: string): { action: string; target?: string } {
  const trimmed = summary.trim().toLowerCase();
  const [verb, ...rest] = trimmed.split(/\s+/);
  const action = verb ?? "act";
  const remainder = rest.join(" ");
  // Strip common prepositions produced by the gesture summarizers.
  const target = remainder.replace(/^(?:into|to|on|the)\s+/, "").trim();
  return target ? { action, target } : { action };
}

function pickString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value : undefined;
}

function pickDirection(value: unknown): MovementStep["direction"] | undefined {
  return value === "up" || value === "down" || value === "left" || value === "right" ? value : undefined;
}
