import type { TrajectoryAction, TrajectoryObservation, TrajectorySpan } from "../capture/trajectory.js";

/**
 * Local-movement learning: turn recorded {@link TrajectorySpan}s into a
 * supervised (context -> action) dataset that a local policy model can be
 * post-trained on, so bee-agent can (a) repeat recorded movements and
 * (b) generalize to new-but-related movements.
 *
 * The extraction is intentionally metadata-driven and deterministic: it reads
 * only the fields the capture adapters already populate (device/os/browser),
 * never invents ordering (events are sorted by `ts`, ties broken by kind), and
 * emits one training example per recorded action conditioned on the context
 * that preceded it. No current-action field ever leaks into the context.
 */

export type MovementContext = {
  /** Foreground app for the step, when the capture layer recorded one. */
  appName?: string;
  /** Source of the most recent preceding observation (e.g. "device", "os"). */
  observationSource?: string;
  /** Human summary of the most recent preceding observation. */
  observationSummary?: string;
  /** Tool of the immediately preceding action in the same trajectory. */
  lastActionTool?: string;
  /** Summary of the immediately preceding action in the same trajectory. */
  lastActionSummary?: string;
  /** 0-based index of this action within its trajectory. */
  stepIndex: number;
};

export type MovementActionLabel = {
  tool: string;
  summary: string;
  gesture?: string;
  target?: string;
  direction?: string;
};

export type MovementExample = {
  trajectoryId: string;
  context: MovementContext;
  action: MovementActionLabel;
};

export type MovementDataset = {
  version: 1;
  exampleCount: number;
  examples: MovementExample[];
};

export type BuildMovementDatasetOptions = {
  /** Only include trajectories whose review status is "approved". */
  requireApprovedReview?: boolean;
};

type TimelineEntry =
  | { kind: "observation"; ts: number; observation: TrajectoryObservation }
  | { kind: "action"; ts: number; action: TrajectoryAction };

export function buildMovementDataset(
  trajectories: TrajectorySpan[],
  options: BuildMovementDatasetOptions = {},
): MovementDataset {
  const examples: MovementExample[] = [];

  for (const trajectory of trajectories) {
    if (options.requireApprovedReview && trajectory.review?.status !== "approved") {
      continue;
    }

    const timeline = buildTimeline(trajectory);
    let lastObservation: TrajectoryObservation | undefined;
    let lastAction: TrajectoryAction | undefined;
    let stepIndex = 0;

    for (const entry of timeline) {
      if (entry.kind === "observation") {
        lastObservation = entry.observation;
        continue;
      }

      examples.push({
        trajectoryId: trajectory.id,
        context: buildContext(lastObservation, lastAction, stepIndex),
        action: buildActionLabel(entry.action),
      });
      lastAction = entry.action;
      stepIndex += 1;
    }
  }

  return { version: 1, exampleCount: examples.length, examples };
}

function buildTimeline(trajectory: TrajectorySpan): TimelineEntry[] {
  const entries: TimelineEntry[] = [
    ...trajectory.observations.map<TimelineEntry>((observation) => ({
      kind: "observation",
      ts: observation.ts,
      observation,
    })),
    ...trajectory.actions.map<TimelineEntry>((action) => ({
      kind: "action",
      ts: action.ts,
      action,
    })),
  ];
  // Stable sort by timestamp; observations before actions at the same tick so an
  // action is always conditioned on the observation that describes its screen.
  return entries.sort((a, b) => (a.ts !== b.ts ? a.ts - b.ts : kindOrder(a.kind) - kindOrder(b.kind)));
}

function kindOrder(kind: TimelineEntry["kind"]): number {
  return kind === "observation" ? 0 : 1;
}

function buildContext(
  observation: TrajectoryObservation | undefined,
  lastAction: TrajectoryAction | undefined,
  stepIndex: number,
): MovementContext {
  const context: MovementContext = { stepIndex };
  const appName = readString(observation?.metadata, "appName");
  if (appName !== undefined) {
    context.appName = appName;
  }
  if (observation) {
    context.observationSource = observation.source;
    context.observationSummary = observation.summary;
  }
  if (lastAction) {
    context.lastActionTool = lastAction.tool;
    context.lastActionSummary = lastAction.summary;
  }
  return context;
}

function buildActionLabel(action: TrajectoryAction): MovementActionLabel {
  const label: MovementActionLabel = { tool: action.tool, summary: action.summary };
  const gesture = readString(action.metadata, "gesture");
  const target = readString(action.metadata, "target");
  const direction = readString(action.metadata, "direction");
  if (gesture !== undefined) {
    label.gesture = gesture;
  }
  if (target !== undefined) {
    label.target = target;
  }
  if (direction !== undefined) {
    label.direction = direction;
  }
  return label;
}

function readString(metadata: Record<string, unknown> | undefined, key: string): string | undefined {
  const value = metadata?.[key];
  return typeof value === "string" ? value : undefined;
}
