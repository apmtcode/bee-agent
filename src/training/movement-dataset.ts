import type { TrajectoryAction, TrajectoryObservation, TrajectorySpan } from "../capture/trajectory.js";

/**
 * Structured, replayable dataset for the local-movement learning subsystem.
 *
 * A {@link MovementDataset} is derived from recorded {@link TrajectorySpan}s and
 * expresses each recorded interaction as a `(context -> ordered steps)` example.
 * It is the trainable, on-device format consumed by a {@link MovementModelBackend}
 * (see `movement-model.ts`) so a small local model can learn to (c) repeat the
 * recorded movements and (d) generalize to new-but-related ones.
 *
 * The schema is intentionally backend-agnostic and JSON-serializable so it can be
 * written to disk, shipped to an on-device trainer, or validated in the cloud with
 * synthetic event streams.
 */

/** A normalized description of the situation a movement was performed in. */
export type MovementContext = {
  /** Application identifier the movement occurred in, when known. */
  appId?: string;
  /** Platform the movement occurred on, when known. */
  platform?: string;
  /** Foreground screen / window title, when known. */
  screenTitle?: string;
  /** Free-form goal or intent describing what the movement was for. */
  goal?: string;
  /**
   * Slot overrides supplied at inference time. Keys are step field names the
   * model induced as variable (`"target"`, `"valueSummary"`, `"direction"`); the
   * model substitutes these values when generalizing. Ignored during training.
   */
  slots?: Record<string, string>;
};

/** A single replayable movement step (one recorded action). */
export type MovementStep = {
  /** Tool / channel that performed the action (e.g. `"device"`, `"browser"`). */
  tool: string;
  /** Gesture kind when the action was a device/browser gesture. */
  gesture?: string;
  /** UI target the gesture acted on (button label, field name, element). */
  target?: string;
  /** Direction for swipe/scroll gestures. */
  direction?: string;
  /** Summary of any value entered/selected. */
  valueSummary?: string;
  /** Human-readable summary of the step. */
  summary: string;
};

/** One `(context -> steps)` training example derived from a trajectory. */
export type MovementExample = {
  id: string;
  sessionId: string;
  context: MovementContext;
  steps: MovementStep[];
};

/** The full trainable dataset. */
export type MovementDataset = {
  version: 1;
  examples: MovementExample[];
};

export type BuildMovementDatasetOptions = {
  /**
   * When true (default), only trajectories whose review status is `"approved"`
   * contribute examples — mirroring the reviewed-export gate used elsewhere in
   * the training pipeline. Set false for synthetic/simulated round-trip tests.
   */
  reviewedOnly?: boolean;
};

/**
 * Build a {@link MovementDataset} from recorded trajectory spans. Reads redacted
 * review data when present so exported training data never exceeds what a human
 * approved.
 */
export function buildMovementDataset(
  trajectories: TrajectorySpan[],
  options: BuildMovementDatasetOptions = {},
): MovementDataset {
  const reviewedOnly = options.reviewedOnly ?? true;
  const examples: MovementExample[] = [];

  for (const trajectory of trajectories) {
    if (reviewedOnly && trajectory.review?.status !== "approved") {
      continue;
    }

    const observations = resolveObservations(trajectory);
    const actions = resolveActions(trajectory);
    if (actions.length === 0) {
      continue;
    }

    examples.push({
      id: trajectory.id,
      sessionId: trajectory.sessionId,
      context: deriveContext(observations, trajectory),
      steps: actions.map(toMovementStep),
    });
  }

  return { version: 1, examples };
}

function resolveObservations(trajectory: TrajectorySpan): TrajectoryObservation[] {
  const redacted = trajectory.review?.redactedObservations;
  if (redacted) {
    return redacted.map((observation) => ({
      kind: "observation",
      ts: observation.ts,
      source: observation.source,
      summary: observation.summary,
    }));
  }
  return trajectory.observations;
}

function resolveActions(trajectory: TrajectorySpan): TrajectoryAction[] {
  const redacted = trajectory.review?.redactedActions;
  if (redacted) {
    return redacted.map((action) => ({
      kind: "action",
      ts: action.ts,
      tool: action.tool,
      summary: action.summary,
    }));
  }
  return trajectory.actions;
}

function deriveContext(observations: TrajectoryObservation[], trajectory: TrajectorySpan): MovementContext {
  const context: MovementContext = {};
  for (const observation of observations) {
    const metadata = observation.metadata ?? {};
    context.appId ??= readString(metadata.appId) ?? readString(metadata.appName);
    context.platform ??= readString(metadata.platform);
    context.screenTitle ??= readString(metadata.screenTitle);
  }
  const goal = trajectory.outcome?.summary ?? observations[0]?.summary;
  if (goal) {
    context.goal = goal;
  }
  return context;
}

function toMovementStep(action: TrajectoryAction): MovementStep {
  const metadata = action.metadata ?? {};
  const step: MovementStep = { tool: action.tool, summary: action.summary };
  const gesture = readString(metadata.gesture);
  const target = readString(metadata.target);
  const direction = readString(metadata.direction);
  const valueSummary = readString(metadata.valueSummary);
  if (gesture) {
    step.gesture = gesture;
  }
  if (target) {
    step.target = target;
  }
  if (direction) {
    step.direction = direction;
  }
  if (valueSummary) {
    step.valueSummary = valueSummary;
  }
  return step;
}

function readString(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}
