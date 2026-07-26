import type { TrajectoryAction, TrajectoryObservation, TrajectorySpan } from "../../capture/trajectory.js";

/**
 * A discrete, replayable movement token derived from a recorded action.
 *
 * The token is intentionally coarse and human-readable (e.g. `device:tap:submit`)
 * so that (a) a small local model has a compact vocabulary to learn over and
 * (b) a predicted token maps cleanly back onto a replayable movement.
 */
export type MovementActionToken = string;

/**
 * One supervised training example: given the recent movement history plus the
 * ambient app/screen context, predict the next movement token.
 */
export type MovementSample = {
  trajectoryId: string;
  /** Position of the target action within the trajectory (0-based). */
  index: number;
  /** Preceding movement tokens, oldest first, most-recent last (length ≤ order). */
  context: MovementActionToken[];
  /** Ambient context the movement happened in (app/screen/source). */
  appContext: string;
  /** The movement token to predict. */
  action: MovementActionToken;
};

/**
 * A structured, replayable dataset the local-movement model trains on. This is
 * the on-disk-serialisable bridge between the capture pipeline and any model
 * backend.
 */
export type MovementDataset = {
  version: 1;
  /** Context window size used when the samples were generated. */
  order: number;
  samples: MovementSample[];
  /** Distinct movement tokens observed (sorted, stable). */
  vocabulary: MovementActionToken[];
  /** Distinct ambient contexts observed (sorted, stable). */
  contexts: string[];
};

export type BuildMovementDatasetOptions = {
  /** How many preceding movements to condition on. Defaults to 2. */
  order?: number;
  /**
   * Only include trajectories whose review status is approved. Defaults to
   * false so synthetic/unreviewed trajectories can be used in tests, but the
   * reviewed export pipeline should pass `true`.
   */
  requireApproved?: boolean;
};

const DEFAULT_ORDER = 2;

function slug(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 48) || "unknown";
}

function metadataString(metadata: Record<string, unknown> | undefined, key: string): string | undefined {
  const value = metadata?.[key];
  return typeof value === "string" && value.trim().length > 0 ? value : undefined;
}

/**
 * Reduce a recorded action to a canonical, replayable movement token. Prefers
 * structured gesture/target metadata (from the device/browser/os adapters) and
 * falls back to the human summary so no action is un-tokenisable.
 */
export function tokenizeAction(action: TrajectoryAction): MovementActionToken {
  const metadata = action.metadata;
  const gesture = metadataString(metadata, "gesture");
  const target = metadataString(metadata, "target");
  const direction = metadataString(metadata, "direction");

  const parts: string[] = [slug(action.tool)];
  if (gesture) {
    parts.push(slug(gesture));
  }
  if (target) {
    parts.push(slug(target));
  } else if (direction) {
    parts.push(slug(direction));
  }
  if (!gesture && !target && !direction) {
    parts.push(slug(action.summary));
  }
  return parts.join(":");
}

/**
 * Derive the ambient context a trajectory's movements happened in. Uses the
 * richest available observation signal (app name > source) so the model can
 * condition on "what app/screen am I in" rather than only the raw movement
 * history — this is what lets it generalise across trajectories in the same app.
 */
export function deriveAppContext(observations: TrajectoryObservation[]): string {
  for (const observation of observations) {
    const appName = metadataString(observation.metadata, "appName");
    if (appName) {
      return slug(appName);
    }
  }
  const first = observations[0];
  if (first) {
    return slug(first.source);
  }
  return "unknown";
}

function orderedActions(trajectory: TrajectorySpan): TrajectoryAction[] {
  return [...trajectory.actions].sort((a, b) => a.ts - b.ts);
}

/**
 * Convert reviewed/recorded trajectories into a sliding-window supervised
 * dataset. Each action becomes one prediction target conditioned on the
 * preceding `order` actions and the trajectory's ambient context.
 */
export function buildMovementDataset(
  trajectories: TrajectorySpan[],
  options: BuildMovementDatasetOptions = {},
): MovementDataset {
  const order = Math.max(0, options.order ?? DEFAULT_ORDER);
  const requireApproved = options.requireApproved ?? false;

  const samples: MovementSample[] = [];
  const vocabulary = new Set<MovementActionToken>();
  const contexts = new Set<string>();

  for (const trajectory of trajectories) {
    if (requireApproved && trajectory.review?.status !== "approved") {
      continue;
    }
    const appContext = deriveAppContext(trajectory.observations);
    contexts.add(appContext);

    const tokens = orderedActions(trajectory).map(tokenizeAction);
    for (let index = 0; index < tokens.length; index += 1) {
      const action = tokens[index]!;
      vocabulary.add(action);
      const start = Math.max(0, index - order);
      samples.push({
        trajectoryId: trajectory.id,
        index,
        context: tokens.slice(start, index),
        appContext,
        action,
      });
    }
  }

  return {
    version: 1,
    order,
    samples,
    vocabulary: [...vocabulary].sort(),
    contexts: [...contexts].sort(),
  };
}
