import type { TrajectoryAction, TrajectoryObservation, TrajectorySpan } from "../capture/trajectory.js";

/**
 * Movement-learning dataset.
 *
 * This is the bridge between the capture pipeline (`src/capture`) and the
 * local-movement model backend (`movement-backend.ts`). A {@link MovementDataset}
 * is a flat list of supervised `(context -> action)` examples derived from
 * reviewed trajectory spans: for every recorded action we attach the most
 * recent preceding observation as its context. A backend trains on these pairs
 * and learns to predict the next movement given a fresh context.
 *
 * The dataset is deliberately backend-agnostic and serializable so it can be
 * exported to disk, fed to an on-device trainer, or consumed by the in-process
 * deterministic mock backend used in the cloud/CI.
 */

export type MovementContext = {
  /** Application identifier the movement happened in, when known. */
  app?: string;
  /** Human-readable screen / window title, when known. */
  screen?: string;
  /** Observation source channel (e.g. "device", "browser", "os"). */
  source?: string;
  /** Free-text observation summary — the primary signal for similarity. */
  summary: string;
};

export type MovementAction = {
  /** Tool/channel that performed the action (e.g. "device", "browser"). */
  tool: string;
  /** Structured gesture, when the action carried one (tap/swipe/scroll/...). */
  gesture?: string;
  /** UI element / target the action was applied to. */
  target?: string;
  /** Direction for directional gestures (up/down/left/right). */
  direction?: string;
  /** Redacted summary of any value that was entered. */
  valueSummary?: string;
  /** Human-readable summary of the action. */
  summary: string;
};

export type MovementExample = {
  trajectoryId: string;
  /** Index of the action within its trajectory (stable ordering key). */
  index: number;
  ts: number;
  context: MovementContext;
  action: MovementAction;
};

export type MovementDataset = {
  version: 1;
  examples: MovementExample[];
  /** Sorted vocabulary of apps seen in contexts. */
  apps: string[];
  /** Sorted vocabulary of tools seen in actions. */
  tools: string[];
  /** Sorted vocabulary of gestures seen in actions. */
  gestures: string[];
  /** Sorted vocabulary of targets seen in actions (the generalization slots). */
  targets: string[];
};

export type BuildMovementDatasetOptions = {
  /**
   * When true (default), only trajectories whose review status is "approved"
   * contribute examples — mirroring the training exporter's consent gate. Set
   * false for synthetic/test datasets that carry no review metadata.
   */
  requireApproved?: boolean;
};

function actionFromTrajectoryAction(action: TrajectoryAction): MovementAction {
  const metadata = action.metadata ?? {};
  const gesture = typeof metadata.gesture === "string" ? metadata.gesture : undefined;
  const target = typeof metadata.target === "string" ? metadata.target : undefined;
  const direction = typeof metadata.direction === "string" ? metadata.direction : undefined;
  const valueSummary = typeof metadata.valueSummary === "string" ? metadata.valueSummary : undefined;
  return {
    tool: action.tool,
    summary: action.summary,
    ...(gesture ? { gesture } : {}),
    ...(target ? { target } : {}),
    ...(direction ? { direction } : {}),
    ...(valueSummary ? { valueSummary } : {}),
  };
}

function contextFromObservation(observation: TrajectoryObservation | undefined): MovementContext {
  if (!observation) {
    return { summary: "" };
  }
  const metadata = observation.metadata ?? {};
  const app = typeof metadata.appName === "string" ? metadata.appName : undefined;
  const screen = typeof metadata.screenTitle === "string" ? metadata.screenTitle : undefined;
  return {
    summary: observation.summary,
    source: observation.source,
    ...(app ? { app } : {}),
    ...(screen ? { screen } : {}),
  };
}

/**
 * Derive a supervised movement dataset from trajectory spans. Each action is
 * paired with the latest observation (by timestamp) that precedes it in the
 * same trajectory; if none exists the context falls back to the first
 * observation, then to an empty context.
 */
export function buildMovementDataset(
  trajectories: TrajectorySpan[],
  options: BuildMovementDatasetOptions = {},
): MovementDataset {
  const requireApproved = options.requireApproved ?? true;
  const examples: MovementExample[] = [];
  const apps = new Set<string>();
  const tools = new Set<string>();
  const gestures = new Set<string>();
  const targets = new Set<string>();

  for (const trajectory of trajectories) {
    if (requireApproved && trajectory.review?.status !== "approved") {
      continue;
    }
    const observations = [...trajectory.observations].sort((a, b) => a.ts - b.ts);
    const actions = [...trajectory.actions].sort((a, b) => a.ts - b.ts);

    actions.forEach((action, index) => {
      const preceding = observations.filter((observation) => observation.ts <= action.ts);
      const observation = preceding.length > 0 ? preceding[preceding.length - 1] : observations[0];
      const context = contextFromObservation(observation);
      const movementAction = actionFromTrajectoryAction(action);

      if (context.app) {
        apps.add(context.app);
      }
      tools.add(movementAction.tool);
      if (movementAction.gesture) {
        gestures.add(movementAction.gesture);
      }
      if (movementAction.target) {
        targets.add(movementAction.target);
      }

      examples.push({
        trajectoryId: trajectory.id,
        index,
        ts: action.ts,
        context,
        action: movementAction,
      });
    });
  }

  return {
    version: 1,
    examples,
    apps: [...apps].sort(),
    tools: [...tools].sort(),
    gestures: [...gestures].sort(),
    targets: [...targets].sort(),
  };
}

const TOKEN_SPLIT = /[^a-z0-9]+/i;

/** Lowercase, de-duplicated word tokens for similarity scoring. */
export function tokenizeMovementText(...parts: Array<string | undefined>): string[] {
  const tokens = new Set<string>();
  for (const part of parts) {
    if (!part) {
      continue;
    }
    for (const raw of part.split(TOKEN_SPLIT)) {
      const token = raw.trim().toLowerCase();
      if (token.length > 0) {
        tokens.add(token);
      }
    }
  }
  return [...tokens];
}
