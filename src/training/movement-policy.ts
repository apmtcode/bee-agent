import type { DeviceGestureKind, DevicePlatform } from "../capture/device-adapter.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Movement-policy inference layer for the local-movement learning subsystem.
 *
 * The capture pipeline records movements into {@link TrajectorySpan}s and the
 * training pipeline produces on-device model artifacts. This module closes the
 * loop: it turns a reviewed movement dataset into a servable *policy* that can
 * (a) repeat a recorded movement and (b) generalize to a new-but-related
 * movement by re-parameterizing the closest recorded trajectory.
 *
 * The model backend is pluggable ({@link MovementPolicyBackend}) so a real
 * on-device small model can be dropped in later; the shipped
 * {@link MockMovementPolicyBackend} is deterministic so cloud/CI tests pass
 * without any OS access.
 */

export type MovementDirection = "up" | "down" | "left" | "right";

/** A single atomic movement — the unit a policy predicts and a replayer executes. */
export type MovementStep = {
  gesture: DeviceGestureKind;
  appId: string;
  target?: string;
  direction?: MovementDirection;
  valueSummary?: string;
  /** Monotonic ordering hint within the trajectory (not wall-clock critical). */
  ts: number;
};

/** A labelled sequence of movement steps — one training/eval example. */
export type MovementTrajectory = {
  id: string;
  /** Natural-language-ish label used for nearest-neighbour retrieval. */
  goal: string;
  appId: string;
  platform?: DevicePlatform;
  steps: MovementStep[];
};

/** A request for a movement, optionally carrying overrides for generalization. */
export type MovementContext = {
  goal: string;
  appId?: string;
  /**
   * When set, the policy re-parameterizes the matched trajectory so it applies
   * to a new target/value/direction — this is how "type into A" generalizes to
   * "type into B" without a recorded example for B.
   */
  parameters?: {
    target?: string;
    valueSummary?: string;
    direction?: MovementDirection;
  };
};

export type MovementPrediction = {
  steps: MovementStep[];
  /** Id of the recorded trajectory the prediction was derived from, if any. */
  matchedTrajectoryId: string | null;
  /** Retrieval confidence in [0,1] (goal similarity of the matched trajectory). */
  confidence: number;
  /** True when the matched steps were re-parameterized via `context.parameters`. */
  generalized: boolean;
  backendId: string;
};

/** A fitted, ready-to-serve movement policy. */
export interface MovementPolicyModel {
  readonly backendId: string;
  readonly trajectoryCount: number;
  predict(context: MovementContext): MovementPrediction;
}

/** Pluggable backend that turns a movement dataset into a servable model. */
export interface MovementPolicyBackend {
  readonly id: string;
  fit(dataset: MovementTrajectory[]): MovementPolicyModel | Promise<MovementPolicyModel>;
}

export type TrajectoryToMovementOptions = {
  /** Only actions whose `tool` matches are treated as movements. Default `"device"`. */
  tool?: string;
  /** Derive a goal label. Defaults to the first observation summary, else the id. */
  goal?: (trajectory: TrajectorySpan) => string;
  /** Derive the appId. Defaults to action/observation metadata, else `"unknown"`. */
  appId?: (trajectory: TrajectorySpan) => string;
};

/**
 * Project a captured {@link TrajectorySpan} onto a {@link MovementTrajectory}.
 * Reads the gesture fields written by {@link DeviceCaptureAdapter} out of each
 * action's `metadata`. Returns `undefined` when the span carries no movements.
 */
export function trajectoryToMovement(
  trajectory: TrajectorySpan,
  options: TrajectoryToMovementOptions = {},
): MovementTrajectory | undefined {
  const tool = options.tool ?? "device";
  const appId = (options.appId ?? defaultAppId)(trajectory);
  const steps: MovementStep[] = [];

  for (const action of trajectory.actions) {
    if (action.tool !== tool) {
      continue;
    }
    const metadata = action.metadata ?? {};
    const gesture = asGesture(metadata.gesture);
    if (!gesture) {
      continue;
    }
    steps.push({
      gesture,
      appId,
      ...(asString(metadata.target) ? { target: asString(metadata.target) } : {}),
      ...(asDirection(metadata.direction) ? { direction: asDirection(metadata.direction) } : {}),
      ...(asString(metadata.valueSummary) ? { valueSummary: asString(metadata.valueSummary) } : {}),
      ts: action.ts,
    });
  }

  if (steps.length === 0) {
    return undefined;
  }

  return {
    id: trajectory.id,
    goal: (options.goal ?? defaultGoal)(trajectory),
    appId,
    steps,
  };
}

/**
 * Fits a {@link MovementPolicyBackend} over a dataset and serves predictions.
 * Backend-agnostic: swap the mock for a real on-device model without touching
 * call sites.
 */
export class MovementPolicyEngine {
  private model: MovementPolicyModel | undefined;

  constructor(private readonly backend: MovementPolicyBackend) {}

  get backendId(): string {
    return this.backend.id;
  }

  get fitted(): boolean {
    return this.model !== undefined;
  }

  async fit(dataset: MovementTrajectory[]): Promise<MovementPolicyModel> {
    this.model = await this.backend.fit(dataset);
    return this.model;
  }

  /** Fit directly from captured spans, projecting each via {@link trajectoryToMovement}. */
  async fitFromTrajectories(
    trajectories: TrajectorySpan[],
    options?: TrajectoryToMovementOptions,
  ): Promise<MovementPolicyModel> {
    const dataset = trajectories
      .map((trajectory) => trajectoryToMovement(trajectory, options))
      .filter((movement): movement is MovementTrajectory => movement !== undefined);
    return await this.fit(dataset);
  }

  predict(context: MovementContext): MovementPrediction {
    if (!this.model) {
      throw new Error("MovementPolicyEngine.predict called before fit()");
    }
    return this.model.predict(context);
  }
}

function defaultGoal(trajectory: TrajectorySpan): string {
  return trajectory.observations[0]?.summary ?? trajectory.id;
}

function defaultAppId(trajectory: TrajectorySpan): string {
  for (const action of trajectory.actions) {
    const value = asString(action.metadata?.appId);
    if (value) {
      return value;
    }
  }
  for (const observation of trajectory.observations) {
    const value = asString(observation.metadata?.appId) ?? asString(observation.metadata?.appName);
    if (value) {
      return value;
    }
  }
  return "unknown";
}

const GESTURE_VALUES: readonly DeviceGestureKind[] = ["tap", "swipe", "scroll", "type", "shortcut"];
const DIRECTION_VALUES: readonly MovementDirection[] = ["up", "down", "left", "right"];

function asGesture(value: unknown): DeviceGestureKind | undefined {
  return typeof value === "string" && (GESTURE_VALUES as readonly string[]).includes(value)
    ? (value as DeviceGestureKind)
    : undefined;
}

function asDirection(value: unknown): MovementDirection | undefined {
  return typeof value === "string" && (DIRECTION_VALUES as readonly string[]).includes(value)
    ? (value as MovementDirection)
    : undefined;
}

function asString(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}
