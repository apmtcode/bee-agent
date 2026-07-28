// Bridge from captured trajectories to a movement dataset.
//
// Turns the capture subsystem's `TrajectorySpan[]` (produced by the device /
// os / browser adapters) into the `MovementDataset` the model backends consume.
// This is the "dataset" stage of the capture → schema → dataset → replay →
// train/infer pipeline: it derives canonical action tokens from recorded
// actions and groups steps into per-trajectory demonstrations.

import type { DevicePlatform } from "../capture/device-adapter.js";
import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";
import type { MovementContext, MovementDataset, MovementStep, MovementTrajectory } from "./movement-model.js";

export type BuildMovementDatasetOptions = {
  /** Platform to stamp on derived contexts. Default "macos". */
  platform?: DevicePlatform;
  /** Only include trajectories whose review status is approved. Default false. */
  approvedOnly?: boolean;
};

/**
 * Derive a canonical action token from a recorded action. Direction and gesture
 * kind are folded into the token so the model conditions on them; falls back to
 * the tool name.
 */
export function deriveActionToken(action: TrajectoryAction): string {
  const meta = action.metadata ?? {};
  const kind = typeof meta.gesture === "string" ? meta.gesture : typeof meta.kind === "string" ? meta.kind : undefined;
  const direction = typeof meta.direction === "string" ? meta.direction : undefined;
  const base = kind ?? action.tool;
  return direction ? `${base}:${direction}` : base;
}

function deriveTarget(action: TrajectoryAction): string | undefined {
  const meta = action.metadata ?? {};
  const target = typeof meta.target === "string" ? meta.target : undefined;
  return target && target.trim().length > 0 ? target : undefined;
}

function deriveContext(span: TrajectorySpan, platform: DevicePlatform): MovementContext {
  const meta = span.observations[0]?.metadata ?? {};
  const appId = typeof meta.appName === "string" ? meta.appName : span.observations[0]?.source ?? "unknown-app";
  const screen = typeof meta.screenTitle === "string" ? meta.screenTitle : typeof meta.windowTitle === "string" ? meta.windowTitle : undefined;
  const resolvedPlatform = typeof meta.platform === "string" && isPlatform(meta.platform) ? meta.platform : platform;
  return {
    platform: resolvedPlatform,
    appId,
    ...(screen ? { screen } : {}),
  };
}

function isPlatform(value: string): value is DevicePlatform {
  return value === "ios" || value === "android" || value === "macos" || value === "windows" || value === "linux";
}

/** Build a movement dataset from captured trajectory spans. */
export function buildMovementDataset(
  spans: TrajectorySpan[],
  options: BuildMovementDatasetOptions = {},
): MovementDataset {
  const platform = options.platform ?? "macos";
  const trajectories: MovementTrajectory[] = [];

  for (const span of spans) {
    if (options.approvedOnly && span.review?.status !== "approved") {
      continue;
    }
    if (span.actions.length === 0) {
      continue;
    }
    const steps: MovementStep[] = span.actions
      .slice()
      .sort((a, b) => a.ts - b.ts)
      .map((action) => {
        const target = deriveTarget(action);
        return {
          action: deriveActionToken(action),
          ...(target ? { target } : {}),
          ts: action.ts,
        };
      });
    trajectories.push({ id: span.id, context: deriveContext(span, platform), steps });
  }

  return { version: 1, trajectories };
}
