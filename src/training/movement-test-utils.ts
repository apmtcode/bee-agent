import type { TrajectoryAction, TrajectoryObservation, TrajectorySpan } from "../capture/trajectory.js";

/**
 * Synthetic event-stream helper for validating the movement subsystem without
 * real OS input. Produces a {@link TrajectorySpan} shaped exactly like the
 * output of `DeviceCaptureAdapter` so capture -> dataset -> train -> infer
 * round-trips can be exercised deterministically in the cloud.
 */
export type SyntheticGesture = {
  kind: string;
  target?: string;
  direction?: string;
  valueSummary?: string;
};

export type SyntheticDeviceTrajectoryParams = {
  id: string;
  sessionId: string;
  appId: string;
  platform: string;
  screenTitle?: string;
  goal?: string;
  gestures: SyntheticGesture[];
  approved?: boolean;
};

export function syntheticDeviceTrajectory(params: SyntheticDeviceTrajectoryParams): TrajectorySpan {
  const observation: TrajectoryObservation = {
    kind: "observation",
    source: "device",
    summary: params.screenTitle ? `${params.appId} on ${params.screenTitle}` : `${params.appId} active on device`,
    ts: 1,
    metadata: {
      appId: params.appId,
      platform: params.platform,
      ...(params.screenTitle ? { screenTitle: params.screenTitle } : {}),
    },
  };

  const actions: TrajectoryAction[] = params.gestures.map((gesture, index) => ({
    kind: "action",
    tool: "device",
    summary: summarizeGesture(gesture),
    ts: index + 2,
    metadata: {
      gesture: gesture.kind,
      ...(gesture.target ? { target: gesture.target } : {}),
      ...(gesture.direction ? { direction: gesture.direction } : {}),
      ...(gesture.valueSummary ? { valueSummary: gesture.valueSummary } : {}),
    },
  }));

  return {
    id: params.id,
    sessionId: params.sessionId,
    createdAt: "2026-01-01T00:00:00.000Z",
    captureTier: "app",
    observations: [observation],
    actions,
    ...(params.goal ? { outcome: { status: "success", summary: params.goal } } : {}),
    ...(params.approved
      ? { review: { status: "approved", reviewedAt: "2026-01-01T00:00:00.000Z", reviewedBy: "operator" } }
      : {}),
  };
}

function summarizeGesture(gesture: SyntheticGesture): string {
  switch (gesture.kind) {
    case "tap":
      return gesture.target ? `tapped ${gesture.target}` : "tapped device";
    case "swipe":
      return gesture.direction ? `swiped ${gesture.direction}` : "swiped device";
    case "scroll":
      return gesture.direction ? `scrolled ${gesture.direction}` : "scrolled device";
    case "type":
      return gesture.target ? `typed into ${gesture.target}` : "typed on device";
    default:
      return gesture.target ? `${gesture.kind} ${gesture.target}` : `${gesture.kind} device`;
  }
}
