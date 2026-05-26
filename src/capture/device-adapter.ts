import type { CaptureRecordResult, ConsentAwareCaptureRecorder } from "./recorder.js";
import type { CaptureTier, TrajectorySpan } from "./trajectory.js";

export type DevicePlatform = "ios" | "android" | "macos" | "windows" | "linux";
export type DeviceGestureKind = "tap" | "swipe" | "scroll" | "type" | "shortcut";

export type DeviceCaptureInput = {
  sessionId: string;
  deviceId: string;
  platform: DevicePlatform;
  appId: string;
  appName: string;
  screenTitle?: string;
  selectionSummary?: string;
  captureTier?: CaptureTier;
  visibleIndicator: boolean;
  ts: number;
  gesture?: {
    kind: DeviceGestureKind;
    target?: string;
    direction?: "up" | "down" | "left" | "right";
    valueSummary?: string;
    ts: number;
  };
  outcome?: TrajectorySpan["outcome"];
};

export class DeviceCaptureAdapter {
  constructor(private readonly recorder: ConsentAwareCaptureRecorder) {}

  async record(input: DeviceCaptureInput): Promise<CaptureRecordResult> {
    return await this.recorder.record({
      sessionId: input.sessionId,
      appId: input.appId,
      captureTier: input.captureTier,
      visibleIndicator: input.visibleIndicator,
      observations: [
        {
          kind: "observation",
          source: "device",
          summary: input.screenTitle ? `${input.appName} on ${input.screenTitle}` : `${input.appName} active on device`,
          ts: input.ts,
          metadata: {
            deviceId: input.deviceId,
            platform: input.platform,
            appName: input.appName,
            ...(input.screenTitle ? { screenTitle: input.screenTitle } : {}),
            ...(input.selectionSummary ? { selectionSummary: input.selectionSummary } : {}),
          },
        },
      ],
      actions: input.gesture
        ? [
            {
              kind: "action",
              tool: "device",
              summary: summarizeGesture(input.gesture),
              ts: input.gesture.ts,
              metadata: {
                gesture: input.gesture.kind,
                ...(input.gesture.target ? { target: input.gesture.target } : {}),
                ...(input.gesture.direction ? { direction: input.gesture.direction } : {}),
                ...(input.gesture.valueSummary ? { valueSummary: input.gesture.valueSummary } : {}),
              },
            },
          ]
        : [],
      outcome: input.outcome,
    });
  }
}

function summarizeGesture(gesture: NonNullable<DeviceCaptureInput["gesture"]>): string {
  switch (gesture.kind) {
    case "tap":
      return gesture.target ? `tapped ${gesture.target}` : "tapped device";
    case "swipe":
      return gesture.direction ? `swiped ${gesture.direction}` : "swiped device";
    case "scroll":
      return gesture.direction ? `scrolled ${gesture.direction}` : "scrolled device";
    case "type":
      return gesture.target ? `typed into ${gesture.target}` : "typed on device";
    case "shortcut":
      return gesture.target ? `triggered ${gesture.target}` : "triggered device shortcut";
  }
}
