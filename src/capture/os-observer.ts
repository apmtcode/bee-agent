import type { CaptureRecordResult, ConsentAwareCaptureRecorder } from "./recorder.js";
import type { CaptureTier, TrajectorySpan } from "./trajectory.js";

export type OsEventKind = "focus-changed" | "window-opened" | "file-opened" | "command-ran";

export type OsObservationInput = {
  sessionId: string;
  appId: string;
  visibleIndicator: boolean;
  captureTier?: CaptureTier;
  ts: number;
  event: OsEventKind;
  windowTitle?: string;
  filePath?: string;
  commandSummary?: string;
  outcome?: TrajectorySpan["outcome"];
};

export class OsCaptureObserver {
  constructor(private readonly recorder: ConsentAwareCaptureRecorder) {}

  async observe(input: OsObservationInput): Promise<CaptureRecordResult> {
    return await this.recorder.record({
      sessionId: input.sessionId,
      appId: input.appId,
      captureTier: input.captureTier,
      visibleIndicator: input.visibleIndicator,
      observations: [
        {
          kind: "observation",
          source: "os",
          summary: summarizeObservation(input),
          ts: input.ts,
          metadata: {
            event: input.event,
            ...(input.windowTitle ? { windowTitle: input.windowTitle } : {}),
            ...(input.filePath ? { filePath: input.filePath } : {}),
            ...(input.commandSummary ? { commandSummary: input.commandSummary } : {}),
          },
        },
      ],
      outcome: input.outcome,
    });
  }
}

function summarizeObservation(input: OsObservationInput): string {
  switch (input.event) {
    case "focus-changed":
      return input.windowTitle ? `focused ${input.windowTitle}` : `focused ${input.appId}`;
    case "window-opened":
      return input.windowTitle ? `opened ${input.windowTitle}` : `opened ${input.appId}`;
    case "file-opened":
      return input.filePath ? `opened ${input.filePath}` : `opened file in ${input.appId}`;
    case "command-ran":
      return input.commandSummary ? `ran ${input.commandSummary}` : `ran command in ${input.appId}`;
  }
}
