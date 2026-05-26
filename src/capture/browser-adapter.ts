import type { CaptureRecordResult, ConsentAwareCaptureRecorder } from "./recorder.js";
import type { CaptureTier, TrajectorySpan } from "./trajectory.js";

export type BrowserActionKind = "navigate" | "click" | "type" | "submit";

export type BrowserCaptureInput = {
  sessionId: string;
  pageUrl: string;
  pageTitle?: string;
  domSummary?: string;
  activeElementLabel?: string;
  screenshotRef?: string;
  appId?: string;
  captureTier?: CaptureTier;
  visibleIndicator: boolean;
  ts: number;
  action?: {
    kind: BrowserActionKind;
    target?: string;
    valueSummary?: string;
    ts: number;
  };
  outcome?: TrajectorySpan["outcome"];
};

export class BrowserCaptureAdapter {
  constructor(private readonly recorder: ConsentAwareCaptureRecorder) {}

  async record(input: BrowserCaptureInput): Promise<CaptureRecordResult> {
    return await this.recorder.record({
      sessionId: input.sessionId,
      appId: input.appId ?? "browser",
      captureTier: input.captureTier,
      visibleIndicator: input.visibleIndicator,
      observations: [
        {
          kind: "observation",
          source: "browser",
          summary: summarizeBrowserObservation(input),
          ts: input.ts,
          metadata: {
            pageUrl: input.pageUrl,
            ...(input.pageTitle ? { pageTitle: input.pageTitle } : {}),
            ...(input.domSummary ? { domSummary: input.domSummary } : {}),
            ...(input.activeElementLabel ? { activeElementLabel: input.activeElementLabel } : {}),
            ...(input.screenshotRef ? { screenshotRef: input.screenshotRef } : {}),
          },
        },
      ],
      actions: input.action
        ? [
            {
              kind: "action",
              tool: "browser",
              summary: summarizeBrowserAction(input.action),
              ts: input.action.ts,
              metadata: {
                action: input.action.kind,
                ...(input.action.target ? { target: input.action.target } : {}),
                ...(input.action.valueSummary ? { valueSummary: input.action.valueSummary } : {}),
              },
            },
          ]
        : [],
      outcome: input.outcome,
    });
  }
}

function summarizeBrowserObservation(input: BrowserCaptureInput): string {
  return input.pageTitle ? `browser at ${input.pageTitle}` : `browser at ${input.pageUrl}`;
}

function summarizeBrowserAction(action: BrowserCaptureInput["action"]): string {
  if (!action) {
    return "browser action";
  }
  switch (action.kind) {
    case "navigate":
      return action.target ? `navigated to ${action.target}` : "navigated browser";
    case "click":
      return action.target ? `clicked ${action.target}` : "clicked browser element";
    case "type":
      return action.target ? `typed into ${action.target}` : "typed in browser";
    case "submit":
      return action.target ? `submitted ${action.target}` : "submitted browser action";
  }
}
