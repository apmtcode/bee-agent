import { randomUUID } from "node:crypto";
import { FileCaptureConsentStore } from "./consent-store.js";
import { FileTrajectoryStore } from "./trajectory-store.js";
import {
  buildTrajectorySpan,
  defaultCapturePolicy,
  isSensitiveApp,
  type CapturePolicy,
  type CaptureTier,
  type TrajectoryAction,
  type TrajectoryObservation,
  type TrajectorySpan,
} from "./trajectory.js";

export type CaptureRecorderOptions = {
  consentStore: FileCaptureConsentStore;
  trajectoryStore: FileTrajectoryStore;
  policy?: CapturePolicy;
};

export type CaptureRecorderInput = {
  sessionId: string;
  appId?: string;
  captureTier?: CaptureTier;
  visibleIndicator?: boolean;
  observations?: TrajectoryObservation[];
  actions?: TrajectoryAction[];
  outcome?: TrajectorySpan["outcome"];
};

export type CaptureRecordResult =
  | { recorded: true; trajectory: TrajectorySpan }
  | {
      recorded: false;
      reason: "sensitive-app" | "missing-visible-indicator" | "missing-consent";
    };

export class ConsentAwareCaptureRecorder {
  private readonly policy: CapturePolicy;

  constructor(private readonly options: CaptureRecorderOptions) {
    this.policy = options.policy ?? defaultCapturePolicy();
  }

  async record(input: CaptureRecorderInput): Promise<CaptureRecordResult> {
    if (input.appId && isSensitiveApp(input.appId, this.policy)) {
      return { recorded: false, reason: "sensitive-app" };
    }

    const captureTier = input.captureTier ?? this.policy.tier;
    if (this.policy.requireVisibleIndicator && captureTier !== "off" && !input.visibleIndicator) {
      return { recorded: false, reason: "missing-visible-indicator" };
    }

    const consent = await this.options.consentStore.findActiveByTier(captureTier);
    if (!consent) {
      return { recorded: false, reason: "missing-consent" };
    }

    const trajectory = buildTrajectorySpan({
      id: randomUUID(),
      sessionId: input.sessionId,
      captureTier,
      consentGrantId: consent.id,
      observations: input.observations,
      actions: input.actions,
      outcome: input.outcome,
    });
    await this.options.trajectoryStore.add(trajectory);
    return { recorded: true, trajectory };
  }
}
