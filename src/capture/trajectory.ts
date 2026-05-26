export const CAPTURE_TIER_VALUES = ["off", "operator", "app", "full"] as const;

export type CaptureTier = (typeof CAPTURE_TIER_VALUES)[number];

export type CapturePolicy = {
  tier: CaptureTier;
  allowApps: string[];
  denyApps: string[];
  retainRawDays: number;
  retainDerivedDays: number;
  requireVisibleIndicator: boolean;
  allowTrainingExport: boolean;
};

export type CaptureConsentGrant = {
  id: string;
  tier: CaptureTier;
  grantedAt: string;
  expiresAt?: string;
  revokedAt?: string;
  purpose: string;
};

export type TrajectoryObservation = {
  kind: "observation";
  source: string;
  summary: string;
  ts: number;
  metadata?: Record<string, unknown>;
};

export type TrajectoryAction = {
  kind: "action";
  tool: string;
  summary: string;
  ts: number;
  metadata?: Record<string, unknown>;
};

export type TrajectoryReviewStatus = "pending" | "approved" | "rejected";

export type ReviewedTrajectoryObservation = {
  ts: number;
  source: string;
  summary: string;
};

export type ReviewedTrajectoryAction = {
  ts: number;
  tool: string;
  summary: string;
};

export type ReviewedTranscriptMessage = {
  ts: number;
  role: "system" | "user" | "assistant" | "tool";
  content: string;
};

export type TrajectoryReview = {
  status: TrajectoryReviewStatus;
  reviewedAt: string;
  reviewedBy: string;
  reviewNote?: string;
  redactedObservations?: ReviewedTrajectoryObservation[];
  redactedActions?: ReviewedTrajectoryAction[];
  redactedTranscript?: ReviewedTranscriptMessage[];
};

export type TrajectorySpan = {
  id: string;
  sessionId: string;
  createdAt: string;
  captureTier: CaptureTier;
  consentGrantId?: string;
  observations: TrajectoryObservation[];
  actions: TrajectoryAction[];
  outcome?: {
    status: "success" | "failure" | "aborted";
    summary: string;
    reward?: number;
  };
  review?: TrajectoryReview;
};

export const DEFAULT_SENSITIVE_APP_DENYLIST = [
  "1password",
  "lastpass",
  "bitwarden",
  "bank",
  "mail-compose",
  "secrets-manager",
  "system-settings",
  "auth-dialog",
  "credential-terminal",
] as const;

export function normalizeCaptureTier(value: string | null | undefined): CaptureTier {
  const normalized = value?.trim().toLowerCase();
  if (normalized === "off" || normalized === "operator" || normalized === "app" || normalized === "full") {
    return normalized;
  }
  return "operator";
}

export function defaultCapturePolicy(): CapturePolicy {
  return {
    tier: "operator",
    allowApps: [],
    denyApps: [...DEFAULT_SENSITIVE_APP_DENYLIST],
    retainRawDays: 1,
    retainDerivedDays: 30,
    requireVisibleIndicator: true,
    allowTrainingExport: false,
  };
}

export function isSensitiveApp(appId: string, policy: CapturePolicy): boolean {
  const normalized = appId.trim().toLowerCase();
  return policy.denyApps.some((entry) => normalized.includes(entry.toLowerCase()));
}

export function isConsentGrantActive(
  grant: CaptureConsentGrant,
  now: Date = new Date(),
): boolean {
  if (grant.revokedAt && new Date(grant.revokedAt).getTime() <= now.getTime()) {
    return false;
  }
  if (grant.expiresAt && new Date(grant.expiresAt).getTime() <= now.getTime()) {
    return false;
  }
  return true;
}

export function buildTrajectorySpan(params: {
  id: string;
  sessionId: string;
  captureTier?: CaptureTier;
  consentGrantId?: string;
  observations?: TrajectoryObservation[];
  actions?: TrajectoryAction[];
  outcome?: TrajectorySpan["outcome"];
}): TrajectorySpan {
  return {
    id: params.id,
    sessionId: params.sessionId,
    createdAt: new Date().toISOString(),
    captureTier: params.captureTier ?? "operator",
    ...(params.consentGrantId ? { consentGrantId: params.consentGrantId } : {}),
    observations: [...(params.observations ?? [])],
    actions: [...(params.actions ?? [])],
    ...(params.outcome ? { outcome: params.outcome } : {}),
  };
}
