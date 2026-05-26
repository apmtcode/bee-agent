import { describe, expect, it } from "vitest";
import {
  buildTrajectorySpan,
  defaultCapturePolicy,
  isConsentGrantActive,
  isSensitiveApp,
  normalizeCaptureTier,
} from "./trajectory.js";

describe("trajectory capture defaults", () => {
  it("defaults to safe operator-tier capture", () => {
    const policy = defaultCapturePolicy();
    expect(policy.tier).toBe("operator");
    expect(policy.allowTrainingExport).toBe(false);
    expect(isSensitiveApp("1Password", policy)).toBe(true);
    expect(isSensitiveApp("Cursor", policy)).toBe(false);
  });

  it("normalizes capture tiers and builds trajectory spans", () => {
    expect(normalizeCaptureTier("FULL")).toBe("full");
    expect(normalizeCaptureTier("unknown")).toBe("operator");

    const span = buildTrajectorySpan({
      id: "traj-1",
      sessionId: "sess-1",
      actions: [{ kind: "action", tool: "browser", summary: "clicked deploy", ts: 10 }],
      observations: [{ kind: "observation", source: "browser", summary: "deploy page", ts: 9 }],
      outcome: { status: "success", summary: "deploy started", reward: 1 },
    });

    expect(span.captureTier).toBe("operator");
    expect(span.actions).toHaveLength(1);
    expect(span.observations).toHaveLength(1);
    expect(span.outcome?.status).toBe("success");
  });

  it("treats expired or revoked consent grants as inactive", () => {
    expect(
      isConsentGrantActive({
        id: "grant-active",
        tier: "app",
        grantedAt: "2026-01-01T00:00:00.000Z",
        expiresAt: "2026-01-02T00:00:00.000Z",
        purpose: "capture",
      }, new Date("2026-01-01T12:00:00.000Z")),
    ).toBe(true);
    expect(
      isConsentGrantActive({
        id: "grant-expired",
        tier: "app",
        grantedAt: "2026-01-01T00:00:00.000Z",
        expiresAt: "2026-01-01T01:00:00.000Z",
        purpose: "capture",
      }, new Date("2026-01-01T12:00:00.000Z")),
    ).toBe(false);
    expect(
      isConsentGrantActive({
        id: "grant-revoked",
        tier: "full",
        grantedAt: "2026-01-01T00:00:00.000Z",
        revokedAt: "2026-01-01T00:30:00.000Z",
        purpose: "capture",
      }, new Date("2026-01-01T12:00:00.000Z")),
    ).toBe(false);
  });
});
