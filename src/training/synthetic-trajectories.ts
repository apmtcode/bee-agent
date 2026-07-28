import type { DeviceGestureKind } from "../capture/device-adapter.js";
import { buildTrajectorySpan, type TrajectorySpan } from "../capture/trajectory.js";

/**
 * Deterministic synthetic trajectory generator.
 *
 * The real local-movement pipeline records mouse/keyboard/UI events on the user's
 * machine. In the cloud we have no such machine, so we synthesize structurally
 * realistic trajectory streams to validate the capture -> dataset -> train -> replay
 * round-trip and to build held-out sets for generalization evaluation. Output is a
 * fully deterministic function of its inputs (no wall-clock / RNG).
 */

export type SyntheticMovementStep = {
  tool?: string;
  gesture: DeviceGestureKind;
  target?: string;
  direction?: "up" | "down" | "left" | "right";
  valueSummary?: string;
};

export type SyntheticMovementTemplate = {
  name: string;
  steps: SyntheticMovementStep[];
};

/** A common "focus search, type a query, submit, open first result" motion. */
export const SEARCH_AND_OPEN_TEMPLATE: SyntheticMovementTemplate = {
  name: "search-and-open",
  steps: [
    { gesture: "tap", target: "search-field" },
    { gesture: "type", target: "search-field", valueSummary: "query" },
    { gesture: "shortcut", target: "submit" },
    { gesture: "scroll", direction: "down" },
    { gesture: "tap", target: "first-result" },
  ],
};

/** A "compose, format, send" motion — structurally distinct from search-and-open. */
export const COMPOSE_AND_SEND_TEMPLATE: SyntheticMovementTemplate = {
  name: "compose-and-send",
  steps: [
    { gesture: "tap", target: "compose-button" },
    { gesture: "type", target: "body-field", valueSummary: "message" },
    { gesture: "shortcut", target: "bold" },
    { gesture: "tap", target: "send-button" },
  ],
};

export type GenerateSyntheticTrajectoryParams = {
  id: string;
  sessionId: string;
  app: string;
  template: SyntheticMovementTemplate;
  startTs?: number;
  stepIntervalMs?: number;
  outcome?: TrajectorySpan["outcome"];
};

export function generateSyntheticTrajectory(params: GenerateSyntheticTrajectoryParams): TrajectorySpan {
  const startTs = params.startTs ?? 1_000;
  const interval = params.stepIntervalMs ?? 100;
  const actions = params.template.steps.map((step, index) => ({
    kind: "action" as const,
    tool: step.tool ?? "device",
    summary: `${params.template.name}:${step.gesture}${step.target ? ` ${step.target}` : ""}`,
    ts: startTs + (index + 1) * interval,
    metadata: {
      gesture: step.gesture,
      ...(step.target ? { target: step.target } : {}),
      ...(step.direction ? { direction: step.direction } : {}),
      ...(step.valueSummary ? { valueSummary: step.valueSummary } : {}),
    },
  }));

  return buildTrajectorySpan({
    id: params.id,
    sessionId: params.sessionId,
    captureTier: "app",
    observations: [
      {
        kind: "observation",
        source: "device",
        summary: `${params.app} active`,
        ts: startTs,
        metadata: { appName: params.app, platform: "macos" },
      },
    ],
    actions,
    outcome: params.outcome,
  });
}

export type GenerateSyntheticFamilyParams = {
  template: SyntheticMovementTemplate;
  apps: string[];
  sessionId?: string;
  baseId?: string;
  startTs?: number;
  stepIntervalMs?: number;
};

/**
 * Generate one trajectory per app from the same template — a "family" of
 * new-but-related movements that differ only in their surrounding app context.
 */
export function generateSyntheticTrajectoryFamily(params: GenerateSyntheticFamilyParams): TrajectorySpan[] {
  const baseId = params.baseId ?? params.template.name;
  const sessionId = params.sessionId ?? `synthetic-${params.template.name}`;
  return params.apps.map((app, index) =>
    generateSyntheticTrajectory({
      id: `${baseId}-${app}`,
      sessionId: `${sessionId}-${index}`,
      app,
      template: params.template,
      ...(params.startTs !== undefined ? { startTs: params.startTs } : {}),
      ...(params.stepIntervalMs !== undefined ? { stepIntervalMs: params.stepIntervalMs } : {}),
    }),
  );
}
