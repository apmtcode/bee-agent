import type { ExportedReplayManifest } from "./export-manifest.js";

/**
 * Synthetic movement-stream generator. bee-agent runs in the cloud with no
 * access to a real machine, so we validate the capture -> dataset -> train ->
 * infer loop against deterministic synthetic replay manifests that mimic the
 * shape produced by the real capture pipeline (device/os adapters -> recorder ->
 * replay manifest). No RNG — sequences are fixed for reproducible CI.
 */

export type SyntheticGesture = {
  /** Observation source/verb, e.g. `["device", "focused"]`. */
  observe?: [source: string, verb: string];
  /** Action tool/verb, e.g. `["device", "tapped"]`. */
  act?: [tool: string, verb: string];
};

export type SyntheticTrajectorySpec = {
  sessionId: string;
  trajectoryId: string;
  steps: SyntheticGesture[];
};

/** Build one exported-replay-shaped manifest from a synthetic trajectory spec. */
export function buildSyntheticReplay(spec: SyntheticTrajectorySpec): ExportedReplayManifest {
  const events: ExportedReplayManifest["events"] = [];
  let ts = 0;
  for (const step of spec.steps) {
    if (step.observe) {
      const [source, verb] = step.observe;
      events.push({
        kind: "observation",
        ts: (ts += 1),
        trajectoryId: spec.trajectoryId,
        source,
        summary: `${verb} target`,
      });
    }
    if (step.act) {
      const [tool, verb] = step.act;
      events.push({
        kind: "action",
        ts: (ts += 1),
        trajectoryId: spec.trajectoryId,
        tool,
        summary: `${verb} target`,
      });
    }
  }
  return {
    sessionId: spec.sessionId,
    trajectoryIds: [spec.trajectoryId],
    eventCount: events.length,
    events,
  };
}

export function buildSyntheticReplays(specs: SyntheticTrajectorySpec[]): ExportedReplayManifest[] {
  return specs.map(buildSyntheticReplay);
}

/**
 * A small library of related "open a document" workflows that share structure
 * (open -> ... -> save) but diverge in the middle, so a model trained on some
 * can be scored on held-out others to measure generalization.
 */
export function sampleDocumentWorkflows(): SyntheticTrajectorySpec[] {
  const open: SyntheticGesture = { observe: ["os", "opened"], act: ["device", "tapped"] };
  const save: SyntheticGesture = { act: ["device", "tapped"], observe: ["os", "saved"] };
  return [
    {
      sessionId: "s-edit",
      trajectoryId: "t-edit",
      steps: [open, { act: ["editor", "typed"] }, save],
    },
    {
      sessionId: "s-search",
      trajectoryId: "t-search",
      steps: [open, { act: ["editor", "searched"] }, save],
    },
    {
      sessionId: "s-format",
      trajectoryId: "t-format",
      steps: [open, { act: ["editor", "formatted"] }, save],
    },
  ];
}
