import type { ReplayTimelineEvent } from "../capture/replay.js";

/**
 * Deterministic synthetic movement-stream generator for the local-movement
 * learning subsystem. It fabricates capture-shaped replay timelines so the
 * capture → dataset → model → replay loop can be validated in the cloud
 * without any real OS input.
 *
 * Determinism is intentional (no RNG): a given spec always yields the same
 * stream, which keeps model-training tests reproducible. Variation comes from
 * an integer `seed` fed through a small linear-congruential sequence.
 */

export type SyntheticMovementStep = {
  /** Action verb, e.g. `click`, `type`, `scroll`. */
  verb: string;
  /** Tool/channel the action runs on. Defaults to `device`. */
  tool?: string;
  /** Optional target summary detail. */
  target?: string;
};

export type SyntheticMovementSpec = {
  trajectoryId: string;
  /** App/window context surfaced as a leading observation. */
  app: string;
  observationSource?: string;
  steps: SyntheticMovementStep[];
  /** Optional integer seed to vary timestamps deterministically. */
  seed?: number;
  /** Base timestamp (ms). Defaults to a fixed epoch for reproducibility. */
  startTs?: number;
};

export type SyntheticReplay = {
  sessionId: string;
  trajectoryIds: string[];
  eventCount: number;
  events: ReplayTimelineEvent[];
};

function lcg(seed: number): () => number {
  // Numerical Recipes LCG — deterministic pseudo-jitter, never Math.random.
  let state = (Math.abs(Math.floor(seed)) % 2147483647) || 1;
  return () => {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    return state / 0x7fffffff;
  };
}

export function generateSyntheticMovementStream(spec: SyntheticMovementSpec): SyntheticReplay {
  const start = spec.startTs ?? 1_700_000_000_000;
  const next = lcg(spec.seed ?? 1);
  let ts = start;
  const events: ReplayTimelineEvent[] = [
    {
      kind: "observation",
      ts,
      trajectoryId: spec.trajectoryId,
      source: spec.observationSource ?? "os",
      summary: `focused ${spec.app}`,
    },
  ];
  for (const step of spec.steps) {
    ts += 50 + Math.floor(next() * 200);
    events.push({
      kind: "action",
      ts,
      trajectoryId: spec.trajectoryId,
      tool: step.tool ?? "device",
      summary: step.target ? `${step.verb} ${step.target}` : step.verb,
    });
  }
  return {
    sessionId: `synthetic-${spec.trajectoryId}`,
    trajectoryIds: [spec.trajectoryId],
    eventCount: events.length,
    events,
  };
}

/** Generate a family of related streams: same task grammar across several apps. */
export function generateSyntheticMovementFamily(params: {
  apps: string[];
  steps: SyntheticMovementStep[];
  idPrefix?: string;
  observationSource?: string;
  seed?: number;
}): SyntheticReplay[] {
  const prefix = params.idPrefix ?? "traj";
  return params.apps.map((app, index) =>
    generateSyntheticMovementStream({
      trajectoryId: `${prefix}-${index}`,
      app,
      observationSource: params.observationSource,
      steps: params.steps,
      seed: (params.seed ?? 1) + index,
    }),
  );
}
