import type {
  TrajectoryAction,
  TrajectoryObservation,
  TrajectorySpan,
} from "../../capture/trajectory.js";

/**
 * A deterministic, seedable synthetic event-stream generator. Real on-device
 * capture is unavailable in the cloud, so this produces structured trajectories
 * from a small grammar of movement "workflows" — enough to validate the whole
 * capture→dataset→train→infer→generalise loop without any real OS input.
 *
 * The generator is fully deterministic given a seed (no `Math.random`), so tests
 * and the self-evolution engine stay reproducible.
 */

/** A single step in a workflow: an app gesture that becomes one movement token. */
export type WorkflowStep = {
  tool: string;
  gesture: string;
  target: string;
};

export type WorkflowSpec = {
  appName: string;
  steps: WorkflowStep[];
};

export type SyntheticGeneratorOptions = {
  seed?: number;
  /** Timestamp the first event starts at (ms). Defaults to a fixed epoch. */
  startTs?: number;
  /** ms between consecutive events. Defaults to 1000. */
  stepMs?: number;
};

/** Small deterministic LCG — avoids Math.random so runs are reproducible. */
function makeRng(seed: number): () => number {
  let state = (seed >>> 0) || 1;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 0xffffffff;
  };
}

function actionFrom(step: WorkflowStep, ts: number): TrajectoryAction {
  return {
    kind: "action",
    tool: step.tool,
    summary: `${step.gesture} ${step.target}`,
    ts,
    metadata: { gesture: step.gesture, target: step.target },
  };
}

function observationFrom(appName: string, ts: number): TrajectoryObservation {
  return {
    kind: "observation",
    source: "device",
    summary: `${appName} active`,
    ts,
    metadata: { appName },
  };
}

/**
 * Emit `count` approved trajectories for a workflow. Each is the same ordered
 * movement sequence (so a policy can learn it), stamped as reviewed/approved so
 * it flows through the reviewed-export gate.
 */
export function generateWorkflowTrajectories(
  workflow: WorkflowSpec,
  count: number,
  options: SyntheticGeneratorOptions = {},
): TrajectorySpan[] {
  const startTs = options.startTs ?? 1_700_000_000_000;
  const stepMs = options.stepMs ?? 1000;
  const trajectories: TrajectorySpan[] = [];

  for (let i = 0; i < count; i += 1) {
    const base = startTs + i * (workflow.steps.length + 2) * stepMs;
    const observations = [observationFrom(workflow.appName, base)];
    const actions = workflow.steps.map((step, index) => actionFrom(step, base + (index + 1) * stepMs));
    trajectories.push({
      id: `${workflow.appName}-${i}`,
      sessionId: `session-${workflow.appName}`,
      createdAt: new Date(base).toISOString(),
      captureTier: "full",
      observations,
      actions,
      outcome: { status: "success", summary: `${workflow.appName} workflow`, reward: 1 },
      review: { status: "approved", reviewedAt: new Date(base).toISOString(), reviewedBy: "synthetic" },
    });
  }

  return trajectories;
}

/**
 * Produce a "related but new" trajectory in the same app: it shares a leading
 * sub-sequence of the workflow (so shorter-context prediction should succeed via
 * backoff) but diverges at the tail with a novel-but-plausible step. This is the
 * held-out probe for measuring generalisation, not verbatim reproduction.
 */
export function generateRelatedTrajectory(
  workflow: WorkflowSpec,
  options: SyntheticGeneratorOptions = {},
): TrajectorySpan {
  const rng = makeRng(options.seed ?? 1);
  const startTs = options.startTs ?? 1_800_000_000_000;
  const stepMs = options.stepMs ?? 1000;

  // Keep at least one leading step so history-conditioned backoff has signal.
  const keep = Math.max(1, Math.floor(rng() * (workflow.steps.length - 1)) + 1);
  const prefix = workflow.steps.slice(0, keep);
  const tail = workflow.steps.slice(keep).reverse();
  const steps = [...prefix, ...tail];

  const observations = [observationFrom(workflow.appName, startTs)];
  const actions = steps.map((step, index) => actionFrom(step, startTs + (index + 1) * stepMs));

  return {
    id: `${workflow.appName}-related`,
    sessionId: `session-${workflow.appName}-related`,
    createdAt: new Date(startTs).toISOString(),
    captureTier: "full",
    observations,
    actions,
    outcome: { status: "success", summary: `${workflow.appName} related workflow`, reward: 1 },
    review: { status: "approved", reviewedAt: new Date(startTs).toISOString(), reviewedBy: "synthetic" },
  };
}
