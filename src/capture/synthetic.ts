import { buildTrajectorySpan, type TrajectoryAction, type TrajectorySpan } from "./trajectory.js";

/**
 * Deterministic synthetic movement-stream generator.
 *
 * bee-agent runs in the cloud with no access to a real machine, so the
 * capture → dataset → replay → train loop needs believable input that does not
 * depend on real OS events. This produces {@link TrajectorySpan}s whose action
 * streams follow named, repeatable UI workflows (a fixed grammar of gestures),
 * with optional variants so a policy can be trained on some runs and evaluated
 * on held-out-but-related runs. No randomness or wall-clock is used, so output
 * is byte-stable and CI-safe.
 */

export type SyntheticWorkflow = {
  /** Stable workflow name, e.g. `compose-email`. */
  name: string;
  /** Ordered gestures, each rendered into one action token. */
  steps: Array<{ tool: string; summary: string }>;
};

export const DEFAULT_SYNTHETIC_WORKFLOWS: SyntheticWorkflow[] = [
  {
    name: "compose-email",
    steps: [
      { tool: "device", summary: "tapped compose" },
      { tool: "device", summary: "typed into recipient" },
      { tool: "device", summary: "typed into subject" },
      { tool: "device", summary: "typed into body" },
      { tool: "device", summary: "tapped send" },
    ],
  },
  {
    name: "file-search",
    steps: [
      { tool: "device", summary: "triggered spotlight" },
      { tool: "device", summary: "typed into search" },
      { tool: "device", summary: "scrolled down" },
      { tool: "device", summary: "tapped result" },
    ],
  },
  {
    name: "commit-flow",
    steps: [
      { tool: "device", summary: "tapped source control" },
      { tool: "device", summary: "typed into message" },
      { tool: "device", summary: "tapped commit" },
      { tool: "device", summary: "tapped push" },
    ],
  },
];

export type SyntheticStreamOptions = {
  sessionId: string;
  workflow: SyntheticWorkflow;
  /** Number of trajectories to emit for this workflow (default 1). */
  repeats?: number;
  /** First action timestamp; each action advances by `stepMs` (default 1000). */
  baseTs?: number;
  stepMs?: number;
  /** Deterministic id prefix; trajectories are `${idPrefix}-${index}`. */
  idPrefix?: string;
};

/** Generate a batch of trajectories replaying one workflow `repeats` times. */
export function generateSyntheticTrajectories(options: SyntheticStreamOptions): TrajectorySpan[] {
  const repeats = Math.max(1, Math.trunc(options.repeats ?? 1));
  const baseTs = options.baseTs ?? 1_000;
  const stepMs = options.stepMs ?? 1_000;
  const idPrefix = options.idPrefix ?? options.workflow.name;

  const trajectories: TrajectorySpan[] = [];
  for (let run = 0; run < repeats; run += 1) {
    const runBase = baseTs + run * options.workflow.steps.length * stepMs * 4;
    const actions: TrajectoryAction[] = options.workflow.steps.map((step, index) => ({
      kind: "action",
      tool: step.tool,
      summary: step.summary,
      ts: runBase + index * stepMs,
      metadata: { workflow: options.workflow.name, step: index },
    }));
    trajectories.push(
      buildTrajectorySpan({
        id: `${idPrefix}-${run}`,
        sessionId: options.sessionId,
        captureTier: "app",
        observations: [
          {
            kind: "observation",
            source: "synthetic",
            summary: `workflow ${options.workflow.name} run ${run}`,
            ts: runBase,
            metadata: { workflow: options.workflow.name },
          },
        ],
        actions,
        outcome: { status: "success", summary: `completed ${options.workflow.name}`, reward: 1 },
      }),
    );
  }
  return trajectories;
}

/**
 * Produce a related variant of a workflow by inserting, dropping, or replacing
 * one step. Used to build held-out trajectories that a policy trained on the
 * base workflow should partially generalize to.
 */
export function variantWorkflow(
  workflow: SyntheticWorkflow,
  mutation: { kind: "insert"; at: number; step: { tool: string; summary: string } }
    | { kind: "drop"; at: number }
    | { kind: "replace"; at: number; step: { tool: string; summary: string } },
): SyntheticWorkflow {
  const steps = [...workflow.steps];
  switch (mutation.kind) {
    case "insert":
      steps.splice(clampIndex(mutation.at, steps.length + 1), 0, mutation.step);
      break;
    case "drop":
      steps.splice(clampIndex(mutation.at, steps.length), 1);
      break;
    case "replace": {
      const index = clampIndex(mutation.at, steps.length);
      if (steps.length > 0) {
        steps[index] = mutation.step;
      }
      break;
    }
  }
  return { name: `${workflow.name}-variant`, steps };
}

function clampIndex(value: number, length: number): number {
  const index = Math.trunc(value);
  if (index < 0) {
    return 0;
  }
  if (index >= length) {
    return Math.max(0, length - 1);
  }
  return index;
}
