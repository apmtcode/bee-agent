import { buildTrajectorySpan, type TrajectoryAction, type TrajectorySpan } from "../capture/trajectory.js";

/**
 * Deterministic synthetic movement-stream generator.
 *
 * Standing objective #2 says the real recording/training executes only when the
 * user runs bee-agent locally; in the cloud we validate the *code* with
 * synthetic/simulated event streams. This generator emits structured, seeded
 * trajectories that follow a small library of reusable "workflows" (ordered
 * action templates) with controlled variation, so we can validate the
 * capture → dataset → train → replay → eval round-trip without any real OS
 * input. Same seed ⇒ identical output (no Date/Math.random).
 */

export type SyntheticMovementWorkflow = {
  id: string;
  /** Ordered action steps; each is rendered into a `TrajectoryAction`. */
  steps: { tool: string; verb: string; targets: string[] }[];
};

export type SyntheticMovementOptions = {
  seed: number;
  sessions: number;
  /** Workflows to sample from. Defaults to a built-in editor/browser set. */
  workflows?: SyntheticMovementWorkflow[];
  /** Base timestamp; each action advances by `stepMillis`. Default 1_700_000_000_000. */
  startTs?: number;
  stepMillis?: number;
};

export const DEFAULT_SYNTHETIC_WORKFLOWS: SyntheticMovementWorkflow[] = [
  {
    id: "editor-edit-save",
    steps: [
      { tool: "os", verb: "focus-changed", targets: ["editor"] },
      { tool: "device", verb: "tap", targets: ["file-tree", "open-file"] },
      { tool: "device", verb: "type", targets: ["code-buffer"] },
      { tool: "device", verb: "shortcut", targets: ["save"] },
      { tool: "os", verb: "command-ran", targets: ["build", "test"] },
    ],
  },
  {
    id: "browser-search-open",
    steps: [
      { tool: "os", verb: "focus-changed", targets: ["browser"] },
      { tool: "device", verb: "tap", targets: ["address-bar"] },
      { tool: "device", verb: "type", targets: ["query"] },
      { tool: "device", verb: "tap", targets: ["result-1", "result-2"] },
      { tool: "device", verb: "scroll", targets: ["page"] },
    ],
  },
];

/** Tiny deterministic PRNG (mulberry32) — seeded, no global state. */
function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function generateSyntheticMovementTrajectories(
  options: SyntheticMovementOptions,
): TrajectorySpan[] {
  const workflows = options.workflows ?? DEFAULT_SYNTHETIC_WORKFLOWS;
  if (workflows.length === 0) {
    return [];
  }
  const startTs = options.startTs ?? 1_700_000_000_000;
  const stepMillis = options.stepMillis ?? 1000;
  const random = mulberry32(options.seed);

  const trajectories: TrajectorySpan[] = [];
  for (let s = 0; s < options.sessions; s += 1) {
    const workflow = workflows[Math.floor(random() * workflows.length)]!;
    let ts = startTs + s * stepMillis * 1000;
    const actions: TrajectoryAction[] = workflow.steps.map((step) => {
      const target = step.targets[Math.floor(random() * step.targets.length)]!;
      ts += stepMillis;
      return {
        kind: "action",
        tool: step.tool,
        summary: `${step.verb} ${target}`,
        ts,
        metadata: { gesture: step.verb, target },
      } satisfies TrajectoryAction;
    });

    trajectories.push(
      buildTrajectorySpan({
        id: `synthetic-${workflow.id}-${s}`,
        sessionId: `synthetic-session-${s}`,
        captureTier: "full",
        actions,
        outcome: { status: "success", summary: `completed ${workflow.id}`, reward: 1 },
      }),
    );
  }

  return trajectories;
}
