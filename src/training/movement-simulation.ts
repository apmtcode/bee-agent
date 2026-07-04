import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

/**
 * Synthetic movement-stream generator.
 *
 * The mandate forbids access to a real machine, so the local-movement
 * subsystem is validated against *simulated* event streams. This generator
 * emits deterministic {@link TrajectorySpan}s whose actions follow a small set
 * of structured "workflow grammars" (e.g. open-app → search → type → submit),
 * with bounded, index-derived variation so the data is learnable *and* exercises
 * generalization (held-out variants that share sub-sequences with training).
 *
 * Everything is a pure function of the inputs — no `Math.random`, no `Date` —
 * so tests and replay round-trips are reproducible.
 */

export type SyntheticMovementOptions = {
  /** How many trajectories to generate. */
  count: number;
  /** Session id stamped on every generated span. */
  sessionId?: string;
  /** How many distinct workflow templates to cycle through. Default 3. */
  templateCount?: number;
  /** Timestamp (ms) of the first event; subsequent events increment. */
  startTs?: number;
};

type GestureStep = {
  gesture: "tap" | "swipe" | "scroll" | "type" | "shortcut";
  target?: string;
  direction?: "up" | "down" | "left" | "right";
};

/** Workflow grammars — each a canonical movement path a user might repeat. */
const WORKFLOW_TEMPLATES: Array<{ name: string; steps: GestureStep[] }> = [
  {
    name: "search-flow",
    steps: [
      { gesture: "tap", target: "launcher" },
      { gesture: "tap", target: "search-box" },
      { gesture: "type", target: "search-box" },
      { gesture: "shortcut", target: "submit" },
      { gesture: "tap", target: "first-result" },
    ],
  },
  {
    name: "browse-flow",
    steps: [
      { gesture: "tap", target: "launcher" },
      { gesture: "scroll", direction: "down" },
      { gesture: "scroll", direction: "down" },
      { gesture: "tap", target: "list-item" },
      { gesture: "swipe", direction: "left" },
    ],
  },
  {
    name: "compose-flow",
    steps: [
      { gesture: "tap", target: "launcher" },
      { gesture: "tap", target: "compose" },
      { gesture: "type", target: "editor" },
      { gesture: "type", target: "editor" },
      { gesture: "shortcut", target: "send" },
    ],
  },
];

/**
 * Generate `count` trajectories cycling through workflow templates. Even-indexed
 * trajectories follow a template verbatim; odd-indexed ones insert one extra
 * related step, producing "new but related" movements that share sub-sequences
 * with the verbatim runs (the signal a generalizing model should pick up).
 */
export function generateSyntheticTrajectories(options: SyntheticMovementOptions): TrajectorySpan[] {
  const sessionId = options.sessionId ?? "sim-session";
  const templateCount = Math.max(1, Math.min(options.templateCount ?? 3, WORKFLOW_TEMPLATES.length));
  const startTs = options.startTs ?? 0;
  const spans: TrajectorySpan[] = [];

  for (let index = 0; index < options.count; index += 1) {
    const template = WORKFLOW_TEMPLATES[index % templateCount]!;
    const steps = withVariation(template.steps, index);
    const baseTs = startTs + index * 100_000;
    const actions: TrajectoryAction[] = steps.map((step, stepIndex) => ({
      kind: "action",
      tool: "device",
      summary: describeStep(step),
      ts: baseTs + stepIndex * 1_000,
      metadata: {
        gesture: step.gesture,
        ...(step.target ? { target: step.target } : {}),
        ...(step.direction ? { direction: step.direction } : {}),
      },
    }));

    // Construct the span directly (not via `buildTrajectorySpan`, which stamps a
    // wall-clock `createdAt`) so the generator is fully deterministic.
    spans.push({
      id: `${sessionId}-traj-${String(index).padStart(4, "0")}`,
      sessionId,
      createdAt: new Date(baseTs).toISOString(),
      captureTier: "full",
      observations: [],
      actions,
      outcome: { status: "success", summary: `${template.name} completed`, reward: 1 },
    });
  }

  return spans;
}

function withVariation(steps: GestureStep[], index: number): GestureStep[] {
  if (index % 2 === 0) {
    return steps;
  }
  // Related-but-new variant: repeat the penultimate scroll/tap once more before
  // the terminal step, keeping the shared prefix intact for back-off.
  const insertAt = Math.max(1, steps.length - 1);
  const repeated = steps[insertAt - 1]!;
  return [...steps.slice(0, insertAt), repeated, ...steps.slice(insertAt)];
}

function describeStep(step: GestureStep): string {
  switch (step.gesture) {
    case "tap":
      return step.target ? `tapped ${step.target}` : "tapped";
    case "swipe":
      return step.direction ? `swiped ${step.direction}` : "swiped";
    case "scroll":
      return step.direction ? `scrolled ${step.direction}` : "scrolled";
    case "type":
      return step.target ? `typed into ${step.target}` : "typed";
    case "shortcut":
      return step.target ? `triggered ${step.target}` : "triggered shortcut";
  }
}
