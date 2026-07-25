import { buildTrajectorySpan, type TrajectoryAction, type TrajectorySpan } from "../capture/trajectory.js";
import { createSeededRng } from "./movement-model.js";

// ---------------------------------------------------------------------------
// Synthetic movement-stream generator.
//
// Because the engine runs in Anthropic's cloud with no access to the user's
// real machine, we cannot record genuine mouse/keyboard/window events. This
// generator produces deterministic, seeded synthetic movement trajectories from
// a small grammar of UI "flows" (open → navigate → type → submit …) so the
// capture → dataset → model → generalization pipeline can be validated end-to-end
// in CI. The same code path consumes real captured trajectories on-device.
// ---------------------------------------------------------------------------

type MovementStep = {
  gesture: "tap" | "swipe" | "scroll" | "type" | "shortcut";
  target?: string;
  direction?: "up" | "down" | "left" | "right";
};

export type MovementFlow = {
  name: string;
  /** Ordered steps. Entries may offer alternatives; one is chosen per sample. */
  steps: Array<MovementStep | MovementStep[]>;
};

/**
 * A small library of related UI flows. They share vocabulary (tap:menu,
 * type:search, tap:submit …) so a model trained on some flows can be evaluated
 * for generalization to a held-out but related flow.
 */
export const DEFAULT_MOVEMENT_FLOWS: MovementFlow[] = [
  {
    name: "search-and-open",
    steps: [
      { gesture: "tap", target: "menu" },
      { gesture: "type", target: "search" },
      { gesture: "tap", target: "result" },
      { gesture: "scroll", direction: "down" },
    ],
  },
  {
    name: "compose-and-submit",
    steps: [
      { gesture: "tap", target: "compose" },
      { gesture: "type", target: "body" },
      [
        { gesture: "tap", target: "attach" },
        { gesture: "shortcut", target: "attach" },
      ],
      { gesture: "tap", target: "submit" },
    ],
  },
  {
    name: "navigate-and-filter",
    steps: [
      { gesture: "tap", target: "menu" },
      { gesture: "swipe", direction: "left" },
      { gesture: "type", target: "filter" },
      { gesture: "tap", target: "apply" },
    ],
  },
];

function stepToAction(step: MovementStep, ts: number): TrajectoryAction {
  const summaryParts = [step.gesture, step.target, step.direction].filter(
    (part): part is string => typeof part === "string",
  );
  return {
    kind: "action",
    tool: "device",
    summary: summaryParts.join(" "),
    ts,
    metadata: {
      gesture: step.gesture,
      ...(step.target ? { target: step.target } : {}),
      ...(step.direction ? { direction: step.direction } : {}),
    },
  };
}

export type SyntheticMovementOptions = {
  flows?: MovementFlow[];
  /** Number of trajectories to generate. */
  count: number;
  seed: number;
  /** Base epoch millis for the first event; each action advances by ~stepMs. */
  baseTs?: number;
  stepMs?: number;
};

/**
 * Generate `count` deterministic synthetic trajectory spans by sampling flows
 * and resolving each alternative step with the seeded PRNG.
 */
export function generateSyntheticTrajectories(options: SyntheticMovementOptions): TrajectorySpan[] {
  const flows = options.flows ?? DEFAULT_MOVEMENT_FLOWS;
  if (flows.length === 0) {
    return [];
  }
  const rng = createSeededRng(options.seed);
  const baseTs = options.baseTs ?? 1_700_000_000_000;
  const stepMs = options.stepMs ?? 500;
  const trajectories: TrajectorySpan[] = [];

  for (let i = 0; i < options.count; i += 1) {
    const flow = flows[Math.floor(rng() * flows.length)] ?? flows[0];
    const actions: TrajectoryAction[] = [];
    let ts = baseTs + i * stepMs * 100;
    for (const entry of flow.steps) {
      const step = Array.isArray(entry) ? entry[Math.floor(rng() * entry.length)] ?? entry[0] : entry;
      actions.push(stepToAction(step, ts));
      ts += stepMs;
    }
    trajectories.push(
      buildTrajectorySpanDeterministic({
        id: `synthetic-${options.seed}-${i}`,
        sessionId: `synthetic-session-${options.seed}`,
        actions,
      }),
    );
  }

  return trajectories;
}

// `buildTrajectorySpan` stamps `createdAt` with the wall clock; for reproducible
// datasets we pin it to a value derived from the first action instead.
function buildTrajectorySpanDeterministic(params: {
  id: string;
  sessionId: string;
  actions: TrajectoryAction[];
}): TrajectorySpan {
  const span = buildTrajectorySpan({
    id: params.id,
    sessionId: params.sessionId,
    captureTier: "app",
    actions: params.actions,
  });
  const firstTs = params.actions[0]?.ts ?? 0;
  return { ...span, createdAt: new Date(firstTs).toISOString() };
}
