import { buildTrajectorySpan, type TrajectoryAction, type TrajectorySpan } from "../capture/trajectory.js";
import type { DeviceGestureKind } from "../capture/device-adapter.js";

/**
 * Synthetic movement-stream generator.
 *
 * The engine runs in the cloud with NO access to a real machine, so the
 * capture -> dataset -> train -> replay -> generalize loop is validated against
 * *simulated* event streams produced here. Generation is fully deterministic
 * given a seed, so tests and the generalization eval are reproducible.
 *
 * Each "flow" is a small grammar of UI movements (a login, a search, ...).
 * Instances of a flow share their movement-token shape but vary their concrete
 * targets, which is exactly the structure the model must learn to repeat and
 * generalize over.
 */

export type MovementFlowStep = {
  gesture: DeviceGestureKind;
  direction?: "up" | "down" | "left" | "right";
  /** Candidate targets; one is chosen per instance via the seeded PRNG. */
  targets: string[];
};

export type MovementFlow = {
  name: string;
  appId: string;
  steps: MovementFlowStep[];
};

export const DEFAULT_MOVEMENT_FLOWS: MovementFlow[] = [
  {
    name: "login",
    appId: "com.example.workspace",
    steps: [
      { gesture: "tap", targets: ["email-field", "username-field"] },
      { gesture: "type", targets: ["credential", "identifier"] },
      { gesture: "tap", targets: ["password-field"] },
      { gesture: "type", targets: ["secret"] },
      { gesture: "tap", targets: ["sign-in", "continue"] },
    ],
  },
  {
    name: "search",
    appId: "com.example.browser",
    steps: [
      { gesture: "tap", targets: ["search-box", "omnibox"] },
      { gesture: "type", targets: ["query"] },
      { gesture: "scroll", direction: "down", targets: ["results"] },
      { gesture: "tap", targets: ["result-1", "result-2", "result-3"] },
    ],
  },
  {
    name: "settings",
    appId: "com.example.workspace",
    steps: [
      { gesture: "tap", targets: ["menu", "avatar"] },
      { gesture: "tap", targets: ["settings"] },
      { gesture: "scroll", direction: "down", targets: ["preferences"] },
      { gesture: "swipe", direction: "left", targets: ["toggle-notifications"] },
    ],
  },
];

export type GenerateSyntheticTrajectoriesParams = {
  seed: number;
  /** Number of trajectory spans to emit. */
  sequenceCount: number;
  flows?: MovementFlow[];
  idPrefix?: string;
  /** Epoch ms for the first event; subsequent events advance deterministically. */
  startTs?: number;
};

/**
 * Produce simulated capture output: real {@link TrajectorySpan}s (the same shape
 * the device adapter records), so the full pipeline — tokenize, train, evaluate
 * — exercises production schema rather than a bespoke test fixture.
 */
export function generateSyntheticTrajectories(params: GenerateSyntheticTrajectoriesParams): TrajectorySpan[] {
  const flows = params.flows ?? DEFAULT_MOVEMENT_FLOWS;
  if (flows.length === 0) {
    return [];
  }
  const rng = mulberry32(params.seed);
  const idPrefix = params.idPrefix ?? "syn";
  const baseTs = params.startTs ?? 1_700_000_000_000;
  const trajectories: TrajectorySpan[] = [];

  for (let i = 0; i < params.sequenceCount; i += 1) {
    const flow = flows[Math.floor(rng() * flows.length) % flows.length]!;
    let ts = baseTs + i * 60_000;
    const actions: TrajectoryAction[] = flow.steps.map((step) => {
      const target = step.targets[Math.floor(rng() * step.targets.length) % step.targets.length]!;
      ts += 250 + Math.floor(rng() * 500);
      return {
        kind: "action",
        tool: "device",
        summary: summarizeStep(step, target),
        ts,
        metadata: {
          gesture: step.gesture,
          target,
          ...(step.direction ? { direction: step.direction } : {}),
          flow: flow.name,
        },
      };
    });

    trajectories.push(
      buildTrajectorySpan({
        id: `${idPrefix}-${flow.name}-${i}`,
        sessionId: `${idPrefix}-session-${i}`,
        captureTier: "full",
        actions,
        outcome: { status: "success", summary: `completed ${flow.name} flow`, reward: 1 },
      }),
    );
  }

  return trajectories;
}

function summarizeStep(step: MovementFlowStep, target: string): string {
  switch (step.gesture) {
    case "tap":
      return `tapped ${target}`;
    case "type":
      return `typed into ${target}`;
    case "scroll":
      return `scrolled ${step.direction ?? "down"} ${target}`;
    case "swipe":
      return `swiped ${step.direction ?? "left"} ${target}`;
    case "shortcut":
      return `triggered ${target}`;
  }
}

/**
 * Small, fast, deterministic PRNG (mulberry32). Used instead of `Math.random`
 * so synthetic datasets — and therefore eval numbers — are reproducible.
 */
export function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
