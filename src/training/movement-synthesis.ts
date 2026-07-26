import type { TrajectoryAction } from "../capture/trajectory.js";
import type { MovementTrajectoryInput } from "./movement-policy.js";

/**
 * Synthetic movement-stream generator.
 *
 * bee-agent has no access to the user's machine in the cloud, so real mouse /
 * keyboard / gesture capture cannot run here. This deterministic generator
 * fabricates recorded-looking movement trajectories from gesture *templates*
 * instantiated over per-slot target vocabularies, letting the capture -> dataset
 * -> train -> infer pipeline be validated end-to-end in CI. Because targets are
 * drawn per-slot, holding out a vocabulary produces trajectories with the same
 * gesture grammar over unseen targets — exactly the "new but related movement"
 * case objective #2(d) must generalize to.
 *
 * Determinism comes from a seeded PRNG (no Math.random), so generated datasets
 * are byte-stable across runs.
 */

export type MovementStep = {
  tool: string;
  gesture: string;
  direction?: "up" | "down" | "left" | "right";
  /** Name of the vocabulary slot to draw this step's target from, if any. */
  targetSlot?: string;
};

export type MovementTemplate = {
  name: string;
  steps: MovementStep[];
};

export type SyntheticMovementParams = {
  templates: MovementTemplate[];
  /** slot name -> candidate targets to sample from. */
  targetVocabulary: Record<string, string[]>;
  count: number;
  seed: number;
  sessionPrefix?: string;
  /** Base epoch ms for the first event; each step advances by stepIntervalMs. */
  startTs?: number;
  stepIntervalMs?: number;
};

/** Deterministic 32-bit PRNG (mulberry32) — seeded, no global randomness. */
function createPrng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function pick<T>(rng: () => number, items: T[]): T | undefined {
  if (items.length === 0) {
    return undefined;
  }
  return items[Math.floor(rng() * items.length) % items.length];
}

export function generateSyntheticMovementTrajectories(params: SyntheticMovementParams): MovementTrajectoryInput[] {
  const rng = createPrng(params.seed);
  const prefix = params.sessionPrefix ?? "synthetic";
  const startTs = params.startTs ?? 1_700_000_000_000;
  const stepInterval = params.stepIntervalMs ?? 250;
  const trajectories: MovementTrajectoryInput[] = [];

  for (let index = 0; index < params.count; index += 1) {
    const template = pick(rng, params.templates);
    if (!template) {
      break;
    }
    const trajectoryTs = startTs + index * template.steps.length * stepInterval;
    const actions: TrajectoryAction[] = template.steps.map((step, stepIndex) => {
      const target = step.targetSlot ? pick(rng, params.targetVocabulary[step.targetSlot] ?? []) : undefined;
      const metadata: Record<string, unknown> = { gesture: step.gesture };
      if (step.direction) {
        metadata.direction = step.direction;
      }
      if (target) {
        metadata.target = target;
      }
      return {
        kind: "action",
        tool: step.tool,
        summary: buildSummary(step, target),
        ts: trajectoryTs + stepIndex * stepInterval,
        metadata,
      } satisfies TrajectoryAction;
    });
    trajectories.push({ id: `${prefix}-${template.name}-${index}`, actions });
  }

  return trajectories;
}

function buildSummary(step: MovementStep, target: string | undefined): string {
  const where = target ? ` ${target}` : "";
  const dir = step.direction ? ` ${step.direction}` : "";
  return `${step.gesture}${dir}${where}`.trim();
}

/** A small ready-made template library covering common desktop movement idioms. */
export const DEFAULT_MOVEMENT_TEMPLATES: MovementTemplate[] = [
  {
    name: "open-and-save",
    steps: [
      { tool: "device", gesture: "tap", targetSlot: "menu" },
      { tool: "device", gesture: "tap", targetSlot: "item" },
      { tool: "device", gesture: "type", targetSlot: "field" },
      { tool: "device", gesture: "tap", targetSlot: "confirm" },
    ],
  },
  {
    name: "scroll-and-select",
    steps: [
      { tool: "device", gesture: "scroll", direction: "down" },
      { tool: "device", gesture: "scroll", direction: "down" },
      { tool: "device", gesture: "tap", targetSlot: "item" },
    ],
  },
];
