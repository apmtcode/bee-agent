import type {
  DeviceGestureKind,
  DevicePlatform,
} from "../capture/device-adapter.js";
import { buildTrajectorySpan, type TrajectoryAction, type TrajectorySpan } from "../capture/trajectory.js";

/**
 * Deterministic synthetic movement-stream generator.
 *
 * Standing objective #2 forbids relying on the user's real machine, so we
 * validate the capture -> dataset -> train -> replay -> generalize pipeline with
 * reproducible synthetic trajectories. Generation is fully seeded (a small xorshift
 * PRNG) so the same seed always yields the same dataset — no `Math.random`/`Date`.
 */

/** A reusable, named movement pattern ("skill") expressed as a step template. */
export type MovementSkillTemplate = {
  name: string;
  tool: string;
  steps: {
    action: DeviceGestureKind | string;
    /** Candidate targets; one is chosen per generated instance. */
    targets?: string[];
    direction?: "up" | "down" | "left" | "right";
  }[];
};

export type SyntheticMovementOptions = {
  /** Deterministic seed. */
  seed?: number;
  /** Number of trajectories to generate. Defaults to 8. */
  count?: number;
  /** Skill templates to sample from. Defaults to {@link DEFAULT_SKILL_TEMPLATES}. */
  templates?: MovementSkillTemplate[];
  /** Session id prefix. Defaults to "synthetic". */
  sessionPrefix?: string;
  /** Simulated platform recorded in metadata. Defaults to "macos". */
  platform?: DevicePlatform;
  /** Base epoch ms for the first event. Defaults to 0 (deterministic). */
  baseTs?: number;
};

/** A small xorshift32 PRNG — deterministic and dependency-free. */
class SeededRandom {
  private state: number;

  constructor(seed: number) {
    // Avoid a zero state, which would lock xorshift at 0.
    this.state = (seed | 0) === 0 ? 0x9e3779b9 : seed | 0;
  }

  next(): number {
    let x = this.state;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.state = x | 0;
    // Map to [0, 1).
    return ((x >>> 0) % 0xffffffff) / 0xffffffff;
  }

  int(maxExclusive: number): number {
    if (maxExclusive <= 0) return 0;
    return Math.floor(this.next() * maxExclusive) % maxExclusive;
  }

  pick<T>(items: readonly T[]): T {
    return items[this.int(items.length)]!;
  }
}

export const DEFAULT_SKILL_TEMPLATES: MovementSkillTemplate[] = [
  {
    name: "compose-and-send",
    tool: "device",
    steps: [
      { action: "tap", targets: ["compose-button", "new-message"] },
      { action: "type", targets: ["recipient-field", "to-field"] },
      { action: "type", targets: ["body-field", "message-body"] },
      { action: "tap", targets: ["send-button"] },
    ],
  },
  {
    name: "scroll-and-open",
    tool: "device",
    steps: [
      { action: "scroll", direction: "down" },
      { action: "scroll", direction: "down" },
      { action: "tap", targets: ["list-item", "result-row"] },
      { action: "tap", targets: ["open-button", "detail-link"] },
    ],
  },
  {
    name: "save-document",
    tool: "device",
    steps: [
      { action: "tap", targets: ["editor-canvas"] },
      { action: "type", targets: ["editor-canvas"] },
      { action: "shortcut", targets: ["save"] },
      { action: "tap", targets: ["confirm-button", "ok-button"] },
    ],
  },
];

function summarizeStep(action: string, target: string | undefined, direction: string | undefined): string {
  switch (action) {
    case "tap":
      return target ? `tapped ${target}` : "tapped device";
    case "swipe":
      return direction ? `swiped ${direction}` : "swiped device";
    case "scroll":
      return direction ? `scrolled ${direction}` : "scrolled device";
    case "type":
      return target ? `typed into ${target}` : "typed on device";
    case "shortcut":
      return target ? `triggered ${target}` : "triggered device shortcut";
    default:
      return target ? `${action} ${target}` : action;
  }
}

/**
 * Generate a deterministic set of synthetic movement trajectories. The same
 * `seed` + options always produce identical output, so they are safe to assert
 * against in tests and to use as held-out generalization sets.
 */
export function generateSyntheticTrajectories(options: SyntheticMovementOptions = {}): TrajectorySpan[] {
  const templates = options.templates ?? DEFAULT_SKILL_TEMPLATES;
  if (templates.length === 0) {
    return [];
  }
  const count = Math.max(0, options.count ?? 8);
  const rng = new SeededRandom(options.seed ?? 1);
  const platform = options.platform ?? "macos";
  const sessionPrefix = options.sessionPrefix ?? "synthetic";
  const baseTs = options.baseTs ?? 0;

  const trajectories: TrajectorySpan[] = [];
  let ts = baseTs;

  for (let i = 0; i < count; i += 1) {
    const template = rng.pick(templates);
    const actions: TrajectoryAction[] = template.steps.map((step) => {
      const target = step.targets ? rng.pick(step.targets) : undefined;
      const direction = step.direction;
      ts += 1 + rng.int(5);
      return {
        kind: "action",
        tool: template.tool,
        summary: summarizeStep(step.action, target, direction),
        ts,
        metadata: {
          gesture: step.action,
          platform,
          skill: template.name,
          ...(target ? { target } : {}),
          ...(direction ? { direction } : {}),
        },
      };
    });

    const span = buildTrajectorySpan({
      id: `${sessionPrefix}-traj-${i}`,
      sessionId: `${sessionPrefix}-session-${i}`,
      captureTier: "full",
      actions,
      outcome: { status: "success", summary: `completed ${template.name}` },
    });
    // Make `createdAt` deterministic too (buildTrajectorySpan stamps wall-clock),
    // so the same seed yields a byte-identical dataset.
    span.createdAt = new Date(baseTs + i).toISOString();
    trajectories.push(span);
  }

  return trajectories;
}
