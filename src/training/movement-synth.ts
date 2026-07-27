/**
 * Local-movement learning subsystem — synthetic event streams.
 *
 * We run in the cloud with no access to the user's real machine, so we can't
 * capture genuine mouse/keyboard/window events. This module deterministically
 * synthesizes movement streams from a small grammar of UI "tasks" (focus →
 * click → type → submit, etc.) so the capture→dataset→train→infer→generalize
 * pipeline can be validated end to end. It is seeded (no Math.random), so the
 * same seed always yields the same streams — essential for reproducible tests
 * and for the resume-safe engine.
 */
import type { ReplayTimelineEvent } from "../capture/replay.js";

/** A deterministic 32-bit LCG — same seed ⇒ same stream, no global RNG. */
class SeededRandom {
  private state: number;

  constructor(seed: number) {
    this.state = (seed >>> 0) || 0x9e3779b9;
  }

  next(): number {
    // Numerical Recipes LCG constants.
    this.state = (Math.imul(this.state, 1664525) + 1013904223) >>> 0;
    return this.state / 0x100000000;
  }

  int(maxExclusive: number): number {
    return Math.floor(this.next() * maxExclusive);
  }

  pick<T>(items: readonly T[]): T {
    return items[this.int(items.length)]!;
  }
}

type MovementStep = {
  kind: "observation" | "action";
  channel: string;
  verb: string;
};

/** A named UI task = an ordered movement grammar with slot variation. */
export type MovementTaskTemplate = {
  name: string;
  steps: MovementStep[];
  targets: string[];
};

export const DEFAULT_MOVEMENT_TASKS: MovementTaskTemplate[] = [
  {
    name: "open-and-submit-form",
    steps: [
      { kind: "observation", channel: "os", verb: "focused" },
      { kind: "action", channel: "mouse", verb: "clicked" },
      { kind: "action", channel: "keyboard", verb: "typed" },
      { kind: "action", channel: "mouse", verb: "clicked" },
      { kind: "observation", channel: "app", verb: "confirmed" },
    ],
    targets: ["login-panel", "settings-dialog", "profile-editor", "billing-form"],
  },
  {
    name: "browse-and-scroll",
    steps: [
      { kind: "observation", channel: "browser", verb: "opened" },
      { kind: "action", channel: "mouse", verb: "scrolled" },
      { kind: "action", channel: "mouse", verb: "hovered" },
      { kind: "action", channel: "mouse", verb: "clicked" },
    ],
    targets: ["dashboard", "report-page", "feed", "search-results"],
  },
  {
    name: "keyboard-shortcut-flow",
    steps: [
      { kind: "observation", channel: "os", verb: "focused" },
      { kind: "action", channel: "keyboard", verb: "pressed" },
      { kind: "action", channel: "keyboard", verb: "typed" },
      { kind: "action", channel: "keyboard", verb: "pressed" },
      { kind: "observation", channel: "app", verb: "saved" },
    ],
    targets: ["editor", "terminal", "notebook", "canvas"],
  },
];

export type SyntheticMovementSequence = {
  id: string;
  taskName: string;
  events: ReplayTimelineEvent[];
};

export type GenerateSyntheticMovementsOptions = {
  seed?: number;
  sequenceCount?: number;
  tasks?: MovementTaskTemplate[];
  /** Base timestamp; each event advances by a small deterministic delta. */
  baseTs?: number;
};

/**
 * Generate `sequenceCount` movement streams by sampling task templates. Each
 * step becomes a replay event whose summary embeds a per-sequence target, so
 * the normalized tokens stay stable (the verb) while surface text varies —
 * exactly the shape the Markov backoff must generalize over.
 */
export function generateSyntheticMovements(
  options: GenerateSyntheticMovementsOptions = {},
): SyntheticMovementSequence[] {
  const rng = new SeededRandom(options.seed ?? 1);
  const tasks = options.tasks ?? DEFAULT_MOVEMENT_TASKS;
  const sequenceCount = options.sequenceCount ?? 12;
  const baseTs = options.baseTs ?? 0;

  const sequences: SyntheticMovementSequence[] = [];
  let ts = baseTs;

  for (let index = 0; index < sequenceCount; index += 1) {
    const task = rng.pick(tasks);
    const target = rng.pick(task.targets);
    const events: ReplayTimelineEvent[] = task.steps.map((step, stepIndex) => {
      ts += 1 + rng.int(5);
      const summary = `${step.verb} ${target}#${stepIndex}`;
      if (step.kind === "observation") {
        return {
          kind: "observation",
          ts,
          trajectoryId: `synthetic-${index}`,
          source: step.channel,
          summary,
        };
      }
      return {
        kind: "action",
        ts,
        trajectoryId: `synthetic-${index}`,
        tool: step.channel,
        summary,
      };
    });
    sequences.push({ id: `synthetic-${index}`, taskName: task.name, events });
  }

  return sequences;
}
