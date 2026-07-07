import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";

/**
 * Deterministic synthetic movement-stream generator. bee-agent runs in the cloud
 * with no access to a real mouse/keyboard, so the capture → dataset → replay →
 * train loop is validated against simulated event streams produced here. Given
 * the same seed, output is byte-for-byte reproducible (no global RNG), which
 * keeps CI/cloud tests stable.
 *
 * Each "movement pattern" is a small grammar over device actions (e.g. a
 * point-and-click, a form fill, a copy-paste). Sampling a pattern many times with
 * slight positional/timing jitter yields a family of *related-but-distinct*
 * trajectories — exactly what the generalization objective needs.
 */

export type SyntheticMovementStep = {
  kind: "action" | "observation";
  /** Tool for actions (e.g. "mouse.move"), source for observations (e.g. "screen"). */
  channel: string;
  summary: string;
};

export type SyntheticMovementPattern = {
  name: string;
  steps: SyntheticMovementStep[];
};

/** A small library of canonical UI movement patterns. */
export const SYNTHETIC_MOVEMENT_PATTERNS: SyntheticMovementPattern[] = [
  {
    name: "point-and-click",
    steps: [
      { kind: "observation", channel: "screen", summary: "target visible" },
      { kind: "action", channel: "mouse.move", summary: "move to target" },
      { kind: "action", channel: "mouse.click", summary: "click target" },
      { kind: "observation", channel: "screen", summary: "target activated" },
    ],
  },
  {
    name: "form-fill",
    steps: [
      { kind: "action", channel: "mouse.click", summary: "focus field" },
      { kind: "action", channel: "keyboard.type", summary: "enter text" },
      { kind: "action", channel: "keyboard.key", summary: "press tab" },
      { kind: "action", channel: "keyboard.type", summary: "enter text" },
      { kind: "action", channel: "mouse.click", summary: "submit" },
    ],
  },
  {
    name: "copy-paste",
    steps: [
      { kind: "action", channel: "mouse.drag", summary: "select text" },
      { kind: "action", channel: "keyboard.shortcut", summary: "copy" },
      { kind: "action", channel: "mouse.click", summary: "focus destination" },
      { kind: "action", channel: "keyboard.shortcut", summary: "paste" },
    ],
  },
];

export type GenerateSyntheticOptions = {
  seed: number;
  /** Number of trajectories to emit. */
  count: number;
  /** Restrict to these pattern names (defaults to all). */
  patterns?: string[];
  /** Timestamp step between events (ms). */
  timeStepMs?: number;
};

/** One generated trajectory as an ordered timeline, ready for tokenization/replay. */
export type SyntheticTrajectory = {
  id: string;
  pattern: string;
  events: ReplayTimelineEvent[];
};

export function generateSyntheticTrajectories(options: GenerateSyntheticOptions): SyntheticTrajectory[] {
  const timeStep = options.timeStepMs ?? 50;
  const library = options.patterns
    ? SYNTHETIC_MOVEMENT_PATTERNS.filter((pattern) => options.patterns!.includes(pattern.name))
    : SYNTHETIC_MOVEMENT_PATTERNS;
  if (library.length === 0) {
    return [];
  }

  let state = (options.seed >>> 0) || 0x9e3779b9;
  const trajectories: SyntheticTrajectory[] = [];
  for (let index = 0; index < options.count; index += 1) {
    state = nextRng(state);
    const pattern = library[state % library.length]!;
    const trajectoryId = `synthetic-${pattern.name}-${index}`;
    const events: ReplayTimelineEvent[] = pattern.steps.map((step, stepIndex) => {
      const ts = index * 10_000 + stepIndex * timeStep;
      if (step.kind === "action") {
        return {
          kind: "action",
          ts,
          trajectoryId,
          tool: step.channel,
          summary: step.summary,
        };
      }
      return {
        kind: "observation",
        ts,
        trajectoryId,
        source: step.channel,
        summary: step.summary,
      };
    });
    trajectories.push({ id: trajectoryId, pattern: pattern.name, events });
  }
  return trajectories;
}

/** Wrap generated trajectories into a single replay manifest (dataset unit). */
export function syntheticReplayManifest(
  sessionId: string,
  trajectories: SyntheticTrajectory[],
): ReplayManifest {
  const events = trajectories.flatMap((trajectory) => trajectory.events);
  return {
    version: 1,
    sessionId,
    trajectoryIds: trajectories.map((trajectory) => trajectory.id),
    eventCount: events.length,
    events,
  };
}

function nextRng(state: number): number {
  let x = state === 0 ? 0x9e3779b9 : state;
  x ^= x << 13;
  x ^= x >>> 17;
  x ^= x << 5;
  return x >>> 0;
}
