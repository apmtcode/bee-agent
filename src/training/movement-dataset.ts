/**
 * Adapters and a deterministic synthetic generator for movement datasets.
 *
 * Bridges the existing capture schema ({@link TrajectorySpan},
 * {@link ReplayManifest}) into the movement-policy {@link MovementDataset}
 * format, and provides a seeded synthetic event-stream generator so the
 * capture -> dataset -> train -> replay loop can be validated in the cloud with
 * no real OS input (standing objective #2, and the roadmap "synthetic
 * event-stream generator" item).
 */

import type { ReplayManifest } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";
import type { MovementDataset, MovementEvent, MovementTrajectory } from "./movement-policy.js";

/** Derive a movement trajectory from a captured {@link TrajectorySpan}. */
export function movementTrajectoryFromSpan(span: TrajectorySpan): MovementTrajectory {
  const events: MovementEvent[] = [
    ...span.observations.map<MovementEvent>((observation) => ({
      ts: observation.ts,
      channel: "observation",
      action: observation.source,
      target: observation.summary,
    })),
    ...span.actions.map<MovementEvent>((action) => ({
      ts: action.ts,
      channel: "tool",
      action: action.tool,
      target: action.summary,
    })),
  ].sort((a, b) => a.ts - b.ts);
  return { id: span.id, events };
}

export function movementDatasetFromSpans(spans: TrajectorySpan[]): MovementDataset {
  return { version: 1, trajectories: spans.map(movementTrajectoryFromSpan) };
}

/** Derive a movement trajectory from a replay manifest's timeline. */
export function movementTrajectoryFromReplay(manifest: ReplayManifest): MovementTrajectory {
  const events: MovementEvent[] = manifest.events.map((event) => {
    switch (event.kind) {
      case "action":
        return { ts: event.ts, channel: "tool", action: event.tool, target: event.summary };
      case "observation":
        return { ts: event.ts, channel: "observation", action: event.source, target: event.summary };
      case "transcript":
        return { ts: event.ts, channel: "window", action: event.role, target: event.messageId };
    }
  });
  return { id: manifest.sessionId, events };
}

/**
 * A tiny deterministic PRNG (mulberry32) so synthetic datasets are stable
 * across runs and machines — no `Math.random`, no wall-clock dependence.
 */
export function createSeededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export type SyntheticMovementOptions = {
  seed?: number;
  trajectoryCount?: number;
  minLength?: number;
  maxLength?: number;
  /** Named UI "flows" whose ordered steps the generator samples and perturbs. */
  flows?: MovementEvent[][];
};

const DEFAULT_FLOWS: MovementEvent[][] = [
  [
    { ts: 0, channel: "window", action: "focus", target: "editor" },
    { ts: 1, channel: "pointer", action: "move", target: "toolbar" },
    { ts: 2, channel: "pointer", action: "click", target: "save" },
    { ts: 3, channel: "keyboard", action: "keydown", target: "cmd+s" },
    { ts: 4, channel: "observation", action: "toast", target: "saved" },
  ],
  [
    { ts: 0, channel: "window", action: "focus", target: "browser" },
    { ts: 1, channel: "pointer", action: "move", target: "urlbar" },
    { ts: 2, channel: "pointer", action: "click", target: "urlbar" },
    { ts: 3, channel: "keyboard", action: "type", target: "url" },
    { ts: 4, channel: "keyboard", action: "keydown", target: "enter" },
    { ts: 5, channel: "observation", action: "load", target: "page" },
  ],
];

/**
 * Generate a deterministic synthetic movement dataset by sampling named UI
 * flows and applying small perturbations (truncation, repeated steps). The
 * repeated structure is learnable, so a trained policy scores well above
 * chance — which is exactly what makes this useful as a test fixture and a
 * generalization probe.
 */
export function generateSyntheticMovementDataset(
  options: SyntheticMovementOptions = {},
): MovementDataset {
  const random = createSeededRandom(options.seed ?? 1);
  const flows = options.flows ?? DEFAULT_FLOWS;
  const trajectoryCount = options.trajectoryCount ?? 12;
  const minLength = options.minLength ?? 3;
  const maxLength = options.maxLength ?? Math.max(...flows.map((flow) => flow.length));

  const trajectories: MovementTrajectory[] = [];
  for (let index = 0; index < trajectoryCount; index += 1) {
    const flow = flows[Math.floor(random() * flows.length)] ?? flows[0]!;
    const length = clamp(
      minLength + Math.floor(random() * (maxLength - minLength + 1)),
      1,
      flow.length,
    );
    const events: MovementEvent[] = [];
    let ts = 0;
    for (let step = 0; step < length; step += 1) {
      const base = flow[step]!;
      events.push({ ...base, ts });
      ts += 1;
      // Occasionally repeat a pointer move (a realistic, learnable perturbation).
      if (base.channel === "pointer" && base.action === "move" && random() < 0.25) {
        events.push({ ...base, ts });
        ts += 1;
      }
    }
    trajectories.push({ id: `synthetic-${index}`, events });
  }

  return { version: 1, trajectories };
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}
