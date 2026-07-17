import type { ExportedReplayManifest } from "./export-manifest.js";
import type { ReplayManifest } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";
import type { MovementSequence, MovementStep } from "./movement-model.js";

/**
 * Bridges captured data into the movement-model training format, and provides a
 * deterministic synthetic event-stream generator so the capture → dataset →
 * train → replay loop can be validated without any real OS input (the recorder
 * only produces real events on the user's local machine).
 */

type ReplayLike = Pick<ReplayManifest | ExportedReplayManifest, "events">;

/**
 * Extract one ordered {@link MovementSequence} per trajectory from replay
 * manifests, keeping only `action` events (the movements) in timestamp order.
 */
export function extractMovementSequences(replays: ReplayLike[]): MovementSequence[] {
  const byTrajectory = new Map<string, Array<{ ts: number; step: MovementStep }>>();
  for (const replay of replays) {
    for (const event of replay.events) {
      if (event.kind !== "action") {
        continue;
      }
      const bucket = byTrajectory.get(event.trajectoryId) ?? [];
      bucket.push({ ts: event.ts, step: { tool: event.tool, summary: event.summary } });
      byTrajectory.set(event.trajectoryId, bucket);
    }
  }
  return [...byTrajectory.entries()].map(([trajectoryId, entries]) => ({
    trajectoryId,
    steps: entries.sort((a, b) => a.ts - b.ts).map((entry) => entry.step),
  }));
}

/** Extract movement sequences directly from trajectory spans (pre-export). */
export function sequencesFromTrajectories(trajectories: TrajectorySpan[]): MovementSequence[] {
  return trajectories.map((trajectory) => ({
    trajectoryId: trajectory.id,
    steps: [...trajectory.actions]
      .sort((a, b) => a.ts - b.ts)
      .map((action) => ({ tool: action.tool, summary: action.summary })),
  }));
}

/** Tiny deterministic PRNG (mulberry32) so synthetic streams are reproducible. */
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

export type SyntheticMovementFlow = {
  name: string;
  /** Tool tokens in order. Optional branches introduce controlled variation. */
  tools: string[];
};

export type SyntheticMovementOptions = {
  /** Deterministic seed. Same seed + inputs → identical dataset. */
  seed?: number;
  /** How many sequences to emit. Defaults to 12. */
  count?: number;
  /** Flow templates to sample from. Defaults to {@link DEFAULT_SYNTHETIC_FLOWS}. */
  flows?: SyntheticMovementFlow[];
  /** Base timestamp (ms) for the first event. Defaults to 0. */
  startTs?: number;
};

/**
 * A small library of related UI flows (open an app, save a file, switch
 * windows). They share overlapping movement vocabulary so a trained model can
 * both repeat them and generalize across them.
 */
export const DEFAULT_SYNTHETIC_FLOWS: SyntheticMovementFlow[] = [
  { name: "launch-app", tools: ["window.focus", "mouse.move", "mouse.click", "app.launch", "window.focus"] },
  { name: "save-file", tools: ["window.focus", "key.combo", "key.type", "mouse.move", "mouse.click"] },
  { name: "switch-window", tools: ["key.combo", "window.focus", "mouse.move", "mouse.click"] },
  { name: "search-select", tools: ["mouse.click", "key.type", "key.press", "mouse.move", "mouse.click"] },
];

/**
 * Generate a deterministic synthetic movement dataset. Each sequence is a real
 * flow template with light, seeded jitter (an occasional extra `mouse.move`),
 * producing repeatable-yet-varied streams to validate the training pipeline.
 */
export function generateSyntheticMovementSequences(
  options: SyntheticMovementOptions = {},
): MovementSequence[] {
  const flows = options.flows ?? DEFAULT_SYNTHETIC_FLOWS;
  if (flows.length === 0) {
    return [];
  }
  const count = Math.max(0, Math.floor(options.count ?? 12));
  const random = createPrng(options.seed ?? 1);
  const startTs = options.startTs ?? 0;
  const sequences: MovementSequence[] = [];
  let ts = startTs;

  for (let i = 0; i < count; i += 1) {
    const flow = flows[Math.floor(random() * flows.length)]!;
    const steps: MovementStep[] = [];
    for (let stepIndex = 0; stepIndex < flow.tools.length; stepIndex += 1) {
      const tool = flow.tools[stepIndex]!;
      ts += 10 + Math.floor(random() * 40);
      steps.push({ tool, summary: `${flow.name}:${tool}#${stepIndex} @${ts}` });
      // Seeded jitter: sometimes insert a redundant micro-movement.
      if (tool === "mouse.move" && random() < 0.25) {
        ts += 5;
        steps.push({ tool: "mouse.move", summary: `${flow.name}:mouse.move.jitter @${ts}` });
      }
    }
    sequences.push({ trajectoryId: `synthetic-${flow.name}-${i}`, steps });
  }
  return sequences;
}
