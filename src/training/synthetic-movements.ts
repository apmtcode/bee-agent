/**
 * Deterministic synthetic movement generator.
 *
 * bee-agent runs in Anthropic's cloud with no access to a real machine, so the
 * capture → dataset → replay → train → infer loop must be validated on
 * *simulated* event streams. This module emits reproducible {@link TrajectorySpan}
 * objects for a small library of "task families" (each a canonical movement
 * grammar with bounded variation), so tests can:
 *   - build a movement dataset and train the local model on it,
 *   - hold out related-but-unseen variants and measure generalization fidelity,
 *   - exercise the recorder/replay/exporter pipeline without any OS hooks.
 *
 * A seeded PRNG (mulberry32) keeps every run identical for a given seed.
 */

import { buildTrajectorySpan } from "../capture/trajectory.js";
import type {
  CaptureTier,
  TrajectoryAction,
  TrajectoryObservation,
  TrajectorySpan,
} from "../capture/trajectory.js";

/** A canonical movement grammar: an ordered spine of steps plus optional inserts. */
export type MovementTaskFamily = {
  id: string;
  /** Ordered required steps every trajectory in the family performs. */
  spine: MovementStep[];
  /** Optional steps randomly inserted between spine steps to create related variation. */
  inserts?: MovementStep[];
  /** Terminal outcome the family drives toward. */
  outcome: "success" | "failure";
};

export type MovementStep = {
  kind: "action" | "observation";
  /** Tool (for actions) or source (for observations). */
  channel: string;
  summary: string;
};

export type SyntheticTrajectoryOptions = {
  family: MovementTaskFamily;
  seed: number;
  count: number;
  /** Milliseconds between successive events. Default 250. */
  stepIntervalMs?: number;
  /** Epoch ms for the first event of the first trajectory. Default 1_700_000_000_000. */
  startTs?: number;
  captureTier?: CaptureTier;
  /** Probability (0..1) that an optional insert is emitted at each gap. Default 0.4. */
  insertProbability?: number;
  /** Prefix for generated trajectory ids. Default the family id. */
  idPrefix?: string;
};

/** Built-in task families covering distinct movement shapes. */
export const SYNTHETIC_TASK_FAMILIES: Record<string, MovementTaskFamily> = {
  fileSave: {
    id: "file-save",
    spine: [
      { kind: "observation", channel: "screen", summary: "editor focused" },
      { kind: "action", channel: "focusWindow", summary: "focus editor window" },
      { kind: "action", channel: "keyboard", summary: "type document body" },
      { kind: "action", channel: "hotkey", summary: "press save shortcut" },
      { kind: "observation", channel: "screen", summary: "saved indicator visible" },
    ],
    inserts: [
      { kind: "action", channel: "mouse", summary: "reposition cursor" },
      { kind: "observation", channel: "screen", summary: "toolbar hover" },
    ],
    outcome: "success",
  },
  browserSearch: {
    id: "browser-search",
    spine: [
      { kind: "action", channel: "focusWindow", summary: "focus browser window" },
      { kind: "action", channel: "hotkey", summary: "open address bar" },
      { kind: "action", channel: "keyboard", summary: "type query" },
      { kind: "action", channel: "keyboard", summary: "press enter" },
      { kind: "observation", channel: "screen", summary: "results rendered" },
      { kind: "action", channel: "mouse", summary: "click first result" },
    ],
    inserts: [
      { kind: "observation", channel: "screen", summary: "autocomplete dropdown" },
      { kind: "action", channel: "scroll", summary: "scroll results" },
    ],
    outcome: "success",
  },
};

/** Generate a reproducible set of related trajectories for a task family. */
export function generateSyntheticTrajectories(options: SyntheticTrajectoryOptions): TrajectorySpan[] {
  const stepIntervalMs = options.stepIntervalMs ?? 250;
  const startTs = options.startTs ?? 1_700_000_000_000;
  const insertProbability = options.insertProbability ?? 0.4;
  const idPrefix = options.idPrefix ?? options.family.id;
  const rng = mulberry32(options.seed);

  const spans: TrajectorySpan[] = [];
  let ts = startTs;

  for (let index = 0; index < options.count; index += 1) {
    const observations: TrajectoryObservation[] = [];
    const actions: TrajectoryAction[] = [];

    const emit = (step: MovementStep): void => {
      if (step.kind === "action") {
        actions.push({ kind: "action", tool: step.channel, summary: step.summary, ts });
      } else {
        observations.push({ kind: "observation", source: step.channel, summary: step.summary, ts });
      }
      ts += stepIntervalMs;
    };

    for (const step of options.family.spine) {
      emit(step);
      const inserts = options.family.inserts ?? [];
      if (inserts.length > 0 && rng() < insertProbability) {
        const insert = inserts[Math.floor(rng() * inserts.length)]!;
        emit(insert);
      }
    }

    spans.push(
      buildTrajectorySpan({
        id: `${idPrefix}-${index}`,
        sessionId: `synthetic-${idPrefix}`,
        captureTier: options.captureTier ?? "app",
        observations,
        actions,
        outcome: {
          status: options.family.outcome,
          summary: `${options.family.id} completed`,
          reward: options.family.outcome === "success" ? 1 : 0,
        },
      }),
    );

    // Gap between trajectories so timestamps never collide across spans.
    ts += stepIntervalMs * 4;
  }

  return spans;
}

/** Small, fast, seedable PRNG. Deterministic for reproducible tests. */
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
