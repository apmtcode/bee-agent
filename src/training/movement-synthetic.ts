import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";

/**
 * Deterministic synthetic movement-event generator.
 *
 * Produces {@link ReplayManifest}s from a small task "grammar" so the whole
 * capture → dataset → train → replay/generalize loop can be validated in the
 * cloud without any real OS input. A seeded PRNG makes runs reproducible, and a
 * controllable variation rate lets tests build *related-but-unseen* trajectories
 * (for generalization eval) from the same underlying task.
 */

/** A named phase of a task, each contributing one action + optional observation. */
export type MovementPhase = {
  /** Tool driving the action event (→ `action:<tool>` token). */
  tool: string;
  summary: string;
  /** Optional observation source preceding the action (→ `observation:<source>`). */
  observe?: string;
};

export type SyntheticTaskGrammar = {
  name: string;
  phases: MovementPhase[];
  /** Optional alternative tools per phase index, used when a variation fires. */
  alternates?: Record<number, string[]>;
};

export type SyntheticGenerateOptions = {
  grammar: SyntheticTaskGrammar;
  count: number;
  seed?: number;
  /** 0..1 chance each phase swaps to an alternate tool (drives generalization). */
  variationRate?: number;
  /** Milliseconds between successive events on the synthetic timeline. */
  stepMs?: number;
  /** Epoch ms of the first event (kept explicit so output is deterministic). */
  startTs?: number;
};

/** Small, fast, deterministic PRNG (mulberry32) — no global RNG dependency. */
function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** The canonical "open an app, edit, save" task used as a default fixture. */
export const DEFAULT_TASK_GRAMMAR: SyntheticTaskGrammar = {
  name: "edit-and-save",
  phases: [
    { tool: "focus-window", summary: "focus editor window", observe: "window-manager" },
    { tool: "mouse-click", summary: "click into document body" },
    { tool: "key-type", summary: "type revised paragraph", observe: "text-field" },
    { tool: "key-chord", summary: "press save shortcut" },
    { tool: "mouse-move", summary: "move to confirmation toast", observe: "notification" },
  ],
  alternates: {
    1: ["mouse-double-click", "touch-tap"],
    3: ["menu-select", "toolbar-click"],
  },
};

export function generateSyntheticReplays(options: SyntheticGenerateOptions): ReplayManifest[] {
  const { grammar, count } = options;
  const random = mulberry32(options.seed ?? 1);
  const variationRate = options.variationRate ?? 0;
  const stepMs = options.stepMs ?? 250;
  const startTs = options.startTs ?? 0;

  const manifests: ReplayManifest[] = [];
  for (let index = 0; index < count; index += 1) {
    const events: ReplayTimelineEvent[] = [];
    const sessionId = `${grammar.name}-${index}`;
    let ts = startTs;

    grammar.phases.forEach((phase, phaseIndex) => {
      let tool = phase.tool;
      const alternates = grammar.alternates?.[phaseIndex];
      if (alternates && alternates.length > 0 && random() < variationRate) {
        tool = alternates[Math.floor(random() * alternates.length) % alternates.length]!;
      }

      if (phase.observe) {
        events.push({
          kind: "observation",
          ts,
          trajectoryId: sessionId,
          source: phase.observe,
          summary: `${phase.observe} state`,
        });
        ts += stepMs;
      }
      events.push({
        kind: "action",
        ts,
        trajectoryId: sessionId,
        tool,
        summary: phase.summary,
      });
      ts += stepMs;
    });

    manifests.push({
      version: 1,
      sessionId,
      trajectoryIds: [sessionId],
      eventCount: events.length,
      events,
    });
  }
  return manifests;
}
