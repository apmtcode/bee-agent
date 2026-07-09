import type { ReplayManifest, ReplayTimelineEvent } from "../../capture/replay.js";

/**
 * Deterministic synthetic movement-stream generator. Because the self-evolution
 * engine runs in the cloud with no access to a real machine, this produces
 * repeatable synthetic replay manifests — the same structure the on-device
 * capture pipeline emits — so the dataset → train → infer → eval loop can be
 * validated end-to-end without any real OS input. No randomness is used: streams
 * are a pure function of their parameters, so tests are stable.
 */

export type SyntheticMovementTemplate = {
  /** Stable name used to derive trajectory ids. */
  name: string;
  /** Observation summary presented at the start of the trajectory. */
  observation: string;
  /** Ordered actions, each an `{ tool, summary }` movement. */
  actions: Array<{ tool: string; summary: string }>;
};

export type SyntheticMovementOptions = {
  sessionId?: string;
  templates: readonly SyntheticMovementTemplate[];
  /** How many trajectories to emit per template (default 1). */
  repeatsPerTemplate?: number;
  /** Base timestamp; each event advances by `stepMs` (default 1000). */
  baseTs?: number;
  stepMs?: number;
};

/**
 * Build one replay manifest per generated trajectory. Each trajectory starts with
 * an observation event followed by its action events, with monotonically
 * increasing timestamps.
 */
export function generateSyntheticReplays(options: SyntheticMovementOptions): ReplayManifest[] {
  const sessionId = options.sessionId ?? "synthetic-session";
  const repeats = Math.max(1, Math.floor(options.repeatsPerTemplate ?? 1));
  const stepMs = options.stepMs ?? 1000;
  let ts = options.baseTs ?? 1_000_000;

  const manifests: ReplayManifest[] = [];
  for (const template of options.templates) {
    for (let repeat = 0; repeat < repeats; repeat += 1) {
      const trajectoryId = `${template.name}-${repeat}`;
      const events: ReplayTimelineEvent[] = [];
      events.push({
        kind: "observation",
        ts,
        trajectoryId,
        source: "synthetic",
        summary: template.observation,
      });
      ts += stepMs;
      for (const action of template.actions) {
        events.push({
          kind: "action",
          ts,
          trajectoryId,
          tool: action.tool,
          summary: action.summary,
        });
        ts += stepMs;
      }
      manifests.push({
        version: 1,
        sessionId,
        trajectoryIds: [trajectoryId],
        eventCount: events.length,
        events,
      });
    }
  }

  return manifests;
}
