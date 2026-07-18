import { buildTrajectorySpan, type TrajectoryAction, type TrajectorySpan } from "./trajectory.js";

/**
 * Deterministic synthetic movement/event-stream generator.
 *
 * bee-agent runs in the cloud with no access to a real machine, so the
 * capture→dataset→train→replay pipeline is validated against *simulated* event
 * streams produced here. Everything is seeded (mulberry32) so runs are
 * reproducible without `Math.random`.
 */

/** A named workflow: an ordered list of tool "movements" a user performs. */
export type SyntheticWorkflowTemplate = {
  name: string;
  tools: string[];
};

/**
 * A small library of realistic, overlapping desktop workflows. Overlap (shared
 * prefixes/suffixes across templates) is deliberate: it forces a movement model
 * to rely on longer context to disambiguate, which is what makes the
 * generalization eval meaningful.
 */
export function defaultWorkflowTemplates(): SyntheticWorkflowTemplate[] {
  return [
    { name: "save-document", tools: ["focus-window", "click-menu", "click-save", "type-filename", "press-enter"] },
    { name: "save-as-copy", tools: ["focus-window", "click-menu", "click-save-as", "type-filename", "press-enter"] },
    { name: "search-replace", tools: ["focus-window", "open-find", "type-query", "click-replace", "press-enter"] },
    { name: "open-file", tools: ["focus-window", "click-menu", "click-open", "type-filename", "press-enter"] },
    { name: "copy-paste", tools: ["select-text", "press-copy", "move-cursor", "press-paste"] },
  ];
}

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

export type GenerateSyntheticOptions = {
  /** Templates to sample from. Defaults to {@link defaultWorkflowTemplates}. */
  templates?: SyntheticWorkflowTemplate[];
  /** Number of trajectories to generate. */
  count: number;
  /** PRNG seed for reproducibility. Defaults to 1. */
  seed?: number;
  /** Base timestamp (ms) for the first action. Defaults to 0. */
  startTs?: number;
  /** Milliseconds between successive actions. Defaults to 1000. */
  stepMs?: number;
  /**
   * Probability [0,1] of inserting a benign noise movement between real steps.
   * Models real-world jitter and drives the generalization test. Defaults to 0.
   */
  noise?: number;
  /** Session id assigned to every generated trajectory. Defaults to "synthetic". */
  sessionId?: string;
};

const NOISE_TOOLS = ["idle", "hover", "scroll", "blink-cursor"] as const;

/**
 * Generate deterministic synthetic trajectories, one per template pick. The
 * output is ordinary {@link TrajectorySpan}s, so it flows through the exact same
 * dataset/replay/training code paths as real captured trajectories.
 */
export function generateSyntheticTrajectories(options: GenerateSyntheticOptions): TrajectorySpan[] {
  const templates = options.templates ?? defaultWorkflowTemplates();
  if (templates.length === 0) {
    throw new Error("generateSyntheticTrajectories: at least one template is required");
  }
  const rand = mulberry32(options.seed ?? 1);
  const startTs = options.startTs ?? 0;
  const stepMs = options.stepMs ?? 1000;
  const noise = Math.min(1, Math.max(0, options.noise ?? 0));
  const sessionId = options.sessionId ?? "synthetic";

  const trajectories: TrajectorySpan[] = [];
  for (let i = 0; i < options.count; i += 1) {
    const template = templates[Math.floor(rand() * templates.length) % templates.length]!;
    const actions: TrajectoryAction[] = [];
    let ts = startTs + i * stepMs * 100;

    for (let step = 0; step < template.tools.length; step += 1) {
      if (step > 0 && noise > 0 && rand() < noise) {
        const noiseTool = NOISE_TOOLS[Math.floor(rand() * NOISE_TOOLS.length) % NOISE_TOOLS.length]!;
        actions.push({
          kind: "action",
          tool: noiseTool,
          summary: `${template.name} noise ${noiseTool}`,
          ts,
          metadata: { synthetic: true, noise: true },
        });
        ts += stepMs;
      }
      const tool = template.tools[step]!;
      actions.push({
        kind: "action",
        tool,
        summary: `${template.name} step ${step} ${tool}`,
        ts,
        metadata: { synthetic: true, template: template.name, step },
      });
      ts += stepMs;
    }

    trajectories.push(
      buildTrajectorySpan({
        id: `syn-${sessionId}-${i}`,
        sessionId,
        captureTier: "full",
        actions,
        outcome: { status: "success", summary: `synthetic ${template.name}` },
      }),
    );
  }

  return trajectories;
}
