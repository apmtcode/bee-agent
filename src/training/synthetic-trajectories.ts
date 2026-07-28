import { buildTrajectorySpan, type TrajectoryAction, type TrajectorySpan } from "../capture/trajectory.js";

/**
 * Deterministic synthetic movement-stream generator.
 *
 * We run in the cloud with no access to a real machine, so the movement
 * subsystem is validated against simulated event streams rather than live
 * mouse/keyboard capture. This produces families of *related* trajectories with
 * a controllable amount of structural novelty, so tests can build train /
 * held-out splits that are related-but-not-identical — the setup the
 * generalization eval needs. It is fully seeded (mulberry32), never touching
 * `Math.random`, so datasets are reproducible.
 */

export type SyntheticFamily = "desktop-file-edit" | "web-form-submit";

export type GenerateSyntheticOptions = {
  family: SyntheticFamily;
  count: number;
  seed: number;
  /**
   * 0..1 probability that each optional step is toggled from its default
   * inclusion, producing related-but-structurally-different sequences. 0 yields
   * near-identical trajectories (good for "repeat"); higher values yield the
   * novel-but-related variants the generalization eval scores. Default 0.
   */
  noveltyRate?: number;
  /** Base timestamp for the first event; each step advances by 1s. Default 0. */
  startTs?: number;
};

type StepTemplate = {
  tool: string;
  optional?: boolean;
  render: (rng: () => number) => string;
};

const FAMILIES: Record<SyntheticFamily, StepTemplate[]> = {
  "desktop-file-edit": [
    { tool: "window.focus", render: () => "focus editor window" },
    { tool: "menu.open", optional: true, render: () => "open file menu" },
    { tool: "mouse.click", render: (rng) => `click at (${coord(rng)}, ${coord(rng)})` },
    { tool: "keyboard.type", render: () => "type value into buffer" },
    { tool: "keyboard.shortcut", optional: true, render: () => "press cmd+f" },
    { tool: "keyboard.type", optional: true, render: () => "type value into buffer" },
    { tool: "keyboard.shortcut", render: () => "press cmd+s" },
  ],
  "web-form-submit": [
    { tool: "browser.navigate", render: (rng) => `open https://app.test/page/${coord(rng)}` },
    { tool: "mouse.click", render: (rng) => `click field at (${coord(rng)}, ${coord(rng)})` },
    { tool: "keyboard.type", render: () => "type value into field" },
    { tool: "mouse.click", optional: true, render: (rng) => `click checkbox at (${coord(rng)}, ${coord(rng)})` },
    { tool: "keyboard.type", optional: true, render: () => "type value into field" },
    { tool: "mouse.click", render: (rng) => `click submit at (${coord(rng)}, ${coord(rng)})` },
  ],
};

export function listSyntheticFamilies(): SyntheticFamily[] {
  return Object.keys(FAMILIES) as SyntheticFamily[];
}

export function generateSyntheticTrajectories(options: GenerateSyntheticOptions): TrajectorySpan[] {
  const template = FAMILIES[options.family];
  if (!template) {
    throw new Error(`Unknown synthetic family: ${options.family}`);
  }
  const noveltyRate = clamp01(options.noveltyRate ?? 0);
  const startTs = options.startTs ?? 0;
  const rng = mulberry32(options.seed >>> 0);

  const trajectories: TrajectorySpan[] = [];
  for (let index = 0; index < options.count; index += 1) {
    let ts = startTs;
    const actions: TrajectoryAction[] = [];
    for (const step of template) {
      const include = step.optional ? applyNovelty(true, noveltyRate, rng) : true;
      if (!include) {
        continue;
      }
      actions.push({
        kind: "action",
        tool: step.tool,
        summary: step.render(rng),
        ts,
      });
      ts += 1000;
    }
    trajectories.push(
      buildTrajectorySpanDeterministic({
        id: `${options.family}-${options.seed}-${index}`,
        sessionId: `synthetic-${options.family}-${index}`,
        actions,
        outcome: { status: "success", summary: `${options.family} completed`, reward: 1 },
      }),
    );
  }
  return trajectories;
}

/**
 * Like {@link buildTrajectorySpan} but never reads the wall clock, so generated
 * datasets stay byte-for-byte reproducible from a seed.
 */
function buildTrajectorySpanDeterministic(params: {
  id: string;
  sessionId: string;
  actions: TrajectoryAction[];
  outcome: TrajectorySpan["outcome"];
}): TrajectorySpan {
  const span = buildTrajectorySpan({
    id: params.id,
    sessionId: params.sessionId,
    actions: params.actions,
    outcome: params.outcome,
  });
  const firstTs = params.actions[0]?.ts ?? 0;
  return { ...span, createdAt: new Date(firstTs).toISOString() };
}

function applyNovelty(defaultInclude: boolean, noveltyRate: number, rng: () => number): boolean {
  return rng() < noveltyRate ? !defaultInclude : defaultInclude;
}

function coord(rng: () => number): number {
  return Math.floor(rng() * 1000);
}

function clamp01(value: number): number {
  if (Number.isNaN(value)) {
    return 0;
  }
  return Math.min(1, Math.max(0, value));
}

/** Small deterministic PRNG (mulberry32). */
function mulberry32(seed: number): () => number {
  let state = seed;
  return () => {
    state |= 0;
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
