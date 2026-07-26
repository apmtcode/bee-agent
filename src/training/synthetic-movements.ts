import { buildTrajectorySpan, type TrajectoryAction, type TrajectorySpan } from "../capture/trajectory.js";

/**
 * Synthetic movement-stream generator.
 *
 * bee-agent runs in the cloud with no access to a real machine, so the
 * movement-learning pipeline must be validated against simulated input. This
 * module produces deterministic, seedable trajectories from small "movement
 * grammars" — parametric templates of ordered steps with target pools — so
 * capture -> dataset -> train -> replay round-trips can be exercised without
 * any OS-level capture. Using a disjoint target pool yields a held-out set that
 * is structurally identical but references new targets, which is exactly the
 * generalization case the model must handle.
 */

export type MovementGrammarStep = {
  tool: string;
  /** gesture kind / verb; stored in action metadata so tokenization is faithful. */
  action: string;
  targets: string[];
};

export type MovementGrammar = {
  name: string;
  steps: MovementGrammarStep[];
};

export type GenerateSyntheticParams = {
  grammar: MovementGrammar;
  count: number;
  rng: () => number;
  sessionPrefix?: string;
  startTs?: number;
  /** probability of dropping an optional trailing step, adding realistic variation. */
  dropTailProbability?: number;
};

/** Deterministic PRNG (mulberry32) — no Math.random, fully reproducible from a seed. */
export function createSeededRng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function pick<T>(items: T[], rng: () => number): T {
  if (items.length === 0) {
    throw new Error("cannot pick from an empty pool");
  }
  const index = Math.min(items.length - 1, Math.floor(rng() * items.length));
  return items[index];
}

/** Generate a batch of synthetic movement trajectories from a grammar. */
export function generateSyntheticTrajectories(params: GenerateSyntheticParams): TrajectorySpan[] {
  const prefix = params.sessionPrefix ?? params.grammar.name;
  const dropTail = params.dropTailProbability ?? 0;
  let ts = params.startTs ?? 1_000;
  const trajectories: TrajectorySpan[] = [];

  for (let i = 0; i < params.count; i += 1) {
    const sessionId = `${prefix}-session-${i}`;
    const steps = [...params.grammar.steps];
    if (steps.length > 1 && params.rng() < dropTail) {
      steps.pop();
    }
    const actions: TrajectoryAction[] = steps.map((step) => {
      const target = pick(step.targets, params.rng);
      ts += 1;
      return {
        kind: "action",
        tool: step.tool,
        summary: `${step.action} ${target}`.trim(),
        ts,
        metadata: { gesture: step.action, target },
      };
    });

    trajectories.push(
      buildTrajectorySpanFixed({
        id: `${prefix}-traj-${i}`,
        sessionId,
        actions,
      }),
    );
  }

  return trajectories;
}

/**
 * `buildTrajectorySpan` stamps `createdAt` with `new Date()`, which is fine for
 * production but non-deterministic. We reuse its shape but override the
 * timestamp so synthetic datasets are byte-stable across runs.
 */
function buildTrajectorySpanFixed(params: {
  id: string;
  sessionId: string;
  actions: TrajectoryAction[];
}): TrajectorySpan {
  const span = buildTrajectorySpan({
    id: params.id,
    sessionId: params.sessionId,
    actions: params.actions,
  });
  return { ...span, createdAt: "1970-01-01T00:00:00.000Z" };
}

/** A canonical form-fill movement: focus a field, type, tab, type, submit. */
export const FORM_FILL_GRAMMAR: MovementGrammar = {
  name: "form-fill",
  steps: [
    { tool: "device", action: "tap", targets: ["name-field", "email-field", "search-field"] },
    { tool: "device", action: "type", targets: ["given-name", "family-name", "query-text"] },
    { tool: "device", action: "shortcut", targets: ["tab", "next-field"] },
    { tool: "device", action: "type", targets: ["value-a", "value-b", "value-c"] },
    { tool: "device", action: "tap", targets: ["submit-button", "save-button"] },
  ],
};

/** A menu-navigation movement: open menu, scroll, select, confirm. */
export const NAV_MENU_GRAMMAR: MovementGrammar = {
  name: "nav-menu",
  steps: [
    { tool: "device", action: "tap", targets: ["menu-button", "hamburger"] },
    { tool: "device", action: "scroll", targets: ["down", "up"] },
    { tool: "device", action: "tap", targets: ["settings-item", "profile-item", "help-item"] },
    { tool: "device", action: "tap", targets: ["confirm-button", "ok-button"] },
  ],
};

/**
 * Build a structurally identical grammar whose targets are disjoint from the
 * originals — the held-out set for generalization evaluation.
 */
export function withNovelTargets(grammar: MovementGrammar, suffix = "-v2"): MovementGrammar {
  return {
    name: `${grammar.name}${suffix}`,
    steps: grammar.steps.map((step) => ({
      tool: step.tool,
      action: step.action,
      targets: step.targets.map((target) => `${target}${suffix}`),
    })),
  };
}
