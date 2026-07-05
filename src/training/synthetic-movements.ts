import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Deterministic, seeded generator of synthetic movement trajectories. Standing
 * objective #2 must be validated without access to the user's real machine, so
 * this produces replayable {@link TrajectorySpan}s from a movement recipe with
 * controlled slot variation — enough to exercise capture→dataset→train→eval
 * round-trips and to build held-out "related but unseen" variants for the
 * generalization eval.
 *
 * A recipe describes a repeatable movement (e.g. "open document, click Save").
 * Each step's `observation` carries the varying environment (the `{slot}`
 * placeholder), while the `action` summary stays stable — the same movement across
 * different contexts, which is exactly the generalization signal we want to learn.
 */

export type SyntheticMovementStep = {
  /** Observation summary template; may contain `{slot}` placeholders. */
  observation: string;
  /** Observation source channel (e.g. "screen", "window"). */
  source?: string;
  /** Action tool (e.g. "mouse.click", "keyboard.type"). */
  tool: string;
  /** Action summary template; may contain `{slot}` placeholders. */
  summary: string;
};

export type SyntheticMovementRecipe = {
  name: string;
  steps: SyntheticMovementStep[];
  /** Named slot vocabularies substituted into `{slot}` placeholders. */
  slots?: Record<string, string[]>;
};

export type GenerateSyntheticParams = {
  count: number;
  seed: number;
  /** Prefix for generated trajectory ids; defaults to the recipe name. */
  idPrefix?: string;
  sessionId?: string;
  /** Base epoch ms for the first event; each variant advances deterministically. */
  baseTs?: number;
  /** Fixed createdAt so generation stays fully deterministic. */
  createdAt?: string;
  /** Restrict slot choices to a subset (used to carve out held-out vocab). */
  slotFilter?: (slot: string, values: string[]) => string[];
};

/** Generate `count` deterministic trajectory variants from a recipe. */
export function generateSyntheticTrajectories(
  recipe: SyntheticMovementRecipe,
  params: GenerateSyntheticParams,
): TrajectorySpan[] {
  const rng = createRng(params.seed);
  const idPrefix = params.idPrefix ?? recipe.name;
  const baseTs = params.baseTs ?? 1_700_000_000_000;
  const createdAt = params.createdAt ?? "2026-01-01T00:00:00.000Z";
  const slots = resolveSlots(recipe.slots ?? {}, params.slotFilter);

  const trajectories: TrajectorySpan[] = [];
  for (let variant = 0; variant < params.count; variant += 1) {
    const bindings = pickSlotBindings(slots, rng);
    const variantBaseTs = baseTs + variant * 100_000;
    const observations: TrajectorySpan["observations"] = [];
    const actions: TrajectorySpan["actions"] = [];
    recipe.steps.forEach((step, index) => {
      const stepTs = variantBaseTs + index * 1_000;
      observations.push({
        kind: "observation",
        source: step.source ?? "screen",
        summary: applyBindings(step.observation, bindings),
        ts: stepTs,
      });
      actions.push({
        kind: "action",
        tool: step.tool,
        summary: applyBindings(step.summary, bindings),
        ts: stepTs + 1,
      });
    });
    trajectories.push({
      id: `${idPrefix}-${variant}`,
      sessionId: params.sessionId ?? `${idPrefix}-session`,
      createdAt,
      captureTier: "full",
      observations,
      actions,
      outcome: { status: "success", summary: `${recipe.name} variant ${variant}` },
    });
  }
  return trajectories;
}

function resolveSlots(
  slots: Record<string, string[]>,
  filter: GenerateSyntheticParams["slotFilter"],
): Record<string, string[]> {
  if (!filter) {
    return slots;
  }
  const resolved: Record<string, string[]> = {};
  for (const [slot, values] of Object.entries(slots)) {
    const kept = filter(slot, values);
    resolved[slot] = kept.length > 0 ? kept : values;
  }
  return resolved;
}

function pickSlotBindings(slots: Record<string, string[]>, rng: () => number): Record<string, string> {
  const bindings: Record<string, string> = {};
  for (const [slot, values] of Object.entries(slots)) {
    if (values.length === 0) {
      continue;
    }
    const index = Math.floor(rng() * values.length) % values.length;
    bindings[slot] = values[index] ?? values[0]!;
  }
  return bindings;
}

function applyBindings(template: string, bindings: Record<string, string>): string {
  return template.replace(/\{(\w+)\}/g, (match, slot: string) => bindings[slot] ?? match);
}

/**
 * Small deterministic PRNG (mulberry32). Seeded so generated datasets are stable
 * across runs — no `Math.random`, so tests are reproducible.
 */
export function createRng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
