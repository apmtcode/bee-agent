// Deterministic synthetic movement-stream generator.
//
// The engine runs in the cloud with no access to a real machine, so we cannot
// record real mouse/keyboard/window events. This generator produces reproducible
// synthetic movement trajectories from action "templates" with seedable jitter,
// so the whole capture → dataset → train → infer → replay pipeline can be
// validated end-to-end without any OS input. No `Math.random` — a small LCG
// keeps runs identical across machines and CI.

import type { DevicePlatform } from "../capture/device-adapter.js";
import type { MovementContext, MovementDataset, MovementStep, MovementTrajectory } from "./movement-model.js";

/** A repeatable movement pattern: an ordered list of (action, target) steps. */
export type MovementTemplate = {
  name: string;
  steps: Array<{ action: string; target?: string }>;
};

export type SyntheticContextSpec = {
  platform?: DevicePlatform;
  appId: string;
  screen?: string;
  /** Which templates this context demonstrates, by name. */
  templates: string[];
  /** How many demonstrations of each template to emit. Default 1. */
  repeats?: number;
};

export type SyntheticDatasetSpec = {
  seed?: number;
  templates: MovementTemplate[];
  contexts: SyntheticContextSpec[];
  /** Base timestamp for the first step. Default 1_700_000_000_000. */
  baseTs?: number;
};

/** Minimal deterministic PRNG (mulberry32) — no global state, seedable. */
function makeRng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Generate a deterministic movement dataset from templates and context specs.
 * Timestamps advance monotonically with small seeded jitter so ordering is
 * realistic but reproducible.
 */
export function generateSyntheticDataset(spec: SyntheticDatasetSpec): MovementDataset {
  const rng = makeRng(spec.seed ?? 1);
  const byName = new Map(spec.templates.map((template) => [template.name, template]));
  const trajectories: MovementTrajectory[] = [];
  let ts = spec.baseTs ?? 1_700_000_000_000;
  let counter = 0;

  for (const contextSpec of spec.contexts) {
    const context: MovementContext = {
      platform: contextSpec.platform ?? "macos",
      appId: contextSpec.appId,
      ...(contextSpec.screen ? { screen: contextSpec.screen } : {}),
    };
    const repeats = Math.max(1, contextSpec.repeats ?? 1);
    for (const templateName of contextSpec.templates) {
      const template = byName.get(templateName);
      if (!template) {
        throw new Error(`unknown movement template: ${templateName}`);
      }
      for (let r = 0; r < repeats; r += 1) {
        const steps: MovementStep[] = template.steps.map((step) => {
          ts += 40 + Math.floor(rng() * 120);
          return {
            action: step.action,
            ...(step.target ? { target: step.target } : {}),
            ts,
          };
        });
        counter += 1;
        trajectories.push({
          id: `syn-${contextSpec.appId}-${templateName}-${counter}`,
          context,
          steps,
        });
      }
    }
  }

  return { version: 1, trajectories };
}
