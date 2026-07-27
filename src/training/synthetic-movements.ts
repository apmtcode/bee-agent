import type { MovementSequence, MovementStep } from "./movement-model.js";

/**
 * Deterministic synthetic movement generator. No randomness — sequences are a
 * pure function of the seed inputs so tests are reproducible in the cloud/CI,
 * where no real OS input capture is available. Produces small "workflow"
 * trajectories (open -> focus -> act -> submit) plus related variants for
 * exercising the generalization path.
 */

const APPS = ["mail", "browser", "editor", "terminal"] as const;
const TARGETS = ["compose", "send", "reply", "search", "save", "run"] as const;

function focusStep(app: string): MovementStep {
  return { tool: "os", gesture: "focus", target: app, summary: `focused ${app}` };
}

function tapStep(target: string): MovementStep {
  return { tool: "device", gesture: "tap", target, summary: `tapped ${target}` };
}

function typeStep(target: string): MovementStep {
  return { tool: "device", gesture: "type", target, summary: `typed into ${target}` };
}

function scrollStep(direction: "up" | "down"): MovementStep {
  return { tool: "device", gesture: "scroll", direction, summary: `scrolled ${direction}` };
}

/**
 * A canonical workflow for `(app, target)`: focus the app, tap the target,
 * type, then confirm. Deterministic in the pair alone.
 */
export function syntheticWorkflow(app: string, target: string): MovementStep[] {
  return [focusStep(app), tapStep(target), typeStep(target), tapStep("send")];
}

/**
 * Generate `count` deterministic training sequences by walking the cartesian
 * product of apps x targets in a fixed order.
 */
export function generateSyntheticDataset(count: number): MovementSequence[] {
  const sequences: MovementSequence[] = [];
  let index = 0;
  outer: for (const app of APPS) {
    for (const target of TARGETS) {
      if (sequences.length >= count) {
        break outer;
      }
      sequences.push({ trajectoryId: `synthetic-${index}`, steps: syntheticWorkflow(app, target) });
      index += 1;
    }
  }
  return sequences;
}

/**
 * Produce a related-but-novel variant of a workflow: same skeleton, an extra
 * scroll interaction inserted before the confirm. Used to test that the model
 * generalizes (repeats the learned skeleton) rather than only memorizing.
 */
export function relatedVariant(app: string, target: string): MovementSequence {
  return {
    trajectoryId: `variant-${app}-${target}`,
    steps: [focusStep(app), tapStep(target), typeStep(target), scrollStep("down"), tapStep("send")],
  };
}

export const SYNTHETIC_APPS = APPS;
export const SYNTHETIC_TARGETS = TARGETS;
