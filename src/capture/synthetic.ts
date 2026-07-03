/**
 * Synthetic movement event-stream generator.
 *
 * bee-agent runs in the cloud with no access to a real machine, so the
 * local-movement learning subsystem is validated against deterministic
 * synthetic streams instead of live OS input. This module emits real
 * {@link TrajectorySpan} values (device-style gesture actions) from workflow
 * templates, so downstream code — the tokenizer, dataset exporter, replay
 * manifest, and movement model — exercises the exact production path.
 *
 * Generation is fully deterministic: a seeded linear-congruential PRNG drives
 * both the workflow selection and the small perturbations (dropped / repeated
 * steps) that produce *related-but-new* variants, which is what the model's
 * generalization is measured against.
 */

import { buildTrajectorySpan, type TrajectoryAction, type TrajectorySpan } from "./trajectory.js";

export type SyntheticStep = {
  gesture: "tap" | "swipe" | "scroll" | "type" | "shortcut";
  target?: string;
  direction?: "up" | "down" | "left" | "right";
};

export type SyntheticWorkflow = {
  name: string;
  steps: SyntheticStep[];
};

/** A small library of representative UI workflows used by tests and evals. */
export const DEFAULT_SYNTHETIC_WORKFLOWS: SyntheticWorkflow[] = [
  {
    name: "compose-message",
    steps: [
      { gesture: "tap", target: "inbox" },
      { gesture: "tap", target: "compose" },
      { gesture: "type", target: "recipient" },
      { gesture: "type", target: "body" },
      { gesture: "tap", target: "send" },
    ],
  },
  {
    name: "browse-feed",
    steps: [
      { gesture: "tap", target: "feed" },
      { gesture: "scroll", direction: "down" },
      { gesture: "scroll", direction: "down" },
      { gesture: "tap", target: "article" },
      { gesture: "swipe", direction: "left" },
    ],
  },
  {
    name: "settings-toggle",
    steps: [
      { gesture: "tap", target: "settings" },
      { gesture: "scroll", direction: "down" },
      { gesture: "tap", target: "notifications" },
      { gesture: "shortcut", target: "save" },
    ],
  },
];

export type SyntheticGeneratorOptions = {
  seed?: number;
  count: number;
  workflows?: SyntheticWorkflow[];
  /** Per-step probability of a benign perturbation (drop/repeat). Defaults to 0. */
  variationRate?: number;
  /** Milliseconds between consecutive actions in a span. Defaults to 250. */
  stepIntervalMs?: number;
  /** Base timestamp for the first action of the first span. Defaults to 0. */
  startTs?: number;
};

/** Deterministic 32-bit LCG so identical options always yield identical streams. */
function createRng(seed: number): () => number {
  let state = (seed >>> 0) || 0x9e3779b9;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

function stepToAction(step: SyntheticStep, ts: number): TrajectoryAction {
  const summaryTarget = step.target ?? step.direction ?? "device";
  return {
    kind: "action",
    tool: "device",
    summary: `${step.gesture} ${summaryTarget}`,
    ts,
    metadata: {
      gesture: step.gesture,
      ...(step.target ? { target: step.target } : {}),
      ...(step.direction ? { direction: step.direction } : {}),
    },
  };
}

/**
 * Generate `count` deterministic synthetic trajectory spans. When
 * `variationRate > 0`, each span may drop or repeat individual steps, producing
 * related-but-novel sequences suitable for held-out generalization evaluation.
 */
export function generateSyntheticTrajectories(options: SyntheticGeneratorOptions): TrajectorySpan[] {
  const workflows = options.workflows ?? DEFAULT_SYNTHETIC_WORKFLOWS;
  if (workflows.length === 0) {
    throw new Error("generateSyntheticTrajectories requires at least one workflow");
  }
  const rng = createRng(options.seed ?? 1);
  const variationRate = Math.min(Math.max(options.variationRate ?? 0, 0), 1);
  const stepIntervalMs = options.stepIntervalMs ?? 250;
  let ts = options.startTs ?? 0;
  const spans: TrajectorySpan[] = [];

  for (let index = 0; index < options.count; index += 1) {
    const workflow = workflows[Math.floor(rng() * workflows.length) % workflows.length]!;
    const actions: TrajectoryAction[] = [];
    for (const step of workflow.steps) {
      if (variationRate > 0 && rng() < variationRate) {
        // Drop this step entirely to create a related-but-shorter variant.
        continue;
      }
      actions.push(stepToAction(step, ts));
      ts += stepIntervalMs;
      if (variationRate > 0 && rng() < variationRate) {
        // Occasionally repeat a step (double-tap style noise).
        actions.push(stepToAction(step, ts));
        ts += stepIntervalMs;
      }
    }
    // Guarantee non-empty spans even under aggressive variation.
    if (actions.length === 0) {
      actions.push(stepToAction(workflow.steps[0]!, ts));
      ts += stepIntervalMs;
    }
    spans.push(
      buildTrajectorySpan({
        id: `synthetic-${workflow.name}-${index}`,
        sessionId: `synthetic-session-${index}`,
        captureTier: "app",
        actions,
        outcome: { status: "success", summary: `completed ${workflow.name}` },
      }),
    );
  }

  return spans;
}
