// Deterministic synthetic movement-stream generator (roadmap: "Synthetic
// event-stream generator to validate capture->dataset->replay round-trips
// without real OS input").
//
// bee-agent runs in the cloud with no access to a real machine, so the
// movement-learning subsystem is validated against *simulated* input. This
// generator produces reproducible, structured event sequences that mimic a user
// repeating a task (e.g. "open editor, focus file, type, save") with small,
// controlled variations — enough to exercise both exact-replay and
// generalization paths of a movement model. No randomness: variation is driven
// by the caller-supplied index so runs are byte-for-byte reproducible.

import type { MovementEvent } from "./movement-model.js";

/** A named step in a synthetic task: an observation followed by an action. */
export type SyntheticStep = {
  observationSource: string;
  observationSummary: string;
  actionTool: string;
  actionSummary: string;
};

/** A synthetic task template: an ordered list of steps to repeat. */
export type SyntheticTaskTemplate = {
  name: string;
  steps: SyntheticStep[];
};

export type SyntheticSequenceOptions = {
  /** Milliseconds between consecutive events (default 1000). */
  stepMillis?: number;
  /** Base timestamp for the first event (default 0). */
  startTs?: number;
};

/**
 * Materialize one synthetic movement sequence from a template. Fully
 * deterministic given the template and options.
 */
export function synthesizeSequence(
  template: SyntheticTaskTemplate,
  options?: SyntheticSequenceOptions,
): { trajectoryId: string; events: MovementEvent[] } {
  const stepMillis = options?.stepMillis ?? 1000;
  const startTs = options?.startTs ?? 0;
  const events: MovementEvent[] = [];
  let ts = startTs;
  for (const step of template.steps) {
    events.push({ kind: "observation", source: step.observationSource, summary: step.observationSummary, ts });
    ts += stepMillis;
    events.push({ kind: "action", tool: step.actionTool, summary: step.actionSummary, ts });
    ts += stepMillis;
  }
  return { trajectoryId: `synthetic:${template.name}`, events };
}

/**
 * Produce `count` variant sequences of a template. Variation is a deterministic
 * function of the variant index — a monotonically shifted start timestamp — so
 * the *movement structure* is identical (exact-replay training data) while
 * timestamps differ (mimicking repeated real captures).
 */
export function synthesizeRepetitions(
  template: SyntheticTaskTemplate,
  count: number,
  options?: SyntheticSequenceOptions,
): Array<{ trajectoryId: string; events: MovementEvent[] }> {
  const sequences: Array<{ trajectoryId: string; events: MovementEvent[] }> = [];
  const stepMillis = options?.stepMillis ?? 1000;
  for (let index = 0; index < count; index += 1) {
    const base = synthesizeSequence(template, {
      ...options,
      startTs: (options?.startTs ?? 0) + index * stepMillis * template.steps.length * 4,
    });
    sequences.push({ trajectoryId: `${base.trajectoryId}#${index}`, events: base.events });
  }
  return sequences;
}

/**
 * Derive a "related but new" variant of a template by rewording each step's
 * observation while keeping the same action targets. This exercises a model's
 * *generalization* path: the exact context tokens differ, but the underlying
 * task and the correct next actions are unchanged.
 */
export function relatedVariant(template: SyntheticTaskTemplate, suffix = "again"): SyntheticTaskTemplate {
  return {
    name: `${template.name}-${suffix}`,
    steps: template.steps.map((step) => ({
      ...step,
      observationSummary: `${step.observationSummary} (${suffix})`,
    })),
  };
}
