// Synthetic movement-stream generator. The real capture pipeline needs OS input
// we do not have in the cloud, so this produces deterministic, structured
// movement sequences to exercise the dataset -> train -> eval round-trip. It is
// intentionally *rule-based* (a tiny grammar of app workflows) so that a
// sequence model can actually learn the regularities and generalize across
// sequences that share a workflow but differ in surface detail.

import type { MovementDataset, MovementSequence, MovementStep } from "./movement-model.js";

export type SyntheticWorkflow = {
  /** App/screen context the workflow runs in. */
  context: string;
  /** Ordered gesture template; each entry becomes one movement step. */
  template: Array<{ gesture: string; target?: string; direction?: string }>;
};

/** A small library of realistic, learnable UI workflows. */
export const DEFAULT_SYNTHETIC_WORKFLOWS: SyntheticWorkflow[] = [
  {
    context: "mail",
    template: [
      { gesture: "tap", target: "compose" },
      { gesture: "type", target: "recipient" },
      { gesture: "type", target: "subject" },
      { gesture: "type", target: "body" },
      { gesture: "tap", target: "send" },
    ],
  },
  {
    context: "browser",
    template: [
      { gesture: "tap", target: "address-bar" },
      { gesture: "type", target: "url" },
      { gesture: "shortcut", target: "enter" },
      { gesture: "scroll", direction: "down" },
      { gesture: "tap", target: "link" },
    ],
  },
  {
    context: "files",
    template: [
      { gesture: "tap", target: "folder" },
      { gesture: "scroll", direction: "down" },
      { gesture: "tap", target: "file" },
      { gesture: "shortcut", target: "open" },
    ],
  },
];

/**
 * Generate a deterministic movement dataset. `repeats` copies of each workflow
 * are emitted (timestamps offset per copy) so a model has repeated evidence to
 * learn from, and held-out copies to generalize to. No RNG/clock — the caller
 * supplies a base timestamp so results are reproducible.
 */
export function generateSyntheticMovementDataset(params?: {
  workflows?: SyntheticWorkflow[];
  repeats?: number;
  baseTs?: number;
}): MovementDataset {
  const workflows = params?.workflows ?? DEFAULT_SYNTHETIC_WORKFLOWS;
  const repeats = Math.max(1, params?.repeats ?? 3);
  const baseTs = params?.baseTs ?? 0;
  const sequences: MovementSequence[] = [];

  workflows.forEach((workflow, workflowIndex) => {
    for (let copy = 0; copy < repeats; copy += 1) {
      const startTs = baseTs + (workflowIndex * repeats + copy) * 10_000;
      const steps: MovementStep[] = workflow.template.map((entry, stepIndex) => ({
        ts: startTs + stepIndex * 100,
        gesture: entry.gesture,
        ...(entry.target ? { target: entry.target } : {}),
        ...(entry.direction ? { direction: entry.direction } : {}),
        summary: `${workflow.context}:${entry.gesture}:${entry.target ?? entry.direction ?? ""}`,
      }));
      sequences.push({
        id: `${workflow.context}-${copy}`,
        context: workflow.context,
        steps,
      });
    }
  });

  return { version: 1, sequences };
}
