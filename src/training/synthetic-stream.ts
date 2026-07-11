import type { MovementDataset, MovementSequence, MovementStep } from "./movement-model.js";

/**
 * Deterministic synthetic movement-stream generator.
 *
 * bee-agent runs in the cloud with no access to a real machine, so the
 * local-movement learning pipeline is validated against *simulated* event
 * streams produced here. Generation is seeded (mulberry32) and therefore fully
 * reproducible — the same seed always yields the same dataset, which keeps
 * capture→dataset→train→replay round-trip tests stable.
 */

/** One step of a task grammar: a fixed movement kind with a slot of targets. */
export type MovementStepTemplate = {
  tool: string;
  gesture?: string;
  /** Candidate targets; one is chosen per emitted step to vary the instance. */
  targets: string[];
  /** When set, the step is emitted only with this probability (structural variation). */
  optional?: number;
  /** When set, the step may repeat up to this many extra times (structural variation). */
  maxRepeat?: number;
};

export type MovementTaskTemplate = {
  name: string;
  steps: MovementStepTemplate[];
};

/** A small library of representative UI task grammars for simulation. */
export const DEFAULT_MOVEMENT_TASKS: MovementTaskTemplate[] = [
  {
    name: "open-and-deploy",
    steps: [
      { tool: "window", gesture: "focus", targets: ["deploy-console", "ci-dashboard"] },
      { tool: "pointer", gesture: "click", targets: ["environments-tab", "pipelines-tab"] },
      { tool: "pointer", gesture: "click", targets: ["deploy-button", "release-button"] },
      { tool: "keyboard", gesture: "type", targets: ["confirm", "yes"], optional: 0.5 },
      { tool: "pointer", gesture: "click", targets: ["confirm-dialog", "approve-dialog"] },
    ],
  },
  {
    name: "search-and-select",
    steps: [
      { tool: "keyboard", gesture: "shortcut", targets: ["cmd+k", "cmd+p"] },
      { tool: "keyboard", gesture: "type", targets: ["invoice", "report", "customer"] },
      { tool: "pointer", gesture: "scroll", targets: ["results-list"], optional: 0.6, maxRepeat: 2 },
      { tool: "pointer", gesture: "click", targets: ["first-result", "second-result"] },
    ],
  },
  {
    name: "fill-form",
    steps: [
      { tool: "pointer", gesture: "click", targets: ["name-field"] },
      { tool: "keyboard", gesture: "type", targets: ["value"] },
      { tool: "keyboard", gesture: "tab", targets: ["next-field"], maxRepeat: 3 },
      { tool: "pointer", gesture: "click", targets: ["submit"] },
    ],
  },
];

export type GenerateSyntheticDatasetOptions = {
  seed: number;
  tasks?: MovementTaskTemplate[];
  /** Sequences to synthesize per task. Defaults to 8. */
  sequencesPerTask?: number;
};

/** mulberry32 — a tiny deterministic PRNG seeded from an integer. */
function makeRng(seed: number): () => number {
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
  return items[Math.floor(rng() * items.length) % items.length]!;
}

function renderSummary(step: MovementStepTemplate, target: string): string {
  const verb = step.gesture ?? "act";
  return `${verb} ${target}`.trim();
}

function emitStep(template: MovementStepTemplate, rng: () => number): MovementStep {
  const target = pick(template.targets, rng);
  return {
    tool: template.tool,
    ...(template.gesture ? { gesture: template.gesture } : {}),
    target,
    summary: renderSummary(template, target),
  };
}

/**
 * Generate a reproducible {@link MovementDataset}. Target choices, optional
 * steps, and repeats all vary with the RNG, so different seeds produce
 * *related but structurally distinct* sequences — ideal for a held-out
 * generalization split.
 */
export function generateSyntheticMovementDataset(options: GenerateSyntheticDatasetOptions): MovementDataset {
  const tasks = options.tasks ?? DEFAULT_MOVEMENT_TASKS;
  const perTask = Math.max(1, Math.floor(options.sequencesPerTask ?? 8));
  const rng = makeRng(options.seed);
  const sequences: MovementSequence[] = [];

  for (const task of tasks) {
    for (let i = 0; i < perTask; i += 1) {
      const steps: MovementStep[] = [];
      for (const template of task.steps) {
        if (template.optional !== undefined && rng() > template.optional) {
          continue;
        }
        steps.push(emitStep(template, rng));
        if (template.maxRepeat) {
          const repeats = Math.floor(rng() * (template.maxRepeat + 1));
          for (let r = 0; r < repeats; r += 1) {
            steps.push(emitStep(template, rng));
          }
        }
      }
      sequences.push({ id: `${task.name}-${i}`, label: task.name, steps });
    }
  }

  return { sequences };
}

/**
 * Split a dataset into train/held-out partitions by index stride, so both
 * partitions cover every task grammar (needed for a fair generalization eval).
 */
export function splitMovementDataset(
  dataset: MovementDataset,
  holdOutEvery = 4,
): { train: MovementDataset; heldOut: MovementDataset } {
  const train: MovementSequence[] = [];
  const heldOut: MovementSequence[] = [];
  dataset.sequences.forEach((sequence, index) => {
    if (holdOutEvery > 0 && index % holdOutEvery === holdOutEvery - 1) {
      heldOut.push(sequence);
    } else {
      train.push(sequence);
    }
  });
  return { train: { sequences: train }, heldOut: { sequences: heldOut } };
}
