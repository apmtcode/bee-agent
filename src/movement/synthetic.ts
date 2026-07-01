import { buildTrajectorySpan, type TrajectoryAction, type TrajectorySpan } from "../capture/trajectory.js";
import { tokenizeAction, type MovementSequence, type MovementToken } from "./movement-model.js";

/**
 * Synthetic movement event-stream generator (standing objective #2).
 *
 * bee-agent runs in the cloud with no access to a real machine, so we cannot
 * capture real mouse/keyboard/window events. This module deterministically
 * fabricates *plausible, structured* movement trajectories from small task
 * grammars, so the capture -> dataset -> replay -> train -> infer pipeline can be
 * exercised and evaluated end-to-end without any OS input. Deterministic output
 * (seeded PRNG) keeps tests reproducible.
 */

/** A single synthetic movement step (maps 1:1 to a recorded trajectory action). */
export type SyntheticStep = {
  tool: string;
  summary: string;
};

/** A named task grammar: a canonical flow plus optional variation points. */
export type SyntheticTaskTemplate = {
  app: string;
  /** Steps performed on every run of this task. */
  steps: SyntheticStep[];
  /**
   * Optional steps that may be inserted after the canonical flow with the given
   * probability — the source of "new but related" variation the model must
   * generalize over.
   */
  optionalTrailingSteps?: Array<{ step: SyntheticStep; probability: number }>;
};

export type SyntheticDatasetOptions = {
  seed: number;
  /** How many trajectories to generate. */
  count: number;
  templates?: SyntheticTaskTemplate[];
  /** Per-step chance of dropping a canonical step (models noisy capture). */
  dropProbability?: number;
};

export type SyntheticDataset = {
  spans: TrajectorySpan[];
  sequences: MovementSequence[];
};

/** A deterministic mulberry32 PRNG — reproducible across runs and platforms. */
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

/** A small default set of task grammars covering common desktop flows. */
export function defaultSyntheticTemplates(): SyntheticTaskTemplate[] {
  return [
    {
      app: "editor",
      steps: [
        { tool: "os", summary: "focused editor" },
        { tool: "device", summary: "tapped file tree" },
        { tool: "device", summary: "typed into buffer" },
        { tool: "device", summary: "triggered save" },
      ],
      optionalTrailingSteps: [{ step: { tool: "device", summary: "triggered format" }, probability: 0.5 }],
    },
    {
      app: "browser",
      steps: [
        { tool: "os", summary: "focused browser" },
        { tool: "device", summary: "tapped address bar" },
        { tool: "device", summary: "typed url" },
        { tool: "device", summary: "tapped go" },
      ],
      optionalTrailingSteps: [{ step: { tool: "device", summary: "scrolled down" }, probability: 0.5 }],
    },
    {
      app: "terminal",
      steps: [
        { tool: "os", summary: "focused terminal" },
        { tool: "device", summary: "typed command" },
        { tool: "device", summary: "triggered enter" },
      ],
    },
  ];
}

/**
 * Generate a deterministic synthetic dataset of movement trajectories, returning
 * both recorded {@link TrajectorySpan}s and their tokenized {@link MovementSequence}s.
 */
export function generateSyntheticMovementDataset(options: SyntheticDatasetOptions): SyntheticDataset {
  const templates = options.templates ?? defaultSyntheticTemplates();
  if (templates.length === 0) {
    throw new Error("generateSyntheticMovementDataset requires at least one template");
  }
  const dropProbability = clampProbability(options.dropProbability ?? 0);
  const rng = createSeededRng(options.seed);

  const spans: TrajectorySpan[] = [];
  const sequences: MovementSequence[] = [];

  for (let i = 0; i < options.count; i += 1) {
    const template = templates[Math.floor(rng() * templates.length) % templates.length]!;
    const chosen: SyntheticStep[] = [];
    for (const step of template.steps) {
      if (rng() < dropProbability) {
        continue;
      }
      chosen.push(step);
    }
    for (const optional of template.optionalTrailingSteps ?? []) {
      if (rng() < clampProbability(optional.probability)) {
        chosen.push(optional.step);
      }
    }
    // Guarantee at least one action so downstream stages have signal.
    if (chosen.length === 0) {
      chosen.push(template.steps[0]!);
    }

    const baseTs = 1_000 + i * 1_000;
    const actions: TrajectoryAction[] = chosen.map((step, index) => ({
      kind: "action",
      tool: step.tool,
      summary: step.summary,
      ts: baseTs + index,
      metadata: { app: template.app },
    }));

    const span = buildTrajectorySpan({
      id: `synthetic-${options.seed}-${i}`,
      sessionId: `synthetic-session-${options.seed}`,
      captureTier: "app",
      actions,
    });
    // buildTrajectorySpan stamps a wall-clock createdAt; override it with a
    // value derived from the synthetic timeline so the dataset is fully
    // reproducible (the whole point of a seeded generator).
    span.createdAt = new Date(baseTs).toISOString();
    spans.push(span);
    sequences.push(chosen.map((step) => tokenizeAction(step.tool, step.summary)));
  }

  return { spans, sequences };
}

/** Convenience: the distinct movement tokens present in a dataset. */
export function datasetVocabulary(dataset: SyntheticDataset): MovementToken[] {
  const vocab = new Set<MovementToken>();
  for (const sequence of dataset.sequences) {
    for (const token of sequence) {
      vocab.add(token);
    }
  }
  return [...vocab].sort();
}

function clampProbability(value: number): number {
  if (Number.isNaN(value)) {
    return 0;
  }
  return Math.min(1, Math.max(0, value));
}
