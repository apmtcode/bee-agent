import type { MovementEvent, MovementDataset, MovementTrajectory } from "./movement-model.js";

/**
 * Deterministic synthetic movement-stream generator.
 *
 * The real capture pipeline records mouse/keyboard/window events from the user's
 * machine, which is unavailable in the cloud. This generator fabricates
 * structurally realistic observation -> action streams from a fixed set of
 * workflow templates, seeded so runs are byte-for-byte reproducible. It lets the
 * capture -> dataset -> train -> infer -> generalize loop be validated end to end
 * without any OS input.
 */

/** One step of a workflow template: an observation followed by the action it triggers. */
type WorkflowStep = {
  source: string;
  observation: string;
  tool: string;
  action: string;
};

type WorkflowTemplate = {
  name: string;
  steps: WorkflowStep[];
};

const WORKFLOW_TEMPLATES: WorkflowTemplate[] = [
  {
    name: "deploy",
    steps: [
      { source: "browser", observation: "opened dashboard", tool: "mouse", action: "clicked deploy tab" },
      { source: "browser", observation: "deploy panel visible", tool: "mouse", action: "clicked deploy button" },
      { source: "browser", observation: "confirm dialog shown", tool: "keyboard", action: "typed confirm" },
      { source: "browser", observation: "deploy running", tool: "mouse", action: "clicked view logs" },
    ],
  },
  {
    name: "search",
    steps: [
      { source: "app", observation: "search field focused", tool: "keyboard", action: "typed query" },
      { source: "app", observation: "results listed", tool: "mouse", action: "clicked first result" },
      { source: "app", observation: "detail opened", tool: "mouse", action: "clicked star" },
    ],
  },
  {
    name: "compose",
    steps: [
      { source: "os", observation: "focused editor", tool: "mouse", action: "clicked new document" },
      { source: "os", observation: "editor ready", tool: "keyboard", action: "typed title" },
      { source: "os", observation: "title entered", tool: "keyboard", action: "typed body" },
      { source: "os", observation: "draft ready", tool: "mouse", action: "clicked save" },
    ],
  },
];

export type SyntheticStreamOptions = {
  /** Seed for the PRNG; identical seeds produce identical datasets. */
  seed: number;
  /** Number of trajectories to emit. */
  trajectoryCount: number;
  /** Restrict to a subset of template names; defaults to all. */
  workflows?: string[];
  /** Minimum steps per trajectory (templates are truncated, never padded). */
  minSteps?: number;
  /**
   * When false, every trajectory runs its template to full length (no random
   * truncation) — useful for training exact-reproduction fixtures. Defaults to
   * true, which yields realistic mixed-length (partially abandoned) episodes.
   */
  truncate?: boolean;
  /** Milliseconds between successive events. Defaults to 100. */
  timeStepMs?: number;
};

/** A small, fast, deterministic PRNG (mulberry32). Avoids Math.random for reproducibility. */
function createRng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Build a deterministic dataset of synthetic movement trajectories. */
export function generateSyntheticMovementDataset(options: SyntheticStreamOptions): MovementDataset {
  const rng = createRng(options.seed);
  const timeStep = options.timeStepMs ?? 100;
  const templates = selectTemplates(options.workflows);
  if (templates.length === 0) {
    throw new Error("no matching workflow templates for synthetic stream");
  }
  const minSteps = Math.max(1, options.minSteps ?? 1);

  const trajectories: MovementTrajectory[] = [];
  for (let index = 0; index < Math.max(0, options.trajectoryCount); index += 1) {
    const template = templates[Math.floor(rng() * templates.length)]!;
    const maxSteps = template.steps.length;
    const stepCount =
      options.truncate === false
        ? maxSteps
        : Math.max(Math.min(minSteps, maxSteps), Math.min(maxSteps, 1 + Math.floor(rng() * maxSteps)));
    const events = buildTrajectoryEvents(template, stepCount, index, timeStep);
    trajectories.push({ id: `${template.name}-${index}`, events });
  }
  return { trajectories };
}

/**
 * Produce a single "held-out but related" trajectory: a real template whose
 * first observation is swapped for a novel phrasing the model never trained on.
 * Used by the generalization eval to prove backoff recovers the recorded action
 * chain from an unseen entry point.
 */
export function generateRelatedTrajectory(options: {
  seed: number;
  workflow?: string;
  noveltyPrefix?: string;
  timeStepMs?: number;
}): MovementTrajectory {
  const rng = createRng(options.seed);
  const templates = selectTemplates(options.workflow ? [options.workflow] : undefined);
  const template = templates[Math.floor(rng() * templates.length)]!;
  const timeStep = options.timeStepMs ?? 100;
  const events = buildTrajectoryEvents(template, template.steps.length, 0, timeStep);
  const prefix = options.noveltyPrefix ?? "reached via new path";
  if (events.length > 0 && events[0]!.kind === "observation") {
    const first = events[0]!;
    events[0] = { ...first, summary: `${prefix}: ${first.summary}` };
  }
  return { id: `${template.name}-related`, events };
}

function buildTrajectoryEvents(
  template: WorkflowTemplate,
  stepCount: number,
  trajectoryIndex: number,
  timeStep: number,
): MovementEvent[] {
  const trajectoryId = `${template.name}-${trajectoryIndex}`;
  const events: MovementEvent[] = [];
  let ts = 0;
  for (let step = 0; step < stepCount; step += 1) {
    const definition = template.steps[step]!;
    events.push({
      kind: "observation",
      ts,
      trajectoryId,
      source: definition.source,
      summary: definition.observation,
    });
    ts += timeStep;
    events.push({
      kind: "action",
      ts,
      trajectoryId,
      tool: definition.tool,
      summary: definition.action,
    });
    ts += timeStep;
  }
  return events;
}

function selectTemplates(names: string[] | undefined): WorkflowTemplate[] {
  if (!names || names.length === 0) {
    return WORKFLOW_TEMPLATES;
  }
  const wanted = new Set(names);
  return WORKFLOW_TEMPLATES.filter((template) => wanted.has(template.name));
}

/** Names of the built-in workflow templates. */
export function listSyntheticWorkflows(): string[] {
  return WORKFLOW_TEMPLATES.map((template) => template.name);
}
