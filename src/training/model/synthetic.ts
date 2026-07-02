/**
 * Deterministic synthetic movement-stream generator (roadmap: "Synthetic
 * event-stream generator to validate capture->dataset->replay round-trips without
 * real OS input"). Produces {@link MovementSequence}s from named task templates with
 * seeded, reproducible variation so tests can build train/held-out splits that are
 * "related but new" — exactly what objective #2(d) generalization must handle.
 *
 * No RNG: variation is driven by an integer seed through a small deterministic LCG,
 * so a given (template, seed) always yields the same stream.
 */
import type { MovementEvent, MovementSequence } from "./movement-model.js";

export type MovementTemplateStep = {
  kind: "observation" | "action";
  tool?: string;
  source?: string;
  /** Summary with optional `{slot}` placeholders filled from the seed. */
  summary: string;
};

export type MovementTemplate = {
  id: string;
  steps: MovementTemplateStep[];
};

/** A small deterministic LCG (Numerical Recipes constants) — no Math.random. */
function lcg(seed: number): () => number {
  let state = (seed >>> 0) || 1;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

/** Built-in templates modelling common desktop task shapes. */
export const BUILTIN_MOVEMENT_TEMPLATES: MovementTemplate[] = [
  {
    id: "open-edit-save",
    steps: [
      { kind: "observation", source: "window", summary: "focus {app}" },
      { kind: "action", tool: "mouse.click", summary: "click menu:File" },
      { kind: "action", tool: "mouse.click", summary: "click item:Open" },
      { kind: "action", tool: "keyboard.type", summary: "type {file}" },
      { kind: "action", tool: "keyboard.press", summary: "press Enter" },
      { kind: "observation", source: "screen", summary: "loaded {file}" },
      { kind: "action", tool: "keyboard.type", summary: "type {text}" },
      { kind: "action", tool: "keyboard.press", summary: "press Cmd+S" },
      { kind: "observation", source: "screen", summary: "saved {file}" },
    ],
  },
  {
    id: "search-select-copy",
    steps: [
      { kind: "observation", source: "window", summary: "focus {app}" },
      { kind: "action", tool: "keyboard.press", summary: "press Cmd+F" },
      { kind: "action", tool: "keyboard.type", summary: "type {query}" },
      { kind: "action", tool: "keyboard.press", summary: "press Enter" },
      { kind: "action", tool: "mouse.drag", summary: "select match:{query}" },
      { kind: "action", tool: "keyboard.press", summary: "press Cmd+C" },
      { kind: "observation", source: "clipboard", summary: "copied {query}" },
    ],
  },
];

const SLOT_VALUES: Record<string, string[]> = {
  app: ["editor", "browser", "terminal", "notes"],
  file: ["report.md", "notes.txt", "data.json", "todo.md"],
  text: ["hello", "summary", "draft", "changelog"],
  query: ["error", "TODO", "config", "result"],
};

function fillSlots(summary: string, next: () => number): string {
  return summary.replace(/\{(\w+)\}/g, (match, slot: string) => {
    const options = SLOT_VALUES[slot];
    if (!options || options.length === 0) {
      return match;
    }
    return options[Math.floor(next() * options.length) % options.length];
  });
}

export type SyntheticSequenceOptions = {
  /** Base millisecond timestamp for the first event. */
  startTs?: number;
  /** Mean inter-event delay in ms (jittered deterministically). */
  stepMs?: number;
};

/** Generate one deterministic movement sequence from a template + seed. */
export function generateSyntheticSequence(
  template: MovementTemplate,
  seed: number,
  options?: SyntheticSequenceOptions,
): MovementSequence {
  const next = lcg(seed);
  const startTs = options?.startTs ?? 1_000_000;
  const stepMs = options?.stepMs ?? 200;
  let ts = startTs;

  const events: MovementEvent[] = template.steps.map((step) => {
    ts += Math.max(1, Math.floor(stepMs * (0.5 + next())));
    const summary = fillSlots(step.summary, next);
    if (step.kind === "action") {
      return { kind: "action", tool: step.tool ?? "unknown", summary, ts };
    }
    return { kind: "observation", source: step.source ?? "unknown", summary, ts };
  });

  return { trajectoryId: `${template.id}-${seed}`, events, outcome: "success" };
}

/** Generate `count` deterministic sequences per template, seeds offset by `seedBase`. */
export function generateSyntheticDataset(params?: {
  templates?: MovementTemplate[];
  countPerTemplate?: number;
  seedBase?: number;
  options?: SyntheticSequenceOptions;
}): MovementSequence[] {
  const templates = params?.templates ?? BUILTIN_MOVEMENT_TEMPLATES;
  const count = params?.countPerTemplate ?? 4;
  const seedBase = params?.seedBase ?? 1;
  const sequences: MovementSequence[] = [];
  for (const template of templates) {
    for (let i = 0; i < count; i += 1) {
      sequences.push(generateSyntheticSequence(template, seedBase + i, params?.options));
    }
  }
  return sequences;
}
