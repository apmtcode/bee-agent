/**
 * Synthetic movement-stream generator + real-data bridges (standing objective #2).
 *
 * bee-agent runs in the cloud with no access to a real mouse/keyboard, so the
 * movement-model pipeline is validated with *simulated* streams. This module
 * produces deterministic, seedable {@link MovementSequence}s with controlled
 * variation — enough structure that a model can learn to repeat a workflow, and
 * enough held-out variation that generalization is measurable — plus adapters
 * that turn real recorded {@link ReplayManifest replay} events and device
 * gestures into the same schema, so the same model consumes synthetic and real
 * data through one path.
 *
 * Determinism: uses a small in-module LCG seeded by an integer, never
 * `Math.random`, so a given seed always yields the same dataset (reproducible
 * training + eval).
 */

import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import type { DeviceGestureKind } from "../capture/device-adapter.js";
import type { MovementSequence, MovementStep } from "./movement-model.js";

/** Deterministic 32-bit LCG (glibc constants) — no Math.random, fully reproducible. */
class Lcg {
  private state: number;
  constructor(seed: number) {
    // Avoid a zero state; keep it in the 32-bit unsigned range.
    this.state = (Math.floor(seed) >>> 0) || 0x2545f491;
  }
  next(): number {
    this.state = (Math.imul(this.state, 1103515245) + 12345) >>> 0;
    return this.state;
  }
  /** Uniform float in [0, 1). */
  float(): number {
    return this.next() / 0x100000000;
  }
  /** Integer in [0, bound). */
  int(bound: number): number {
    return bound <= 1 ? 0 : this.next() % bound;
  }
  pick<T>(items: readonly T[]): T {
    return items[this.int(items.length)]!;
  }
}

export type SyntheticWorkflow = {
  /** Context label (app / task) the sequences are primed with. */
  context: string;
  /** The canonical ordered movements of the workflow. */
  template: MovementStep[];
};

/**
 * A library of small, realistic movement workflows. Each is a repeatable
 * task a user might perform; the generator instantiates them with variation.
 */
export const DEFAULT_SYNTHETIC_WORKFLOWS: SyntheticWorkflow[] = [
  {
    context: "editor",
    template: [
      { actor: "window", action: "focus", target: "editor" },
      { actor: "mouse", action: "click", target: "file:tree" },
      { actor: "keyboard", action: "shortcut", value: "cmd+p" },
      { actor: "keyboard", action: "type", target: "field:quick-open", value: "readme" },
      { actor: "keyboard", action: "shortcut", value: "enter" },
      { actor: "mouse", action: "scroll", direction: "down" },
    ],
  },
  {
    context: "browser",
    template: [
      { actor: "window", action: "focus", target: "browser" },
      { actor: "mouse", action: "click", target: "field:address" },
      { actor: "keyboard", action: "type", target: "field:address", value: "docs" },
      { actor: "keyboard", action: "shortcut", value: "enter" },
      { actor: "mouse", action: "click", target: "link:first-result" },
      { actor: "mouse", action: "scroll", direction: "down" },
    ],
  },
  {
    context: "canvas",
    template: [
      { actor: "window", action: "focus", target: "canvas" },
      { actor: "gesture", action: "swipe", direction: "left" },
      { actor: "mouse", action: "click", target: "tool:brush" },
      { actor: "gesture", action: "swipe", direction: "right" },
      { actor: "keyboard", action: "shortcut", value: "cmd+s" },
    ],
  },
];

export type SyntheticDatasetOptions = {
  seed: number;
  /** Sequences generated per workflow for the training split. Default 8. */
  trainPerWorkflow?: number;
  /** Sequences generated per workflow for the held-out split. Default 3. */
  heldOutPerWorkflow?: number;
  /** 0..1 probability a given step is perturbed (variation). Default 0.2. */
  variation?: number;
  workflows?: SyntheticWorkflow[];
};

export type SyntheticDataset = {
  train: MovementSequence[];
  heldOut: MovementSequence[];
};

const OPTIONAL_INSERTS: MovementStep[] = [
  { actor: "mouse", action: "move", direction: "down" },
  { actor: "window", action: "focus", target: "sidebar" },
  { actor: "keyboard", action: "shortcut", value: "esc" },
];

/**
 * Produce a reproducible train / held-out split. Held-out sequences share each
 * workflow's structure but are generated from an offset seed with the same
 * variation, so they are *related but new* — the right target for measuring
 * generalization rather than memorization.
 */
export function generateSyntheticDataset(options: SyntheticDatasetOptions): SyntheticDataset {
  const workflows = options.workflows ?? DEFAULT_SYNTHETIC_WORKFLOWS;
  const trainPerWorkflow = options.trainPerWorkflow ?? 8;
  const heldOutPerWorkflow = options.heldOutPerWorkflow ?? 3;
  const variation = clamp01(options.variation ?? 0.2);

  const build = (splitSeed: number, count: number, tag: string): MovementSequence[] => {
    const rng = new Lcg(splitSeed);
    const sequences: MovementSequence[] = [];
    for (const workflow of workflows) {
      for (let i = 0; i < count; i += 1) {
        sequences.push({
          id: `${tag}-${workflow.context}-${i}`,
          context: workflow.context,
          steps: perturbWorkflow(workflow.template, rng, variation),
        });
      }
    }
    return sequences;
  };

  return {
    train: build(options.seed, trainPerWorkflow, "train"),
    // Offset (not equal) seed → different variation draws → held-out ≠ train.
    heldOut: build(options.seed + 0x9e3779b9, heldOutPerWorkflow, "held-out"),
  };
}

function perturbWorkflow(template: MovementStep[], rng: Lcg, variation: number): MovementStep[] {
  const steps: MovementStep[] = [];
  for (const step of template) {
    if (variation > 0 && rng.float() < variation) {
      // Occasionally insert an incidental movement before this step.
      steps.push({ ...rng.pick(OPTIONAL_INSERTS) });
    }
    steps.push({ ...step });
  }
  return steps;
}

function clamp01(value: number): number {
  if (Number.isNaN(value)) {
    return 0;
  }
  return Math.min(1, Math.max(0, value));
}

/**
 * Bridge: turn a recorded {@link ReplayManifest} into movement sequences, one
 * per trajectory, so the model consumes real captured data through the same
 * path as synthetic data. Action events map to movements; observations and
 * transcript events are context, not movements, so they are skipped.
 */
export function movementSequencesFromReplay(manifest: ReplayManifest): MovementSequence[] {
  const byTrajectory = new Map<string, MovementStep[]>();
  for (const event of manifest.events) {
    if (event.kind !== "action") {
      continue;
    }
    const steps = byTrajectory.get(event.trajectoryId) ?? [];
    steps.push(movementStepFromReplayAction(event));
    byTrajectory.set(event.trajectoryId, steps);
  }
  return [...byTrajectory.entries()]
    .filter(([, steps]) => steps.length > 0)
    .map(([trajectoryId, steps]) => ({
      id: `${manifest.sessionId}:${trajectoryId}`,
      context: manifest.sessionId,
      steps,
    }));
}

function movementStepFromReplayAction(
  event: Extract<ReplayTimelineEvent, { kind: "action" }>,
): MovementStep {
  return {
    actor: event.tool === "device" ? "gesture" : "tool",
    action: normalizeVerb(event.summary),
    target: event.tool,
    value: event.summary,
  };
}

/** Map a device gesture kind to a movement step (used by the device capture path). */
export function movementStepFromDeviceGesture(gesture: {
  kind: DeviceGestureKind;
  target?: string;
  direction?: "up" | "down" | "left" | "right";
  valueSummary?: string;
}): MovementStep {
  const actionByKind: Record<DeviceGestureKind, string> = {
    tap: "click",
    swipe: "swipe",
    scroll: "scroll",
    type: "type",
    shortcut: "shortcut",
  };
  return {
    actor: "gesture",
    action: actionByKind[gesture.kind],
    ...(gesture.target ? { target: gesture.target } : {}),
    ...(gesture.direction ? { direction: gesture.direction } : {}),
    ...(gesture.valueSummary ? { value: gesture.valueSummary } : {}),
  };
}

/** Canonicalize the leading past-tense verb of a capture summary to a base action. */
const VERB_STEMS: Record<string, string> = {
  tapped: "tap",
  clicked: "click",
  typed: "type",
  swiped: "swipe",
  scrolled: "scroll",
  triggered: "shortcut",
  focused: "focus",
  moved: "move",
  pressed: "press",
  dragged: "drag",
  opened: "open",
  closed: "close",
};

function normalizeVerb(summary: string): string {
  const first = summary.trim().split(/\s+/)[0]?.toLowerCase() ?? "act";
  return VERB_STEMS[first] ?? first;
}
