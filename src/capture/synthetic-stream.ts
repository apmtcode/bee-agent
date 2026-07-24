import type { ReplayTimelineEvent } from "./replay.js";

/**
 * Deterministic synthetic movement-stream generator (objective #2).
 *
 * bee-agent runs in the cloud with no access to a real machine, so the capture
 * pipeline cannot observe real mouse/keyboard/window events here. This module
 * synthesises realistic, *deterministic* movement streams — from a small grammar
 * of task templates plus a seeded PRNG — so the capture → dataset → replay →
 * train → eval loop can be exercised end to end in tests without any OS input.
 *
 * Determinism (a seeded linear-congruential generator, never Math.random) keeps
 * generated corpora reproducible across runs and machines.
 */

/** Seeded LCG (Numerical Recipes constants) — reproducible, no global state. */
class SeededRandom {
  private state: number;

  constructor(seed: number) {
    // Normalise into a non-zero 32-bit state.
    this.state = (seed >>> 0) || 0x9e3779b9;
  }

  next(): number {
    this.state = (Math.imul(this.state, 1664525) + 1013904223) >>> 0;
    return this.state / 0x100000000;
  }

  int(maxExclusive: number): number {
    if (maxExclusive <= 0) {
      return 0;
    }
    return Math.floor(this.next() * maxExclusive);
  }

  pick<T>(items: readonly T[]): T {
    return items[this.int(items.length)];
  }
}

export type MovementTemplateStep = {
  kind: ReplayTimelineEvent["kind"];
  /** Candidate identifiers (tool / source / role). One is chosen per instance. */
  ids: string[];
  /** Optional human summary; a generic one is derived when omitted. */
  summary?: string;
};

export type MovementTemplate = {
  name: string;
  steps: MovementTemplateStep[];
};

/**
 * A default library of task templates modelling common desktop movement flows.
 * Templates share sub-flows (e.g. "focus → click → type") so a model trained on
 * some can be evaluated for generalisation on the others.
 */
export const DEFAULT_MOVEMENT_TEMPLATES: MovementTemplate[] = [
  {
    name: "open-and-edit-document",
    steps: [
      { kind: "observation", ids: ["window.focus"] },
      { kind: "action", ids: ["app.launch", "app.activate"] },
      { kind: "action", ids: ["mouse.move"] },
      { kind: "action", ids: ["mouse.click"] },
      { kind: "action", ids: ["keyboard.type"] },
      { kind: "action", ids: ["file.save"] },
    ],
  },
  {
    name: "search-and-select",
    steps: [
      { kind: "observation", ids: ["window.focus"] },
      { kind: "action", ids: ["mouse.click"] },
      { kind: "action", ids: ["keyboard.type"] },
      { kind: "action", ids: ["keyboard.enter"] },
      { kind: "action", ids: ["mouse.move"] },
      { kind: "action", ids: ["mouse.click"] },
    ],
  },
  {
    name: "copy-between-windows",
    steps: [
      { kind: "observation", ids: ["window.focus"] },
      { kind: "action", ids: ["mouse.click"] },
      { kind: "action", ids: ["keyboard.copy"] },
      { kind: "action", ids: ["window.switch"] },
      { kind: "action", ids: ["mouse.click"] },
      { kind: "action", ids: ["keyboard.paste"] },
      { kind: "action", ids: ["file.save"] },
    ],
  },
];

export type SyntheticStreamOptions = {
  seed?: number;
  /** Number of replay sequences to generate. Defaults to 8. */
  sequenceCount?: number;
  /** Templates to sample from. Defaults to {@link DEFAULT_MOVEMENT_TEMPLATES}. */
  templates?: MovementTemplate[];
  /** Base epoch millis for the first event. Defaults to 0. */
  startTs?: number;
  /** Millis between consecutive events (jittered deterministically). */
  stepMillis?: number;
};

export type SyntheticReplay = {
  sessionId: string;
  templateName: string;
  trajectoryIds: string[];
  eventCount: number;
  events: ReplayTimelineEvent[];
};

function summaryFor(kind: ReplayTimelineEvent["kind"], id: string): string {
  switch (kind) {
    case "action":
      return `perform ${id}`;
    case "observation":
      return `observe ${id}`;
    case "transcript":
      return `${id} message`;
  }
}

function buildEvent(
  kind: ReplayTimelineEvent["kind"],
  id: string,
  ts: number,
  trajectoryId: string,
  summary: string,
): ReplayTimelineEvent {
  switch (kind) {
    case "action":
      return { kind, ts, trajectoryId, tool: id, summary };
    case "observation":
      return { kind, ts, trajectoryId, source: id, summary };
    case "transcript":
      return {
        kind,
        ts,
        messageId: `${trajectoryId}-${ts}`,
        role: id === "user" || id === "assistant" || id === "system" || id === "tool" ? id : "assistant",
        content: summary,
      };
  }
}

/**
 * Generate a deterministic set of synthetic replay sequences. Same seed and
 * options always yield byte-identical output.
 */
export function generateSyntheticReplays(options: SyntheticStreamOptions = {}): SyntheticReplay[] {
  const rng = new SeededRandom(options.seed ?? 1);
  const templates = options.templates ?? DEFAULT_MOVEMENT_TEMPLATES;
  const sequenceCount = options.sequenceCount ?? 8;
  const startTs = options.startTs ?? 0;
  const stepMillis = options.stepMillis ?? 1000;
  if (templates.length === 0) {
    return [];
  }

  const replays: SyntheticReplay[] = [];
  for (let i = 0; i < sequenceCount; i += 1) {
    const template = rng.pick(templates);
    const trajectoryId = `synthetic-traj-${i}`;
    let ts = startTs + i * stepMillis * 100;
    const events: ReplayTimelineEvent[] = template.steps.map((step) => {
      const id = rng.pick(step.ids);
      const summary = step.summary ?? summaryFor(step.kind, id);
      const event = buildEvent(step.kind, id, ts, trajectoryId, summary);
      ts += stepMillis + rng.int(stepMillis);
      return event;
    });
    replays.push({
      sessionId: `synthetic-session-${i}`,
      templateName: template.name,
      trajectoryIds: [trajectoryId],
      eventCount: events.length,
      events,
    });
  }
  return replays;
}

/**
 * Split replays into train/held-out partitions for generalisation evaluation.
 * Deterministic: takes the first `trainFraction` share as train.
 */
export function partitionReplays<T>(
  replays: T[],
  trainFraction = 0.7,
): { train: T[]; heldOut: T[] } {
  const clamped = Math.min(0.95, Math.max(0.05, trainFraction));
  const splitAt = Math.max(1, Math.floor(replays.length * clamped));
  return {
    train: replays.slice(0, splitAt),
    heldOut: replays.slice(splitAt),
  };
}
