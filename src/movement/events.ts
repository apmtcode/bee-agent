/**
 * Low-level local-movement event schema.
 *
 * This is the structured, replayable representation of physical input on the
 * user's machine — mouse, keyboard, scroll, and window/UI focus events. It sits
 * *below* the high-level gesture abstraction in `src/capture/device-adapter.ts`
 * (tap/swipe/type): a gesture is a semantic label; a {@link MovementEvent} is a
 * concrete, replayable sample with coordinates and timing.
 *
 * Everything here is pure data + pure functions so the capture → dataset →
 * train → replay pipeline can be validated deterministically in CI with
 * synthetic streams, with no access to a real OS.
 */

export type Point = { x: number; y: number };

export const POINTER_BUTTONS = ["left", "right", "middle"] as const;
export type PointerButton = (typeof POINTER_BUTTONS)[number];

export type MovementEvent =
  | { kind: "pointer-move"; ts: number; x: number; y: number }
  | { kind: "pointer-down"; ts: number; x: number; y: number; button: PointerButton }
  | { kind: "pointer-up"; ts: number; x: number; y: number; button: PointerButton }
  | { kind: "scroll"; ts: number; x: number; y: number; dx: number; dy: number }
  | { kind: "key-down"; ts: number; key: string }
  | { kind: "key-up"; ts: number; key: string }
  | { kind: "window-focus"; ts: number; window: string };

export type MovementEventKind = MovementEvent["kind"];

/**
 * A single recorded demonstration: an ordered, time-stamped event stream plus
 * the semantic reference frame it was performed in. The frame's `start`/`end`
 * anchors let a model re-target the demonstration to new coordinates while
 * preserving its shape (see {@link "./model"}).
 */
export type MovementDemonstration = {
  id: string;
  intent: string;
  /** Events sorted ascending by `ts`. Invariant enforced by builders below. */
  events: MovementEvent[];
  frame: { start: Point; end: Point };
  metadata?: Record<string, unknown>;
};

export type MovementDataset = {
  version: 1;
  demonstrations: MovementDemonstration[];
};

/** Events that carry an (x, y) coordinate that can be spatially transformed. */
export function isPositionedEvent(
  event: MovementEvent,
): event is Extract<MovementEvent, { x: number; y: number }> {
  return (
    event.kind === "pointer-move" ||
    event.kind === "pointer-down" ||
    event.kind === "pointer-up" ||
    event.kind === "scroll"
  );
}

export function pointOf(event: MovementEvent): Point | undefined {
  return isPositionedEvent(event) ? { x: event.x, y: event.y } : undefined;
}

/** Extract just the pointer path (one point per positioned event), in order. */
export function pointerPath(events: MovementEvent[]): Point[] {
  const path: Point[] = [];
  for (const event of events) {
    if (isPositionedEvent(event)) {
      path.push({ x: event.x, y: event.y });
    }
  }
  return path;
}

export function sortEvents(events: MovementEvent[]): MovementEvent[] {
  return [...events].sort((a, b) => a.ts - b.ts);
}

export function buildMovementDemonstration(params: {
  id: string;
  intent: string;
  events: MovementEvent[];
  frame: { start: Point; end: Point };
  metadata?: Record<string, unknown>;
}): MovementDemonstration {
  return {
    id: params.id,
    intent: params.intent,
    events: sortEvents(params.events),
    frame: { start: { ...params.frame.start }, end: { ...params.frame.end } },
    ...(params.metadata ? { metadata: { ...params.metadata } } : {}),
  };
}

export function buildMovementDataset(
  demonstrations: MovementDemonstration[],
): MovementDataset {
  return { version: 1, demonstrations: [...demonstrations] };
}

/** Distinct intents present in a dataset, in first-seen order. */
export function datasetIntents(dataset: MovementDataset): string[] {
  const seen = new Set<string>();
  const intents: string[] = [];
  for (const demonstration of dataset.demonstrations) {
    if (!seen.has(demonstration.intent)) {
      seen.add(demonstration.intent);
      intents.push(demonstration.intent);
    }
  }
  return intents;
}

export function euclideanDistance(a: Point, b: Point): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}
