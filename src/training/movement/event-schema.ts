// Fine-grained local-movement event schema for the learning subsystem
// (standing objective #2a). These are lower-level than the high-level
// `TrajectoryAction`/`DeviceGesture` records in `src/capture/`: they carry the
// raw pointer coordinates and key transitions a local model needs to learn to
// reproduce and generalize physical movements. Everything here is pure data —
// no OS access — so capture pipelines, synthetic generators, and tests all
// speak the same format.

export type PointerButton = "left" | "right" | "middle";

export type Point = { x: number; y: number };

export type PointerMoveEvent = {
  kind: "pointer-move";
  ts: number;
  x: number;
  y: number;
};

export type PointerButtonEvent = {
  kind: "pointer-down" | "pointer-up";
  ts: number;
  x: number;
  y: number;
  button: PointerButton;
};

export type ScrollEvent = {
  kind: "scroll";
  ts: number;
  x: number;
  y: number;
  dx: number;
  dy: number;
};

export type KeyEvent = {
  kind: "key-down" | "key-up";
  ts: number;
  key: string;
};

export type MovementEvent =
  | PointerMoveEvent
  | PointerButtonEvent
  | ScrollEvent
  | KeyEvent;

export const MOVEMENT_EVENT_KINDS = [
  "pointer-move",
  "pointer-down",
  "pointer-up",
  "scroll",
  "key-down",
  "key-up",
] as const;

/**
 * A single recorded (or synthesized) movement gesture. `target`, when present,
 * is the semantic goal of the gesture (e.g. the pixel a click was aiming at) and
 * is what lets a model generalize to unseen targets.
 */
export type MovementTrajectory = {
  id: string;
  label: string;
  target?: Point;
  events: MovementEvent[];
};

export function isPointerEvent(
  event: MovementEvent,
): event is PointerMoveEvent | PointerButtonEvent | ScrollEvent {
  return event.kind !== "key-down" && event.kind !== "key-up";
}

/**
 * Return the pointer positions of a trajectory in event order. Key events carry
 * no position and are skipped.
 */
export function pointerPath(trajectory: MovementTrajectory): Point[] {
  const path: Point[] = [];
  for (const event of trajectory.events) {
    if (isPointerEvent(event)) {
      path.push({ x: event.x, y: event.y });
    }
  }
  return path;
}

export type MovementValidationIssue = {
  index: number;
  message: string;
};

/**
 * Structural validation of a trajectory. Checks event kinds, finite numeric
 * fields, and that timestamps are non-decreasing (movement is inherently
 * ordered in time). Returns every issue found rather than throwing so callers
 * can surface all problems in a captured/imported dataset at once.
 */
export function validateMovementTrajectory(
  trajectory: MovementTrajectory,
): MovementValidationIssue[] {
  const issues: MovementValidationIssue[] = [];
  if (!trajectory.id) {
    issues.push({ index: -1, message: "trajectory is missing an id" });
  }
  let previousTs = Number.NEGATIVE_INFINITY;
  trajectory.events.forEach((event, index) => {
    if (!MOVEMENT_EVENT_KINDS.includes(event.kind)) {
      issues.push({ index, message: `unknown event kind "${event.kind}"` });
    }
    if (!Number.isFinite(event.ts)) {
      issues.push({ index, message: "event ts is not finite" });
    } else if (event.ts < previousTs) {
      issues.push({
        index,
        message: `event ts ${event.ts} precedes previous ts ${previousTs}`,
      });
    }
    if (Number.isFinite(event.ts)) {
      previousTs = event.ts;
    }
    if (isPointerEvent(event)) {
      if (!Number.isFinite(event.x) || !Number.isFinite(event.y)) {
        issues.push({ index, message: "pointer event has non-finite coordinates" });
      }
    } else if (!event.key) {
      issues.push({ index, message: "key event has empty key" });
    }
  });
  return issues;
}

/**
 * Return a copy with events sorted by timestamp (stable for equal ts). Recording
 * pipelines may interleave sources; downstream dataset building assumes order.
 */
export function normalizeMovementTrajectory(
  trajectory: MovementTrajectory,
): MovementTrajectory {
  const events = trajectory.events
    .map((event, index) => ({ event, index }))
    .sort((a, b) => (a.event.ts !== b.event.ts ? a.event.ts - b.event.ts : a.index - b.index))
    .map((entry) => entry.event);
  return { ...trajectory, events };
}
