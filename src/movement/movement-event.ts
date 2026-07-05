import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

/**
 * Canonical low-level movement schema for the local-movement learning subsystem.
 *
 * A {@link MovementEvent} is the replayable atom the sequence model trains on:
 * a single mouse / keyboard / window / gesture action, normalized so that it is
 * (a) coarse enough to generalize across sessions and (b) precise enough to
 * replay. Events are grouped into ordered {@link MovementSequence}s, and a
 * collection of sequences forms a {@link MovementDataset} — the unit the
 * pluggable model backends consume.
 *
 * This layer is deliberately OS-agnostic and side-effect free: the real
 * on-device capture (see `src/capture/`) produces these events, and this module
 * only defines the schema, tokenization, and a bridge from recorded
 * trajectories so the training/eval code can run in the cloud on synthetic data.
 */
export const MOVEMENT_EVENT_KINDS = [
  "pointer-move",
  "pointer-down",
  "pointer-up",
  "click",
  "scroll",
  "key-type",
  "shortcut",
  "focus",
  "wait",
] as const;

export type MovementEventKind = (typeof MOVEMENT_EVENT_KINDS)[number];

export type MovementDirection = "up" | "down" | "left" | "right";
export type MovementButton = "left" | "right" | "middle";

export type MovementEvent = {
  kind: MovementEventKind;
  /** Milliseconds since the start of the owning sequence. */
  ts: number;
  /** Stable UI target / element identifier the movement acted on. */
  target?: string;
  /** Key or shortcut chord (e.g. "cmd+s"), for keyboard movements. */
  key?: string;
  button?: MovementButton;
  direction?: MovementDirection;
  /** Redacted summary of typed value — never raw keystrokes. */
  value?: string;
  appId?: string;
};

export type MovementSequence = {
  id: string;
  /** Optional human intent this sequence accomplishes (e.g. "save document"). */
  intent?: string;
  appId?: string;
  events: MovementEvent[];
};

export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

/**
 * Discrete token for one movement event. The vocabulary is intentionally coarse
 * (kind + salient qualifier) so a small model can learn transition structure and
 * generalize to new-but-related sequences rather than memorizing coordinates.
 */
export function tokenizeMovementEvent(event: MovementEvent): string {
  switch (event.kind) {
    case "click":
    case "pointer-down":
    case "pointer-up":
      return `${event.kind}:${event.target ?? event.button ?? "surface"}`;
    case "pointer-move":
      return `pointer-move:${event.target ?? "surface"}`;
    case "scroll":
      return `scroll:${event.direction ?? "down"}`;
    case "key-type":
      return `key-type:${event.target ?? "field"}`;
    case "shortcut":
      return `shortcut:${event.key ?? event.target ?? "chord"}`;
    case "focus":
      return `focus:${event.appId ?? event.target ?? "window"}`;
    case "wait":
      return "wait";
  }
}

export function tokenizeSequence(sequence: MovementSequence): string[] {
  return sequence.events.map(tokenizeMovementEvent);
}

export const MOVEMENT_SEQUENCE_START = "<start>";
export const MOVEMENT_SEQUENCE_END = "<end>";

export function isSpecialToken(token: string): boolean {
  return token === MOVEMENT_SEQUENCE_START || token === MOVEMENT_SEQUENCE_END;
}

/**
 * Bridge recorded {@link TrajectorySpan} actions into a {@link MovementSequence}
 * so real captured trajectories can feed the same model backends as synthetic
 * data. Device gestures (see `src/capture/device-adapter.ts`) carry their
 * gesture kind/target/direction in `metadata`; we map those into movement kinds.
 */
export function movementSequenceFromTrajectory(trajectory: TrajectorySpan): MovementSequence {
  const baseTs = trajectory.actions[0]?.ts ?? 0;
  const events = trajectory.actions.map((action) => movementEventFromAction(action, baseTs));
  return {
    id: trajectory.id,
    ...(trajectory.outcome?.summary ? { intent: trajectory.outcome.summary } : {}),
    events,
  };
}

function movementEventFromAction(action: TrajectoryAction, baseTs: number): MovementEvent {
  const metadata = action.metadata ?? {};
  const gesture = typeof metadata.gesture === "string" ? metadata.gesture : undefined;
  const target = typeof metadata.target === "string" ? metadata.target : undefined;
  const direction = normalizeDirection(metadata.direction);
  const valueSummary = typeof metadata.valueSummary === "string" ? metadata.valueSummary : undefined;
  const ts = Math.max(0, action.ts - baseTs);

  const kind = movementKindFromGesture(gesture);
  const event: MovementEvent = { kind, ts };
  if (target) {
    event.target = target;
  }
  if (direction) {
    event.direction = direction;
  }
  if (kind === "shortcut" && target) {
    event.key = target;
  }
  if (kind === "key-type" && valueSummary) {
    event.value = valueSummary;
  }
  return event;
}

function movementKindFromGesture(gesture: string | undefined): MovementEventKind {
  switch (gesture) {
    case "tap":
      return "click";
    case "swipe":
    case "scroll":
      return "scroll";
    case "type":
      return "key-type";
    case "shortcut":
      return "shortcut";
    default:
      return "click";
  }
}

function normalizeDirection(value: unknown): MovementDirection | undefined {
  if (value === "up" || value === "down" || value === "left" || value === "right") {
    return value;
  }
  return undefined;
}

export function buildMovementDataset(sequences: MovementSequence[]): MovementDataset {
  return { version: 1, sequences };
}
