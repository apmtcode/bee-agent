// Low-level local-movement event schema for the movement-learning subsystem.
//
// This is the replayable primitive layer beneath the high-level trajectory
// spans in `src/capture`: raw mouse / keyboard / scroll / window-focus events as
// they would be observed on a local computer. The subsystem records these to a
// structured dataset, trains a local model to repeat them, and generalizes to
// related movements. The actual OS capture runs on the user's machine; in the
// cloud we exercise the code with synthetic streams (see `synthetic.ts`).

export type MouseButton = "left" | "right" | "middle";

export type MovementModifier = "ctrl" | "alt" | "shift" | "meta";

/** A single observed movement/input event on the local computer. */
export type MovementEvent =
  | { kind: "mouse-move"; ts: number; x: number; y: number }
  | { kind: "mouse-down"; ts: number; x: number; y: number; button: MouseButton }
  | { kind: "mouse-up"; ts: number; x: number; y: number; button: MouseButton }
  | { kind: "scroll"; ts: number; x: number; y: number; dx: number; dy: number }
  | { kind: "key-down"; ts: number; key: string; modifiers: MovementModifier[] }
  | { kind: "key-up"; ts: number; key: string; modifiers: MovementModifier[] }
  | { kind: "window-focus"; ts: number; windowId: string; title?: string };

export type MovementEventKind = MovementEvent["kind"];

/** An ordered stream of movement events tagged with the intent/task it realizes. */
export type MovementTrajectory = {
  id: string;
  /** Task label / user intent this trajectory realizes, e.g. "open-and-type". */
  label: string;
  events: MovementEvent[];
};

export type MovementDataset = {
  quantizer: CoordinateQuantizer;
  trajectories: MovementTrajectory[];
};

/**
 * Discretizes continuous screen coordinates into a fixed grid so the model can
 * generalize across small positional jitter and so tokens stay finite. Cells are
 * addressed by `(col, row)`; the inverse maps a cell back to its centre point.
 */
export type CoordinateQuantizer = {
  width: number;
  height: number;
  cols: number;
  rows: number;
};

export function createCoordinateQuantizer(params: {
  width: number;
  height: number;
  cols: number;
  rows: number;
}): CoordinateQuantizer {
  if (params.width <= 0 || params.height <= 0) {
    throw new Error("quantizer requires positive width/height");
  }
  if (params.cols <= 0 || params.rows <= 0) {
    throw new Error("quantizer requires positive cols/rows");
  }
  return { ...params };
}

export function quantizeCell(q: CoordinateQuantizer, x: number, y: number): { col: number; row: number } {
  const clampedX = Math.min(Math.max(x, 0), q.width - 1e-9);
  const clampedY = Math.min(Math.max(y, 0), q.height - 1e-9);
  const col = Math.min(q.cols - 1, Math.floor((clampedX / q.width) * q.cols));
  const row = Math.min(q.rows - 1, Math.floor((clampedY / q.height) * q.rows));
  return { col, row };
}

export function cellCenter(q: CoordinateQuantizer, col: number, row: number): { x: number; y: number } {
  const x = ((col + 0.5) / q.cols) * q.width;
  const y = ((row + 0.5) / q.rows) * q.height;
  return { x, y };
}

/** Sentinel tokens delimit a trajectory so the model can condition on intent. */
export const MOVEMENT_BOS = "<bos>";
export const MOVEMENT_EOS = "<eos>";

export function labelToken(label: string): string {
  return `<label:${label}>`;
}

/**
 * Maps one event to a discrete token. Coordinates are quantized to grid cells,
 * modifiers are sorted for stability, and dynamic values (scroll deltas, typed
 * text) are bucketed so the token vocabulary stays finite and generalizable.
 */
export function tokenizeEvent(event: MovementEvent, q: CoordinateQuantizer): string {
  switch (event.kind) {
    case "mouse-move": {
      const { col, row } = quantizeCell(q, event.x, event.y);
      return `move:${col},${row}`;
    }
    case "mouse-down": {
      const { col, row } = quantizeCell(q, event.x, event.y);
      return `down:${event.button}:${col},${row}`;
    }
    case "mouse-up": {
      const { col, row } = quantizeCell(q, event.x, event.y);
      return `up:${event.button}:${col},${row}`;
    }
    case "scroll": {
      const { col, row } = quantizeCell(q, event.x, event.y);
      return `scroll:${signBucket(event.dx)}${signBucket(event.dy)}:${col},${row}`;
    }
    case "key-down":
      return `kd:${modifierPrefix(event.modifiers)}${normalizeKey(event.key)}`;
    case "key-up":
      return `ku:${modifierPrefix(event.modifiers)}${normalizeKey(event.key)}`;
    case "window-focus":
      return `focus:${event.windowId}`;
  }
}

export function tokenizeTrajectory(trajectory: MovementTrajectory, q: CoordinateQuantizer): string[] {
  return [
    MOVEMENT_BOS,
    labelToken(trajectory.label),
    ...trajectory.events.map((event) => tokenizeEvent(event, q)),
    MOVEMENT_EOS,
  ];
}

function signBucket(value: number): string {
  if (value > 0) return "+";
  if (value < 0) return "-";
  return "0";
}

function modifierPrefix(modifiers: MovementModifier[]): string {
  if (modifiers.length === 0) return "";
  const order: MovementModifier[] = ["ctrl", "alt", "shift", "meta"];
  const sorted = [...new Set(modifiers)].sort((a, b) => order.indexOf(a) - order.indexOf(b));
  return `${sorted.join("+")}-`;
}

/** Collapse single printable characters to a `char` class; keep named keys. */
function normalizeKey(key: string): string {
  if (key.length === 1) {
    if (/[a-z]/i.test(key)) return "char";
    if (/[0-9]/.test(key)) return "digit";
    return "punct";
  }
  return key.toLowerCase();
}
