/**
 * Low-level local-movement event schema and replayable dataset format.
 *
 * Unlike `src/capture` (which records human-readable summaries such as
 * "tapped Deploy"), these events are raw pointer/keyboard/window samples with
 * millisecond timing, so an episode can be replayed and learned from.
 */

export const MOVEMENT_DATASET_VERSION = 1 as const;

export type PointerButton = "left" | "right" | "middle";
export type KeyModifier = "shift" | "ctrl" | "alt" | "meta";

export type Point = { x: number; y: number };
export type ScreenSize = { width: number; height: number };

/** `t` is milliseconds since the start of the episode. Coordinates are screen pixels. */
export type MovementEvent =
  | { type: "pointer.move"; t: number; x: number; y: number }
  | { type: "pointer.down"; t: number; x: number; y: number; button: PointerButton }
  | { type: "pointer.up"; t: number; x: number; y: number; button: PointerButton }
  | { type: "scroll"; t: number; x: number; y: number; dx: number; dy: number }
  | { type: "key.down"; t: number; key: string; modifiers?: KeyModifier[] }
  | { type: "key.up"; t: number; key: string; modifiers?: KeyModifier[] }
  | {
      type: "window.focus";
      t: number;
      appId: string;
      title?: string;
      bounds?: { x: number; y: number; width: number; height: number };
    };

export type MovementEventType = MovementEvent["type"];

export type MovementEpisode = {
  version: typeof MOVEMENT_DATASET_VERSION;
  id: string;
  /** Task label used to group demonstrations of "the same movement". */
  label: string;
  createdAt: string;
  source: "simulated" | "os";
  screen: ScreenSize;
  events: MovementEvent[];
};

const POINTER_TYPES = new Set<MovementEventType>(["pointer.move", "pointer.down", "pointer.up", "scroll"]);
const BUTTONS = new Set<string>(["left", "right", "middle"]);
const MODIFIERS = new Set<string>(["shift", "ctrl", "alt", "meta"]);

export function isPointerEvent(
  event: MovementEvent,
): event is Extract<MovementEvent, { x: number; y: number }> {
  return POINTER_TYPES.has(event.type);
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

/** Returns a list of problems; empty means the event is valid. */
export function validateMovementEvent(value: unknown): string[] {
  if (!value || typeof value !== "object") {
    return ["event must be an object"];
  }
  const event = value as Record<string, unknown>;
  const errors: string[] = [];
  if (!isFiniteNumber(event.t) || event.t < 0) {
    errors.push("t must be a non-negative number");
  }
  switch (event.type) {
    case "pointer.move":
    case "pointer.down":
    case "pointer.up":
    case "scroll":
      if (!isFiniteNumber(event.x) || !isFiniteNumber(event.y)) {
        errors.push(`${event.type} requires numeric x/y`);
      }
      if ((event.type === "pointer.down" || event.type === "pointer.up") && !BUTTONS.has(String(event.button))) {
        errors.push(`${event.type} requires button left|right|middle`);
      }
      if (event.type === "scroll" && (!isFiniteNumber(event.dx) || !isFiniteNumber(event.dy))) {
        errors.push("scroll requires numeric dx/dy");
      }
      break;
    case "key.down":
    case "key.up":
      if (typeof event.key !== "string" || event.key.length === 0) {
        errors.push(`${event.type} requires a non-empty key`);
      }
      if (
        event.modifiers !== undefined &&
        (!Array.isArray(event.modifiers) || event.modifiers.some((m) => !MODIFIERS.has(String(m))))
      ) {
        errors.push(`${event.type} has invalid modifiers`);
      }
      break;
    case "window.focus":
      if (typeof event.appId !== "string" || event.appId.length === 0) {
        errors.push("window.focus requires appId");
      }
      break;
    default:
      errors.push(`unknown event type: ${String(event.type)}`);
  }
  return errors;
}

export function validateMovementEpisode(value: unknown): string[] {
  if (!value || typeof value !== "object") {
    return ["episode must be an object"];
  }
  const episode = value as Record<string, unknown>;
  const errors: string[] = [];
  if (episode.version !== MOVEMENT_DATASET_VERSION) {
    errors.push(`unsupported episode version: ${String(episode.version)}`);
  }
  if (typeof episode.id !== "string" || !episode.id) errors.push("id is required");
  if (typeof episode.label !== "string" || !episode.label) errors.push("label is required");
  const screen = episode.screen as Record<string, unknown> | undefined;
  if (!screen || !isFiniteNumber(screen.width) || !isFiniteNumber(screen.height)) {
    errors.push("screen width/height are required");
  }
  if (!Array.isArray(episode.events)) {
    errors.push("events must be an array");
    return errors;
  }
  let lastT = -Infinity;
  episode.events.forEach((event, index) => {
    for (const problem of validateMovementEvent(event)) {
      errors.push(`events[${index}]: ${problem}`);
    }
    const t = (event as { t?: unknown })?.t;
    if (isFiniteNumber(t)) {
      if (t < lastT) errors.push(`events[${index}]: timestamps must be non-decreasing`);
      lastT = t;
    }
  });
  return errors;
}

export type MovementDatasetHeader = {
  kind: "movement-dataset";
  version: typeof MOVEMENT_DATASET_VERSION;
};

/** Encode episodes as JSONL: a header line followed by one episode per line. */
export function encodeMovementDataset(episodes: MovementEpisode[]): string {
  const header: MovementDatasetHeader = { kind: "movement-dataset", version: MOVEMENT_DATASET_VERSION };
  return [header, ...episodes].map((line) => JSON.stringify(line)).join("\n") + "\n";
}

export type DecodedMovementDataset = {
  episodes: MovementEpisode[];
  /** Lines that failed to parse or validate; they are skipped, not fatal. */
  rejected: Array<{ line: number; errors: string[] }>;
};

export function decodeMovementDataset(jsonl: string): DecodedMovementDataset {
  const lines = jsonl.split("\n");
  const episodes: MovementEpisode[] = [];
  const rejected: DecodedMovementDataset["rejected"] = [];
  let sawHeader = false;
  lines.forEach((raw, index) => {
    const text = raw.trim();
    if (!text) return;
    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch {
      rejected.push({ line: index + 1, errors: ["invalid JSON"] });
      return;
    }
    if (!sawHeader) {
      const header = parsed as Partial<MovementDatasetHeader>;
      if (header.kind !== "movement-dataset" || header.version !== MOVEMENT_DATASET_VERSION) {
        throw new Error("not a movement dataset (missing or unsupported header)");
      }
      sawHeader = true;
      return;
    }
    const errors = validateMovementEpisode(parsed);
    if (errors.length > 0) {
      rejected.push({ line: index + 1, errors });
      return;
    }
    episodes.push(parsed as MovementEpisode);
  });
  if (!sawHeader) {
    throw new Error("not a movement dataset (empty)");
  }
  return { episodes, rejected };
}

export function episodeDuration(episode: MovementEpisode): number {
  return episode.events.length === 0 ? 0 : episode.events[episode.events.length - 1]!.t;
}

/** The first and last pointer positions in the episode, if it has any pointer events. */
export function pointerAnchors(episode: MovementEpisode): { start: Point; end: Point } | undefined {
  const pointer = episode.events.filter(isPointerEvent);
  if (pointer.length === 0) return undefined;
  const first = pointer[0]!;
  const last = pointer[pointer.length - 1]!;
  return { start: { x: first.x, y: first.y }, end: { x: last.x, y: last.y } };
}
