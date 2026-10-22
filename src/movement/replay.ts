import { isPointerEvent, type MovementEpisode, type MovementEvent } from "./events.js";

/**
 * Pluggable output side. A real OS actuator (CGEventPost, uinput, SendInput)
 * implements this outside tests; `RecordingActuator` is the safe default.
 */
export interface InputActuator {
  perform(event: MovementEvent): Promise<void> | void;
}

/** Collects performed events instead of touching the OS. */
export class RecordingActuator implements InputActuator {
  readonly performed: MovementEvent[] = [];

  perform(event: MovementEvent): void {
    this.performed.push(event);
  }
}

export type ReplayOptions = {
  /** Playback speed multiplier; 2 = twice as fast. Must be > 0. */
  speed?: number;
  sleep?: (ms: number) => Promise<void>;
  signal?: AbortSignal;
  /** Allow-list of event types to perform; others are skipped (e.g. pointer-only). */
  allowTypes?: ReadonlyArray<MovementEvent["type"]>;
  /** Hard cap on events performed, a safety brake for runaway generated episodes. */
  maxEvents?: number;
};

export type ReplayResult = {
  performed: number;
  skipped: number;
  aborted: boolean;
  /** Total virtual time slept, in ms (after speed scaling). */
  sleptMs: number;
};

const defaultSleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

/**
 * Replays an episode against an actuator, preserving inter-event timing.
 * Pointer coordinates are clamped to the episode's screen as a safety guard,
 * and every `pointer.down` is guaranteed a matching `pointer.up` even when
 * replay is aborted or capped, so no button is left stuck pressed.
 */
export async function replayEpisode(
  episode: MovementEpisode,
  actuator: InputActuator,
  options: ReplayOptions = {},
): Promise<ReplayResult> {
  const speed = options.speed ?? 1;
  if (!(speed > 0)) {
    throw new Error("replay speed must be > 0");
  }
  const sleep = options.sleep ?? defaultSleep;
  const allow = options.allowTypes ? new Set(options.allowTypes) : undefined;
  const maxEvents = options.maxEvents ?? Infinity;
  const { width, height } = episode.screen;
  const pressed = new Map<string, Extract<MovementEvent, { type: "pointer.down" }>>();
  const heldKeys = new Map<string, Extract<MovementEvent, { type: "key.down" }>>();

  let performed = 0;
  let skipped = 0;
  let sleptMs = 0;
  let aborted = false;
  let lastT = episode.events[0]?.t ?? 0;

  for (const event of episode.events) {
    if (options.signal?.aborted || performed >= maxEvents) {
      aborted = true;
      break;
    }
    const wait = Math.max(0, event.t - lastT) / speed;
    lastT = event.t;
    if (wait > 0) {
      await sleep(wait);
      sleptMs += wait;
      if (options.signal?.aborted) {
        aborted = true;
        break;
      }
    }
    if (allow && !allow.has(event.type)) {
      skipped += 1;
      continue;
    }
    const safe = isPointerEvent(event)
      ? ({
          ...event,
          x: Math.min(width - 1, Math.max(0, event.x)),
          y: Math.min(height - 1, Math.max(0, event.y)),
        } as MovementEvent)
      : event;
    await actuator.perform(safe);
    performed += 1;
    if (safe.type === "pointer.down") pressed.set(safe.button, safe);
    if (safe.type === "pointer.up") pressed.delete(safe.button);
    if (safe.type === "key.down") heldKeys.set(safe.key, safe);
    if (safe.type === "key.up") heldKeys.delete(safe.key);
  }

  // Release anything still held so the machine is never left mid-gesture.
  for (const down of pressed.values()) {
    await actuator.perform({ type: "pointer.up", t: lastT, x: down.x, y: down.y, button: down.button });
  }
  for (const key of heldKeys.values()) {
    await actuator.perform({ type: "key.up", t: lastT, key: key.key, ...(key.modifiers ? { modifiers: key.modifiers } : {}) });
  }

  return { performed, skipped, aborted, sleptMs };
}
