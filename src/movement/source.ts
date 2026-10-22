import { defaultCapturePolicy, isSensitiveApp } from "../capture/trajectory.js";
import {
  MOVEMENT_DATASET_VERSION,
  type MovementEpisode,
  type MovementEvent,
  type Point,
  type ScreenSize,
} from "./events.js";

/**
 * A raw input sample as delivered by an OS hook: same shape as a
 * `MovementEvent` but with an absolute wall-clock timestamp in `t`.
 */
export type RawInputEvent = MovementEvent;

/**
 * Pluggable input source. Real OS hooks (CGEventTap, XInput, SetWindowsHookEx)
 * implement this outside the test path; tests use `SimulatedInputEventSource`.
 */
export interface InputEventSource {
  readonly kind: "simulated" | "os";
  readonly screen: ScreenSize;
  events(): AsyncIterable<RawInputEvent>;
}

/** Deterministic scripted source — no OS access. */
export class SimulatedInputEventSource implements InputEventSource {
  readonly kind = "simulated" as const;

  constructor(
    private readonly script: RawInputEvent[],
    readonly screen: ScreenSize = { width: 1920, height: 1080 },
  ) {}

  async *events(): AsyncIterable<RawInputEvent> {
    for (const event of this.script) {
      yield event;
    }
  }
}

export type RecordEpisodeOptions = {
  id: string;
  label: string;
  /** Stop after this many ms of episode time. */
  maxDurationMs?: number;
  maxEvents?: number;
  /** Returns true for apps whose keystrokes must never be recorded. */
  isSensitiveApp?: (appId: string) => boolean;
  /** Replace printable key values with "•" (keeps timing, drops content). */
  redactKeys?: boolean;
  now?: () => Date;
};

/**
 * Drain a source into a replayable episode: rebases timestamps to t=0, sorts
 * stably, and applies privacy rules. Keystrokes while a sensitive app (default:
 * the capture subsystem's denylist) is focused are dropped entirely.
 */
export async function recordEpisode(
  source: InputEventSource,
  options: RecordEpisodeOptions,
): Promise<MovementEpisode> {
  const policy = defaultCapturePolicy();
  const sensitive = options.isSensitiveApp ?? ((appId: string) => isSensitiveApp(appId, policy));
  const collected: MovementEvent[] = [];
  let origin: number | undefined;
  let focusedSensitive = false;

  for await (const raw of source.events()) {
    origin ??= raw.t;
    const t = Math.max(0, raw.t - origin);
    if (options.maxDurationMs !== undefined && t > options.maxDurationMs) break;
    if (options.maxEvents !== undefined && collected.length >= options.maxEvents) break;

    if (raw.type === "window.focus") {
      focusedSensitive = sensitive(raw.appId);
    }
    if ((raw.type === "key.down" || raw.type === "key.up") && focusedSensitive) {
      continue;
    }
    const event = { ...raw, t } as MovementEvent;
    if ((event.type === "key.down" || event.type === "key.up") && options.redactKeys && event.key.length === 1) {
      event.key = "•";
    }
    collected.push(event);
  }

  const events = collected
    .map((event, index) => ({ event, index }))
    .sort((a, b) => a.event.t - b.event.t || a.index - b.index)
    .map(({ event }) => event);

  return {
    version: MOVEMENT_DATASET_VERSION,
    id: options.id,
    label: options.label,
    createdAt: (options.now?.() ?? new Date()).toISOString(),
    source: source.kind,
    screen: { ...source.screen },
    events,
  };
}

/** Small seeded PRNG (mulberry32) so synthetic data is reproducible. */
export function seededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let x = state;
    x = Math.imul(x ^ (x >>> 15), x | 1);
    x ^= x + Math.imul(x ^ (x >>> 7), x | 61);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

export type SynthesizeDragOptions = {
  from: Point;
  to: Point;
  durationMs?: number;
  steps?: number;
  /** Perpendicular bow of the path as a fraction of its length. */
  curvature?: number;
  /** Pixel jitter amplitude applied to intermediate points. */
  jitter?: number;
  seed?: number;
  startT?: number;
};

/**
 * Human-like synthetic drag: press at `from`, minimum-jerk eased path bowed by
 * `curvature`, release at `to`. Used for tests and data augmentation.
 */
export function synthesizeDrag(options: SynthesizeDragOptions): RawInputEvent[] {
  const { from, to } = options;
  const duration = options.durationMs ?? 600;
  const steps = Math.max(2, options.steps ?? 20);
  const curvature = options.curvature ?? 0.15;
  const jitter = options.jitter ?? 0;
  const random = seededRandom(options.seed ?? 1);
  const t0 = options.startT ?? 0;
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const length = Math.hypot(dx, dy);
  const nx = length === 0 ? 0 : -dy / length;
  const ny = length === 0 ? 0 : dx / length;

  const events: RawInputEvent[] = [{ type: "pointer.down", t: t0, x: from.x, y: from.y, button: "left" }];
  for (let i = 1; i < steps; i += 1) {
    const u = i / steps;
    const s = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5; // minimum-jerk easing
    const bow = Math.sin(Math.PI * s) * curvature * length;
    const jx = jitter ? (random() * 2 - 1) * jitter : 0;
    const jy = jitter ? (random() * 2 - 1) * jitter : 0;
    events.push({
      type: "pointer.move",
      t: t0 + Math.round(u * duration),
      x: from.x + dx * s + nx * bow + jx,
      y: from.y + dy * s + ny * bow + jy,
    });
  }
  events.push({ type: "pointer.move", t: t0 + duration, x: to.x, y: to.y });
  events.push({ type: "pointer.up", t: t0 + duration, x: to.x, y: to.y, button: "left" });
  return events;
}
