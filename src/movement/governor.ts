import type { MovementEvent, Point } from "./events.js";
import type { InputActuator } from "./replay.js";

export type ScreenRegion = { x: number; y: number; width: number; height: number };

export type GovernorViolationKind = "rate" | "velocity" | "region";

export type GovernorViolation = {
  kind: GovernorViolationKind;
  event: MovementEvent;
  /** What the governor did: rejected (threw), clamped (rewrote), or delayed (waited). */
  action: "rejected" | "clamped" | "delayed";
  detail: string;
};

export type GovernedActuatorOptions = {
  /** Max events forwarded per rolling 1000ms window. Releases are exempt. */
  maxEventsPerSecond?: number;
  /** Max pointer speed in px per wall-clock ms between forwarded pointer events. */
  maxPointerSpeedPxPerMs?: number;
  /** Pointer events must land inside one of these regions. Empty/undefined = whole screen. */
  regions?: ScreenRegion[];
  /**
   * When true, a forwarded `window.focus` event carrying `bounds` replaces the
   * allowed regions with exactly that window, so a generated episode is
   * confined to the app it targets even if the window moves.
   */
  followFocus?: boolean;
  /**
   * "reject" throws `GovernorViolationError` (nothing reaches the OS);
   * "clamp" rewrites coordinates into the region / speed envelope and delays
   * rate-limited events until the window frees up. Default "reject".
   */
  mode?: "reject" | "clamp";
  now?: () => number;
  sleep?: (ms: number) => Promise<void>;
  onViolation?: (violation: GovernorViolation) => void;
};

export class GovernorViolationError extends Error {
  constructor(readonly violation: GovernorViolation) {
    super(`actuation governor rejected ${violation.event.type}: ${violation.kind} (${violation.detail})`);
    this.name = "GovernorViolationError";
  }
}

export type GovernorStats = {
  forwarded: number;
  rejected: number;
  clamped: number;
  delayed: number;
};

const WINDOW_MS = 1000;
const defaultSleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

/**
 * Rate/velocity/region governor for any `InputActuator`. Stack it outside a
 * `SafeActuator` — `new GovernedActuator(new SafeActuator(os), …)` — so a
 * generated or corrupted episode can never spam events, teleport the pointer,
 * or click outside the intended window.
 *
 * Release events (`pointer.up`, `key.up`) are never rejected or delayed: a
 * governor must not be the reason a button stays held. They are still
 * clamped into the allowed region in clamp mode.
 */
export class GovernedActuator implements InputActuator {
  private readonly sent: number[] = [];
  private lastPointer: { point: Point; at: number } | undefined;
  private regions: ScreenRegion[];
  private readonly mode: "reject" | "clamp";
  private readonly now: () => number;
  private readonly sleep: (ms: number) => Promise<void>;
  private readonly counters: GovernorStats = { forwarded: 0, rejected: 0, clamped: 0, delayed: 0 };

  constructor(
    private readonly inner: InputActuator,
    private readonly options: GovernedActuatorOptions = {},
  ) {
    for (const [name, value] of [
      ["maxEventsPerSecond", options.maxEventsPerSecond],
      ["maxPointerSpeedPxPerMs", options.maxPointerSpeedPxPerMs],
    ] as const) {
      if (value !== undefined && !(value > 0)) {
        throw new Error(`${name} must be > 0`);
      }
    }
    for (const region of options.regions ?? []) {
      assertRegion(region);
    }
    this.regions = [...(options.regions ?? [])];
    this.mode = options.mode ?? "reject";
    this.now = options.now ?? (() => Date.now());
    this.sleep = options.sleep ?? defaultSleep;
  }

  stats(): GovernorStats {
    return { ...this.counters };
  }

  allowedRegions(): ScreenRegion[] {
    return this.regions.map((region) => ({ ...region }));
  }

  async perform(original: MovementEvent): Promise<void> {
    const release = original.type === "pointer.up" || original.type === "key.up";
    let event = original;

    if (!release) {
      await this.enforceRate(event);
    }

    if ("x" in event) {
      event = this.enforceRegion(event, release);
      if (!release) {
        event = this.enforceVelocity(event);
      }
    }

    await this.inner.perform(event);
    this.counters.forwarded += 1;
    const at = this.now();
    if (!release) {
      this.sent.push(at);
    }
    if ("x" in event) {
      this.lastPointer = { point: { x: event.x, y: event.y }, at };
    }
    if (event.type === "window.focus" && this.options.followFocus && event.bounds) {
      assertRegion(event.bounds);
      this.regions = [{ ...event.bounds }];
    }
  }

  private async enforceRate(event: MovementEvent): Promise<void> {
    const limit = this.options.maxEventsPerSecond;
    if (limit === undefined) return;
    this.prune();
    if (this.sent.length < limit) return;
    const waitMs = this.sent[0]! + WINDOW_MS - this.now();
    const detail = `${this.sent.length} events in last ${WINDOW_MS}ms (limit ${limit})`;
    if (this.mode === "reject") {
      this.violate({ kind: "rate", event, action: "rejected", detail });
    }
    this.record({ kind: "rate", event, action: "delayed", detail: `${detail}; waited ${waitMs}ms` });
    await this.sleep(Math.max(0, waitMs));
    this.prune();
    // A fake/coarse clock may not have advanced; drop the oldest so we never spin.
    while (this.sent.length >= limit) this.sent.shift();
  }

  private enforceRegion<E extends Extract<MovementEvent, { x: number }>>(event: E, release: boolean): E {
    if (this.regions.length === 0) return event;
    if (this.regions.some((region) => contains(region, event))) return event;
    const detail = `(${event.x}, ${event.y}) outside ${this.regions.length} allowed region(s)`;
    if (this.mode === "reject" && !release) {
      this.violate({ kind: "region", event, action: "rejected", detail });
    }
    const clamped = nearestPointInRegions(this.regions, event);
    this.record({ kind: "region", event, action: "clamped", detail });
    return { ...event, ...clamped };
  }

  private enforceVelocity<E extends Extract<MovementEvent, { x: number }>>(event: E): E {
    const maxSpeed = this.options.maxPointerSpeedPxPerMs;
    if (maxSpeed === undefined || !this.lastPointer) return event;
    const elapsed = Math.max(1, this.now() - this.lastPointer.at);
    const dx = event.x - this.lastPointer.point.x;
    const dy = event.y - this.lastPointer.point.y;
    const distance = Math.hypot(dx, dy);
    const maxDistance = maxSpeed * elapsed;
    if (distance <= maxDistance) return event;
    const detail = `${(distance / elapsed).toFixed(2)}px/ms > ${maxSpeed}px/ms`;
    if (this.mode === "reject") {
      this.violate({ kind: "velocity", event, action: "rejected", detail });
    }
    this.record({ kind: "velocity", event, action: "clamped", detail });
    const scale = maxDistance / distance;
    return {
      ...event,
      x: this.lastPointer.point.x + dx * scale,
      y: this.lastPointer.point.y + dy * scale,
    };
  }

  private prune(): void {
    const cutoff = this.now() - WINDOW_MS;
    while (this.sent.length > 0 && this.sent[0]! <= cutoff) this.sent.shift();
  }

  private record(violation: GovernorViolation): void {
    if (violation.action === "clamped") this.counters.clamped += 1;
    if (violation.action === "delayed") this.counters.delayed += 1;
    this.options.onViolation?.(violation);
  }

  private violate(violation: GovernorViolation): never {
    this.counters.rejected += 1;
    this.options.onViolation?.(violation);
    throw new GovernorViolationError(violation);
  }
}

function assertRegion(region: ScreenRegion): void {
  const values = [region.x, region.y, region.width, region.height];
  if (!values.every(Number.isFinite) || region.width <= 0 || region.height <= 0) {
    throw new Error("governor region must have finite x/y and positive width/height");
  }
}

/** Half-open on the far edges, matching pixel grids: [x, x+width). */
function contains(region: ScreenRegion, point: Point): boolean {
  return (
    point.x >= region.x &&
    point.y >= region.y &&
    point.x < region.x + region.width &&
    point.y < region.y + region.height
  );
}

function nearestPointInRegions(regions: ScreenRegion[], point: Point): Point {
  let best: Point = point;
  let bestDistance = Infinity;
  for (const region of regions) {
    const candidate = {
      x: Math.min(Math.max(point.x, region.x), region.x + region.width - 1),
      y: Math.min(Math.max(point.y, region.y), region.y + region.height - 1),
    };
    const distance = Math.hypot(candidate.x - point.x, candidate.y - point.y);
    if (distance < bestDistance) {
      best = candidate;
      bestDistance = distance;
    }
  }
  return best;
}
