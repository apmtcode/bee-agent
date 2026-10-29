import type { KeyModifier, MovementEvent, PointerButton } from "./events.js";
import type { InputActuator } from "./replay.js";

export type SafeActuatorReleaseReason = "abort" | "idle-timeout" | "signal" | "error" | "dispose" | "manual";

export type SafeActuatorReleaseReport = {
  reason: SafeActuatorReleaseReason;
  /** The synthetic release events that were performed, in order. */
  released: MovementEvent[];
  /** Release events the inner actuator threw on (still attempted, never fatal). */
  failed: MovementEvent[];
};

/** Minimal process-like emitter so signal handling can be tested without real signals. */
export type SignalTarget = {
  on(signal: NodeJS.Signals, listener: () => void): unknown;
  off(signal: NodeJS.Signals, listener: () => void): unknown;
};

export type SafeActuatorOptions = {
  /**
   * Auto-release everything if no event is performed for this long while any
   * button/key is held. Guards against a hung caller leaving a drag pressed.
   * 0 or undefined disables the watchdog.
   */
  idleTimeoutMs?: number;
  /** Releasing on abort covers callers that stop calling `perform` mid-gesture. */
  signal?: AbortSignal;
  setTimer?: (fn: () => void, ms: number) => unknown;
  clearTimer?: (handle: unknown) => void;
  onRelease?: (report: SafeActuatorReleaseReport) => void;
};

export type HeldInputs = {
  buttons: PointerButton[];
  keys: string[];
};

/**
 * Wraps any `InputActuator` and guarantees no pointer button or key is left
 * held on the real machine. It tracks held inputs across every caller (not
 * just one replay), and releases them on abort, idle timeout, process signal,
 * an error inside `guard()`, or `dispose()`. After disposal it refuses input,
 * so a late async caller cannot re-press anything.
 */
export class SafeActuator implements InputActuator {
  private readonly buttons = new Map<PointerButton, true>();
  private readonly keys = new Map<string, KeyModifier[] | undefined>();
  private lastPointer = { x: 0, y: 0 };
  private lastT = 0;
  private timer: unknown;
  private disposed = false;
  private releasing: Promise<SafeActuatorReleaseReport> | undefined;
  private readonly setTimer: (fn: () => void, ms: number) => unknown;
  private readonly clearTimer: (handle: unknown) => void;
  private readonly abortListener?: () => void;

  constructor(
    private readonly inner: InputActuator,
    private readonly options: SafeActuatorOptions = {},
  ) {
    this.setTimer = options.setTimer ?? ((fn, ms) => setTimeout(fn, ms));
    this.clearTimer = options.clearTimer ?? ((handle) => clearTimeout(handle as ReturnType<typeof setTimeout>));
    if (options.signal) {
      this.abortListener = () => void this.releaseAll("abort");
      options.signal.addEventListener("abort", this.abortListener, { once: true });
    }
  }

  get isDisposed(): boolean {
    return this.disposed;
  }

  held(): HeldInputs {
    return { buttons: [...this.buttons.keys()], keys: [...this.keys.keys()] };
  }

  async perform(event: MovementEvent): Promise<void> {
    if (this.disposed) {
      throw new Error("SafeActuator is disposed; refusing input");
    }
    if (this.options.signal?.aborted && isPress(event)) {
      throw new Error("SafeActuator signal aborted; refusing new presses");
    }
    if (this.releasing) {
      await this.releasing;
    }
    this.lastT = Math.max(this.lastT, event.t);
    if ("x" in event) {
      this.lastPointer = { x: event.x, y: event.y };
    }
    // Track before performing: if the inner actuator throws after the OS saw
    // the press, we must still consider it held and release it.
    if (event.type === "pointer.down") this.buttons.set(event.button, true);
    if (event.type === "key.down") this.keys.set(event.key, event.modifiers);
    await this.inner.perform(event);
    if (event.type === "pointer.up") this.buttons.delete(event.button);
    if (event.type === "key.up") this.keys.delete(event.key);
    this.armWatchdog();
  }

  /** Releases every held input. Concurrent calls share one release pass. */
  releaseAll(reason: SafeActuatorReleaseReason = "manual"): Promise<SafeActuatorReleaseReport> {
    if (!this.releasing) {
      this.releasing = this.doRelease(reason).finally(() => {
        this.releasing = undefined;
      });
    }
    return this.releasing;
  }

  /** Runs `fn`; on throw, releases everything (reason "error") then rethrows. */
  async guard<T>(fn: (actuator: SafeActuator) => Promise<T> | T): Promise<T> {
    try {
      return await fn(this);
    } catch (error) {
      await this.releaseAll("error");
      throw error;
    }
  }

  /**
   * Releases held input on SIGINT/SIGTERM. Handlers only release; they don't
   * exit, so the host's own shutdown logic still runs. Returns a detach fn.
   */
  installSignalGuard(
    target: SignalTarget = process,
    signals: NodeJS.Signals[] = ["SIGINT", "SIGTERM"],
  ): () => void {
    const listener = () => void this.releaseAll("signal");
    for (const signal of signals) target.on(signal, listener);
    return () => {
      for (const signal of signals) target.off(signal, listener);
    };
  }

  async dispose(): Promise<SafeActuatorReleaseReport> {
    const report = await this.releaseAll("dispose");
    this.disposed = true;
    this.cancelWatchdog();
    if (this.abortListener) {
      this.options.signal?.removeEventListener("abort", this.abortListener);
    }
    return report;
  }

  private async doRelease(reason: SafeActuatorReleaseReason): Promise<SafeActuatorReleaseReport> {
    this.cancelWatchdog();
    const pending: MovementEvent[] = [
      ...[...this.buttons.keys()].map(
        (button): MovementEvent => ({ type: "pointer.up", t: this.lastT, ...this.lastPointer, button }),
      ),
      ...[...this.keys.entries()].map(
        ([key, modifiers]): MovementEvent => ({ type: "key.up", t: this.lastT, key, ...(modifiers ? { modifiers } : {}) }),
      ),
    ];
    this.buttons.clear();
    this.keys.clear();
    const released: MovementEvent[] = [];
    const failed: MovementEvent[] = [];
    for (const event of pending) {
      try {
        await this.inner.perform(event);
        released.push(event);
      } catch {
        failed.push(event);
      }
    }
    const report = { reason, released, failed };
    if (pending.length > 0) {
      this.options.onRelease?.(report);
    }
    return report;
  }

  private armWatchdog(): void {
    this.cancelWatchdog();
    const ms = this.options.idleTimeoutMs ?? 0;
    if (ms > 0 && (this.buttons.size > 0 || this.keys.size > 0)) {
      this.timer = this.setTimer(() => {
        this.timer = undefined;
        void this.releaseAll("idle-timeout");
      }, ms);
    }
  }

  private cancelWatchdog(): void {
    if (this.timer !== undefined) {
      this.clearTimer(this.timer);
      this.timer = undefined;
    }
  }
}

function isPress(event: MovementEvent): boolean {
  return event.type === "pointer.down" || event.type === "key.down";
}
