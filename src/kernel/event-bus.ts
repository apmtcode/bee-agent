export type OperatorEvent = {
  type: string;
  payload?: unknown;
  ts?: number;
};

export type OperatorEventFilter<T extends OperatorEvent = OperatorEvent> = (event: T) => boolean;

export type OperatorEventIteratorOptions = {
  replay?: boolean;
};

export class OperatorEventBus<T extends OperatorEvent = OperatorEvent> {
  private readonly replayLimit: number;
  private readonly replayEvents: T[] = [];
  private readonly listeners = new Set<(event: T) => void>();
  private readonly waiters = new Set<() => void>();
  private closed = false;
  private lastTs = 0;

  constructor(options: { replayLimit?: number } = {}) {
    this.replayLimit = options.replayLimit ?? 0;
  }

  publish(event: T): void {
    if (this.closed) {
      return;
    }
    if (typeof event.ts === "number") {
      // Enforce a strictly monotonic event clock so no two events ever share a
      // timestamp. Reconnect/replay cursors filter with `ts > afterTs`, which
      // silently drops a later event whose `Date.now()` collides with the
      // cursor's millisecond on a fast machine (a real, load-dependent flake).
      // Bumping a colliding timestamp to `lastTs + 1` keeps `ts` unique and
      // strictly increasing without any protocol change.
      const ts = event.ts > this.lastTs ? event.ts : this.lastTs + 1;
      this.lastTs = ts;
      if (ts !== event.ts) {
        event.ts = ts;
      }
    }
    if (this.replayLimit > 0) {
      this.replayEvents.push(event);
      const overflow = this.replayEvents.length - this.replayLimit;
      if (overflow > 0) {
        this.replayEvents.splice(0, overflow);
      }
    }
    for (const listener of this.listeners) {
      listener(event);
    }
  }

  snapshot(filter?: OperatorEventFilter<T>): T[] {
    return filter ? this.replayEvents.filter(filter) : [...this.replayEvents];
  }

  close(): void {
    this.closed = true;
    this.replayEvents.length = 0;
    this.listeners.clear();
    for (const wake of this.waiters) {
      wake();
    }
    this.waiters.clear();
  }

  stream(
    filter?: OperatorEventFilter<T>,
    options: OperatorEventIteratorOptions = {},
  ): AsyncIterable<T> {
    return {
      [Symbol.asyncIterator]: (): AsyncIterator<T> => {
        const queue: T[] = options.replay ? this.snapshot(filter) : [];
        let stopped = false;
        let wake: (() => void) | null = null;

        const wakePending = () => {
          if (!wake) {
            return;
          }
          const current = wake;
          wake = null;
          this.waiters.delete(current);
          current();
        };

        const listener = (event: T) => {
          if (!filter || filter(event)) {
            queue.push(event);
            wakePending();
          }
        };

        const cleanup = () => {
          if (stopped) {
            return;
          }
          stopped = true;
          this.listeners.delete(listener);
          wakePending();
        };

        this.listeners.add(listener);

        return {
          next: async (): Promise<IteratorResult<T>> => {
            while (true) {
              if (stopped) {
                break;
              }
              if (queue.length > 0) {
                return { done: false, value: queue.shift() as T };
              }
              if (this.closed) {
                break;
              }
              await new Promise<void>((resolve) => {
                const wakeCurrent = () => {
                  if (wake === wakeCurrent) {
                    wake = null;
                  }
                  this.waiters.delete(wakeCurrent);
                  resolve();
                };
                wake = wakeCurrent;
                this.waiters.add(wakeCurrent);
              });
            }
            cleanup();
            return { done: true, value: undefined as never };
          },
          return: async (): Promise<IteratorResult<T>> => {
            cleanup();
            return { done: true, value: undefined as never };
          },
        };
      },
    };
  }
}
