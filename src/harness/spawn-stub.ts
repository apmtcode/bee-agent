import type { SpawnBackgroundProcess } from "./background-tasks.js";

/**
 * A record of a single simulated background-process launch. Captured by
 * {@link createInMemoryBackgroundSpawn} so callers can assert on what would
 * have been spawned without a real OS process ever being created.
 */
export type RecordedSpawn = {
  pid: number;
  command: string;
  args: string[];
  cwd: string;
};

export type InMemoryBackgroundSpawn = SpawnBackgroundProcess & {
  /** Every launch this backend has served, in order. */
  readonly launches: readonly RecordedSpawn[];
};

/**
 * Build a deterministic, side-effect-free {@link SpawnBackgroundProcess}.
 *
 * Instead of forking a detached child process (which races the caller by
 * writing state/output files on its own schedule), this backend hands back a
 * stable, monotonically increasing pid and records the launch. It lets the
 * background-task bookkeeping (start → markStarted → sync → recover → cancel)
 * be exercised end-to-end in the cloud/CI, where launching real processes makes
 * tests flaky and leaves orphaned children behind.
 *
 * @param startPid First pid to hand out (defaults to a high, unmistakably
 *   synthetic value so it never collides with a real OS pid in assertions).
 */
export function createInMemoryBackgroundSpawn(startPid = 900000): InMemoryBackgroundSpawn {
  const launches: RecordedSpawn[] = [];
  let nextPid = startPid;

  const spawn: SpawnBackgroundProcess = (command, args, options) => {
    const pid = nextPid;
    nextPid += 1;
    launches.push({ pid, command, args: [...args], cwd: options.cwd });
    return {
      pid,
      unref() {
        /* no-op: nothing to detach for an in-memory launch */
      },
    };
  };

  const backend = spawn as InMemoryBackgroundSpawn;

  Object.defineProperty(backend, "launches", {
    get: () => launches,
    enumerable: true,
  });

  return backend;
}
