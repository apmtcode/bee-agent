import type { IsProcessRunning, SpawnBackgroundProcess } from "./background-tasks.js";

/**
 * Deterministic stand-ins for the real OS process spawner + liveness probe used
 * by {@link FileBackgroundTaskStore}. They never launch a child process, so a
 * test that also writes execution state by hand does not race the launch
 * script's own asynchronous state writes (the launch script writes `running`
 * then `completed`/`failed` in the background, which otherwise clobbers a
 * manually written state file at an unpredictable moment).
 *
 * These helpers live in `src/` (not a `*.test.ts` file) only so multiple test
 * suites can share them; they have no production call sites.
 */

/**
 * A spawner that hands out monotonically increasing fake PIDs and never touches
 * the OS. Pair it with an explicit `isProcessRunning` (commonly `() => false`)
 * when the test writes its own execution states and wants a fixed liveness
 * answer.
 */
export function createStubBackgroundSpawn(startPid = 100000): SpawnBackgroundProcess {
  let pid = startPid;
  return () => {
    pid += 1;
    return { pid, unref() {} };
  };
}

/**
 * A paired spawner + liveness probe where a PID is reported "running" iff this
 * spawner issued it. A task launched through {@link control.spawn} therefore
 * reads as alive, while any sentinel PID a test writes by hand (one this stub
 * never issued) reads as dead — reproducing "some tasks live, some died"
 * without depending on real process timing.
 */
export function createStubBackgroundProcessControl(startPid = 100000): {
  spawn: SpawnBackgroundProcess;
  isProcessRunning: IsProcessRunning;
} {
  const live = new Set<number>();
  let pid = startPid;
  return {
    spawn: () => {
      pid += 1;
      live.add(pid);
      return { pid, unref() {} };
    },
    isProcessRunning: (candidate) => live.has(candidate),
  };
}
