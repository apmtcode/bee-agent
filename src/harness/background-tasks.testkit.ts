import type { SpawnBackgroundProcess } from "./background-tasks.js";

/**
 * Test-only helpers for the background-task subsystem.
 *
 * These are intentionally excluded from the production build (only referenced by
 * `*.test.ts` files) and exist so tests can exercise the task/reconcile/breaker
 * logic deterministically, without spawning real OS processes.
 */

/**
 * A {@link SpawnBackgroundProcess} that launches nothing. It returns a
 * plausible, monotonically increasing fake pid and a no-op `unref()`, so the
 * store's bookkeeping (markStarted → status "running", processId tracking)
 * behaves exactly as with a real spawn — but no detached shell/python runs, so
 * nothing races with a test's manually-written execution state.
 *
 * Pair it with `backgroundTaskIsProcessRunning: () => false` to model tasks whose
 * process is not alive: without a real launcher writing its own "running" state,
 * the only state on disk is what the test writes, making reconcile/breaker
 * assertions fully deterministic.
 */
export function createInertBackgroundSpawn(startPid = 100000): SpawnBackgroundProcess {
  let nextPid = startPid;
  return () => {
    const pid = nextPid;
    nextPid += 1;
    return {
      pid,
      unref() {
        // no-op: nothing was actually spawned
      },
    };
  };
}
