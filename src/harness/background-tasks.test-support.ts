import type { SpawnBackgroundProcess } from "./background-tasks.js";

/**
 * Test-only spawn that simulates launching a detached background process
 * WITHOUT actually running one. It hands back a unique, monotonically
 * increasing fake pid and a no-op `unref`, and crucially never executes the
 * launch script — so it writes no execution-state file.
 *
 * This makes reconcile/recovery tests hermetic and deterministic: the only
 * execution state that exists is what the test (or the runtime) writes
 * explicitly. Without it, a real detached process races the test — its
 * launch script may or may not have written `state.json` by the time an
 * assertion runs, so `isProcessRunning`-driven diagnostics flip between
 * "running"/"active" and "missing-process"/"degraded" depending purely on
 * process-startup timing. Pair this with `backgroundTaskIsProcessRunning`
 * to drive liveness deterministically.
 */
export function createInertBackgroundSpawn(startPid = 40000): SpawnBackgroundProcess {
  let nextPid = startPid;
  return () => {
    nextPid += 1;
    return {
      pid: nextPid,
      unref() {
        /* no-op: nothing to detach */
      },
    };
  };
}
