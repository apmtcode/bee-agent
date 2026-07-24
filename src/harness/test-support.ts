import type { SpawnBackgroundProcess } from "./background-tasks.js";

/**
 * A deterministic no-op replacement for the background-task spawner used in
 * tests. It returns an incrementing fake pid without launching any real OS
 * process, so the generated launch script never runs and therefore never races
 * the test's own `writeState` calls on the same execution-state file.
 *
 * Production code always uses the real `spawn`; this seam exists purely so
 * background-task lifecycle tests stay hermetic and deterministic in CI/cloud
 * environments where a real detached process would write state concurrently.
 */
export function makeNoopSpawn(startPid = 100000): SpawnBackgroundProcess {
  let nextPid = startPid;
  return () => {
    const pid = nextPid;
    nextPid += 1;
    return {
      pid,
      unref() {
        /* nothing to release for a process that was never spawned */
      },
    };
  };
}
