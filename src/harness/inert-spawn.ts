import type { SpawnBackgroundProcess } from "./background-tasks.js";

/**
 * Builds a {@link SpawnBackgroundProcess} that allocates a fake, monotonically
 * increasing PID and starts **no real OS process**.
 *
 * Two use cases:
 *
 *  - **Deterministic tests.** A real detached launch script writes the task's
 *    state and output files asynchronously, which races against a test that
 *    also writes those files (or asserts on `control` health before the script
 *    has run). An inert spawn removes the real process entirely, so the only
 *    writes are the ones the test makes — no timing races, no leaked `sleep`
 *    processes lingering after the suite exits.
 *  - **Dry-run execution.** Callers that need a background-task record without
 *    actually executing anything (previews, plan rehearsals) can inject this to
 *    get a well-formed "running" record whose process never exists.
 *
 * PIDs start high to avoid colliding with a real process id; combine with an
 * `isProcessRunning: () => false` seam when a test needs the task to read as
 * "process gone".
 */
export function createInertSpawn(startPid = 900_000): SpawnBackgroundProcess {
  let next = startPid;
  return () => ({ pid: next++, unref() {} });
}
