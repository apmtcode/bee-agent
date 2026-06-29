import type { SpawnBackgroundProcess } from "./background-tasks.js";

/**
 * An inert spawn for tests: it never launches a real OS subprocess, so it never
 * writes a background-task execution state file. Tests that drive execution
 * state explicitly (via `executionService.writeState(...)`) get fully
 * deterministic behaviour — there is no concurrently-running real subprocess
 * racing to write/overwrite the same state file. Each call returns a stable,
 * non-zero fake pid so `launch()` succeeds without spawning anything.
 *
 * Use this whenever a test asserts on precise execution-state transitions
 * (recovery, circuit-breaker counts, missing-process detection) rather than on
 * real subprocess output. For tests that genuinely need real subprocess output,
 * keep the default (real) spawn.
 */
export function createInertBackgroundSpawn(startPid = 424200): SpawnBackgroundProcess {
  let pid = startPid;
  return () => {
    pid += 1;
    return { pid, unref() {} };
  };
}
