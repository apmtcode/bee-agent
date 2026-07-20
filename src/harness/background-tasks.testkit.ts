import type { SpawnBackgroundProcess } from "./background-tasks.js";

/**
 * Deterministic background-task spawn for tests.
 *
 * The production launch script writes the task's `state.json` via a non-atomic
 * shell redirect (`printf … | sed … > state.json`) from a detached process.
 * When a test spawns a real launch script and then immediately inspects task
 * health, two races appear:
 *
 *  1. **Torn reads** — the reader can observe a half-written `state.json` and
 *     throw a JSON `SyntaxError`.
 *  2. **State-exists timing** — whether the launch script has written the
 *     `running` state yet flips derived health between `active` and
 *     `degraded`/`missing-process`.
 *
 * This factory returns a spawn that never runs a real process: it hands back a
 * stable fake pid and writes nothing, so tests that set up state explicitly via
 * `executionService.writeState(...)` are fully deterministic. Combine with
 * `backgroundTaskIsProcessRunning: () => false` (or `true`) to pin liveness.
 */
export function createDeterministicBackgroundSpawn(pid = 424242): SpawnBackgroundProcess {
  return () => ({
    pid,
    unref() {
      /* no-op: nothing is actually running */
    },
  });
}
