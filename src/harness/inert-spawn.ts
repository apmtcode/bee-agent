import type { SpawnBackgroundProcess } from "./background-tasks.js";

/**
 * A background-task spawn that launches nothing. Returns a stable synthetic pid
 * and a no-op `unref`, so callers that only need a handle (and manage state
 * themselves) never start a real detached OS process.
 *
 * Intended for tests and dry-run/simulated environments: the production default
 * spawns a real `bash` launch script, which — when combined with a test that
 * also writes state files directly — races non-deterministically over the same
 * `state.json`. Injecting this makes such flows hermetic and deterministic.
 */
export function createInertBackgroundSpawn(pid = 1_000_000): SpawnBackgroundProcess {
  return () => ({
    pid,
    unref() {
      /* no-op: nothing to detach */
    },
  });
}
