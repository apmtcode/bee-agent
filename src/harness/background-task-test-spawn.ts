import type { SpawnBackgroundProcess } from "./background-tasks.js";

/**
 * A deterministic {@link SpawnBackgroundProcess} for tests.
 *
 * The real spawn runs the generated launch script, which asynchronously writes
 * the task's `state.json` (initial "running", then terminal "completed"/
 * "failed"). Tests that drive recovery/sync scenarios write those state files
 * themselves to model an exact situation — but with a real spawn the launch
 * script races those manual writes and clobbers them, making assertions flaky.
 *
 * This factory returns a spawn that hands back a stable, fake pid and runs
 * nothing, so the launch script never executes and the test's manually written
 * state is the single source of truth. Each returned pid is unique (starting at
 * `basePid`) so distinct tasks never collide.
 */
export function createDeterministicBackgroundSpawn(basePid = 424242): SpawnBackgroundProcess {
  let nextPid = basePid;
  return () => {
    const pid = nextPid;
    nextPid += 1;
    return {
      pid,
      unref() {
        /* no-op: nothing to detach in the mock */
      },
    };
  };
}
