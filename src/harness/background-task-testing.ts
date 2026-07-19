import type { SpawnBackgroundProcess } from "./background-tasks.js";

/**
 * Creates an inert {@link SpawnBackgroundProcess} that launches **no** real OS
 * process.
 *
 * It hands back a stable, incrementing fake pid and a no-op `unref`, and —
 * crucially — writes nothing to the task's state or output files. Tests and
 * simulations that drive the background-task lifecycle can therefore own the
 * execution state deterministically (via `writeState` / `writeOutput`) with no
 * races against a real child process's own, non-atomic state writes.
 *
 * Without this, exercising `startBackgroundTask` spawns a real shell that
 * concurrently rewrites `state.json`, which makes any subsequent `readState`
 * both timing-dependent (running vs. completed) and prone to torn reads. Inject
 * this factory in cloud/CI where a hermetic, reproducible lifecycle is required.
 *
 * @param startPid first fake pid to hand out; each call increments from here.
 */
export function createInertBackgroundTaskSpawn(startPid = 100_000): SpawnBackgroundProcess {
  let nextPid = startPid;
  return () => {
    const pid = nextPid;
    nextPid += 1;
    return {
      pid,
      unref() {
        /* no child to detach */
      },
    };
  };
}
