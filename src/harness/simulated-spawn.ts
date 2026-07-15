import type { SpawnBackgroundProcess } from "./background-tasks.js";

export type SimulatedBackgroundSpawnOptions = {
  /**
   * First synthetic pid handed out. Subsequent launches increment from here so
   * every simulated task gets a distinct, deterministic pid. Defaults to a high
   * value that will not collide with a real OS pid.
   */
  startPid?: number;
  /**
   * Optional hook invoked with the launch arguments each time a task is
   * "spawned". Lets callers assert on what would have been launched without any
   * real process being created.
   */
  onLaunch?: (command: string, args: string[]) => void;
};

/**
 * A deterministic, in-process stand-in for the real detached-process spawn used
 * by {@link BackgroundTaskExecutionService}. It performs NO real OS work: it
 * returns a synthetic, monotonically increasing pid and a no-op `unref`.
 *
 * The production launch path spawns a detached `bash` script that itself writes
 * the task's `state.json`/`output.log` (first "running", later "completed").
 * Any caller that instead simulates task lifecycle by writing those files
 * directly — hermetic tests, and a future dry-run/simulation mode — must NOT
 * also have a real subprocess racing those writes (a mid-write truncation makes
 * the state file momentarily unreadable). Injecting this spawn removes the real
 * subprocess entirely, so the simulated state is authoritative and stable.
 */
export function createSimulatedBackgroundSpawn(
  options: SimulatedBackgroundSpawnOptions = {},
): SpawnBackgroundProcess {
  let nextPid = options.startPid ?? 1_000_000;
  return (command, args) => {
    options.onLaunch?.(command, args);
    const pid = nextPid;
    nextPid += 1;
    return { pid, unref() {} };
  };
}
