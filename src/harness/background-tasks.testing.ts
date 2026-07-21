import type { IsProcessRunning, SpawnBackgroundProcess } from "./background-tasks.js";

/**
 * A deterministic, side-effect-free stand-in for the real OS process spawner
 * used by {@link BackgroundTaskExecutionService}. It never launches a real
 * process and never writes execution-state or output files, so tests keep full
 * control over a task's persisted state via `writeState`/`writeOutput`.
 *
 * The real launcher spawns a detached bash+python wrapper that writes the state
 * file asynchronously and non-atomically; in a cloud/CI sandbox that races the
 * test's own expectations (and depends on `bash`/`python3`/`date` being present
 * and fast). Injecting this spawner removes that non-determinism entirely.
 *
 * Each call returns a distinct, stable fake pid and records it in
 * {@link InertBackgroundSpawn.runningPids}. The paired {@link
 * InertBackgroundSpawn.isProcessRunning} reports those launched pids as alive
 * while any unknown pid (e.g. a hand-written `pid: 999999` used to simulate a
 * dead process) reads as not running — so a test can exercise both the healthy
 * and the missing-process reconciliation paths deterministically.
 */
export interface InertBackgroundSpawn {
  /** Drop-in replacement for the runtime's `backgroundTaskSpawnProcess`. */
  readonly spawn: SpawnBackgroundProcess;
  /** Drop-in replacement for the runtime's `backgroundTaskIsProcessRunning`. */
  readonly isProcessRunning: IsProcessRunning;
  /** The set of pids this spawner has handed out (treated as alive). */
  readonly runningPids: ReadonlySet<number>;
}

/**
 * Create an {@link InertBackgroundSpawn}. `startPid` seeds the first fake pid so
 * a suite can keep the fake range clear of any pid it hand-writes.
 */
export function createInertBackgroundSpawn(startPid = 100000): InertBackgroundSpawn {
  const runningPids = new Set<number>();
  let nextPid = startPid;
  const spawn: SpawnBackgroundProcess = () => {
    const pid = nextPid++;
    runningPids.add(pid);
    return {
      pid,
      unref() {},
    };
  };
  const isProcessRunning: IsProcessRunning = (pid) => runningPids.has(pid);
  return { spawn, isProcessRunning, runningPids };
}
