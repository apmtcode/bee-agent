import type { IsProcessRunning, SpawnBackgroundProcess } from "../harness/background-tasks.js";

/**
 * A deterministic, in-memory background-process backend.
 *
 * The production background-task launcher spawns a real detached OS process
 * whose shell launch-script writes execution-state files asynchronously. That
 * is exactly right in production, but it makes tests non-hermetic: the health
 * checks and reconciliation logic race against the real process settling, so
 * the same test can report a task as `running`, `completed`, or
 * `missing-process` depending purely on machine timing.
 *
 * `createSimulatedBackgroundSpawn` returns a `spawn` that starts no real
 * process and never touches the filesystem — it just hands back a stable,
 * unique fake pid and tracks which pids are "alive". Paired with the returned
 * `isProcessRunning`, callers get fully deterministic liveness: pids the
 * simulator handed out are alive, every other pid (e.g. a synthetic dead pid a
 * test writes into a state file) is not. Tests that want every task to look
 * dead can ignore the returned `isProcessRunning` and inject `() => false`.
 */
export type SimulatedBackgroundSpawn = {
  spawn: SpawnBackgroundProcess;
  isProcessRunning: IsProcessRunning;
  /** Pids currently considered alive by this simulator. */
  readonly alive: ReadonlySet<number>;
  /** Force-terminate a simulated pid so `isProcessRunning` reports it dead. */
  terminate(pid: number): void;
};

export function createSimulatedBackgroundSpawn(
  options: { startPid?: number } = {},
): SimulatedBackgroundSpawn {
  const alive = new Set<number>();
  let nextPid = options.startPid ?? 424242;

  const spawn: SpawnBackgroundProcess = () => {
    const pid = nextPid;
    nextPid += 1;
    alive.add(pid);
    return {
      pid,
      unref() {
        // No underlying handle to detach — the simulated process is inert.
      },
    };
  };

  const isProcessRunning: IsProcessRunning = (pid) => alive.has(pid);

  return {
    spawn,
    isProcessRunning,
    alive,
    terminate(pid) {
      alive.delete(pid);
    },
  };
}
