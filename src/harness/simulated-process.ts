import type { IsProcessRunning, SpawnBackgroundProcess } from "./background-tasks.js";

/**
 * A deterministic, in-memory process backend for background tasks.
 *
 * The default background-task backend shells out to a launch script and tracks
 * real OS process groups. That is correct in production but non-deterministic in
 * tests (and unavailable in sandboxed/cloud environments): a `printf` task exits
 * before assertions run, so recovery reconciles it to `completed`/`failed` on a
 * timing-dependent basis. This backend replaces the OS process table with a
 * simulated one so callers control exactly which "processes" are alive.
 *
 * `spawnProcess` never launches anything — it allocates a synthetic pid and
 * records it as live. `isProcessRunning` reports liveness purely from that set,
 * so a pid the backend never handed out (e.g. a hand-written `999999` in a test)
 * is correctly reported as dead. This is also a documented seam for embedders
 * that supervise task processes out-of-band rather than via the launch script.
 */
export type SimulatedProcessBackend = {
  readonly spawnProcess: SpawnBackgroundProcess;
  readonly isProcessRunning: IsProcessRunning;
  /** Pids currently considered alive by this backend. */
  readonly livePids: ReadonlySet<number>;
  /** Mark a previously-spawned pid as terminated. */
  terminate(pid: number): void;
};

export type SimulatedProcessBackendOptions = {
  /** First synthetic pid to hand out. Defaults to a high, unlikely-to-collide value. */
  startPid?: number;
};

export function createSimulatedProcessBackend(
  options: SimulatedProcessBackendOptions = {},
): SimulatedProcessBackend {
  const live = new Set<number>();
  let nextPid = options.startPid ?? 900_001;

  const spawnProcess: SpawnBackgroundProcess = () => {
    const pid = nextPid;
    nextPid += 1;
    live.add(pid);
    return {
      pid,
      unref() {
        /* nothing to detach — no real process was spawned */
      },
    };
  };

  const isProcessRunning: IsProcessRunning = (pid) => live.has(pid);

  return {
    spawnProcess,
    isProcessRunning,
    livePids: live,
    terminate(pid: number): void {
      live.delete(pid);
    },
  };
}
