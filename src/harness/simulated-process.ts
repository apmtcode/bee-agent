import type { IsProcessRunning, SpawnBackgroundProcess } from "./background-tasks.js";

/**
 * A deterministic, OS-free background-process backend.
 *
 * The real {@link SpawnBackgroundProcess} launches a detached shell that runs a
 * generated launch script; that script writes the task's state/output files and
 * keeps a live OS process. That behaviour is inherently non-deterministic in a
 * sandboxed/cloud environment (processes exit on their own schedule, PIDs are
 * reused, `process.kill` targets real process groups) and it races with tests
 * that simulate execution by writing state files by hand.
 *
 * This simulator returns a stable fake PID for every launch and never touches
 * the OS: no child process is started, so no state/output files are written
 * behind the caller's back. Callers stay in full control of the on-disk task
 * state, which makes runtime/control-plane behaviour reproducible in tests and
 * usable for dry-run/offline modes.
 */
export interface SimulatedLaunch {
  /** Absolute path of the launch script the runtime asked to execute. */
  readonly command: string;
  /** The fake PID handed back for this launch. */
  readonly pid: number;
}

export interface SimulatedBackgroundProcessOptions {
  /**
   * Whether PIDs produced by this backend report as running via
   * {@link SimulatedBackgroundProcess.isProcessRunning}. Defaults to `true`.
   * Set to `false` to exercise missing-process recovery paths deterministically.
   */
  alive?: boolean;
  /**
   * Base value for generated PIDs. Defaults to a value far above any real
   * Linux `pid_max` so that if runtime code falls through to a real
   * `process.kill(-pid, …)` it reliably yields `ESRCH` instead of signalling a
   * live process group.
   */
  basePid?: number;
}

export interface SimulatedBackgroundProcess {
  /** Drop-in replacement for the runtime's `backgroundTaskSpawnProcess`. */
  readonly spawn: SpawnBackgroundProcess;
  /** Drop-in replacement for the runtime's `backgroundTaskIsProcessRunning`. */
  readonly isProcessRunning: IsProcessRunning;
  /** Every launch this backend serviced, in order, for assertions. */
  readonly launches: readonly SimulatedLaunch[];
}

const DEFAULT_BASE_PID = 900_000_000;

/**
 * Create a simulated background-process backend. Wire the returned `spawn` and
 * `isProcessRunning` into a runtime (or `OperatorCliApp`) to keep background
 * task orchestration fully in-memory and deterministic.
 */
export function createSimulatedBackgroundProcess(
  options: SimulatedBackgroundProcessOptions = {},
): SimulatedBackgroundProcess {
  const alive = options.alive ?? true;
  const basePid = options.basePid ?? DEFAULT_BASE_PID;
  const launches: SimulatedLaunch[] = [];
  const knownPids = new Set<number>();
  let counter = 0;

  const spawn: SpawnBackgroundProcess = (command) => {
    counter += 1;
    const pid = basePid + counter;
    knownPids.add(pid);
    launches.push({ command, pid });
    return {
      pid,
      unref() {
        /* no-op: nothing to detach */
      },
    };
  };

  const isProcessRunning: IsProcessRunning = (pid) => {
    if (!knownPids.has(pid)) {
      return false;
    }
    return alive;
  };

  return {
    spawn,
    isProcessRunning,
    launches,
  };
}
