import type { IsProcessRunning, SpawnBackgroundProcess } from "./background-tasks.js";

/**
 * Deterministic, in-process replacement for the background-task OS seams
 * (`spawnProcess` + `isProcessRunning`).
 *
 * The production launch path spawns a real detached child whose shell wrapper
 * writes the execution-state file *asynchronously*. In cloud/CI runs that async
 * write races with everything that reads the state (diagnostics, `sync`,
 * `stop`) and with any test that writes its own state — producing flaky
 * active/degraded flips and, under concurrency, half-written state JSON. Tests
 * that only care about the task *record* lifecycle (not real command output)
 * should inject this simulator so there is no real process and no competing
 * writer.
 *
 * `spawnProcess` returns a fake child with a monotonic pid and never writes a
 * state file, so `readState` stays `undefined` until the store itself
 * reconciles or the test writes state explicitly. `isProcessRunning` reports
 * liveness from an in-memory pid set the test fully controls, so a task can be
 * made to look alive or crashed deterministically via {@link markStopped} /
 * {@link markRunning}.
 */
export interface SimulatedBackgroundExecution {
  /** Injectable `backgroundTaskSpawnProcess` — spawns no real process. */
  readonly spawnProcess: SpawnBackgroundProcess;
  /** Injectable `backgroundTaskIsProcessRunning` — reads {@link livePids}. */
  readonly isProcessRunning: IsProcessRunning;
  /** Pids currently considered alive. Mutated by spawn / mark* helpers. */
  readonly livePids: ReadonlySet<number>;
  /** Simulate a process exiting (its pid stops being "running"). */
  markStopped(pid: number): void;
  /** Simulate a process (re)starting for a known pid. */
  markRunning(pid: number): void;
}

export interface SimulatedBackgroundExecutionOptions {
  /** First pid handed out by `spawnProcess` (monotonic from here). */
  startPid?: number;
  /**
   * Whether a freshly spawned pid is considered alive. `true` (default) models
   * a healthy long-running task; `false` models a task whose process is already
   * gone (so diagnostics can exercise the missing-process path deterministically).
   */
  aliveOnSpawn?: boolean;
}

/**
 * Build a deterministic {@link SimulatedBackgroundExecution}. Wire the returned
 * `spawnProcess` / `isProcessRunning` into `StandaloneOperatorRuntime`
 * (`backgroundTaskSpawnProcess` / `backgroundTaskIsProcessRunning`) or
 * `OperatorCliApp` so background-task tests never touch the real OS.
 */
export function createSimulatedBackgroundExecution(
  options: SimulatedBackgroundExecutionOptions = {},
): SimulatedBackgroundExecution {
  const aliveOnSpawn = options.aliveOnSpawn ?? true;
  const live = new Set<number>();
  let nextPid = options.startPid ?? 100_000;

  const spawnProcess: SpawnBackgroundProcess = () => {
    const pid = nextPid;
    nextPid += 1;
    if (aliveOnSpawn) {
      live.add(pid);
    }
    return {
      pid,
      unref() {
        /* no-op: nothing to detach from */
      },
    };
  };

  const isProcessRunning: IsProcessRunning = (pid) => live.has(pid);

  return {
    spawnProcess,
    isProcessRunning,
    livePids: live,
    markStopped(pid) {
      live.delete(pid);
    },
    markRunning(pid) {
      live.add(pid);
    },
  };
}
