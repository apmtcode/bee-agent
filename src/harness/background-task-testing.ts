import type { IsProcessRunning, SpawnBackgroundProcess } from "./background-tasks.js";

/**
 * Deterministic, in-memory stand-in for the OS process surface that backs
 * {@link BackgroundTaskExecutionService}. Tests (and any simulated/cloud run
 * without a real desktop) inject this so background-task code never spawns a
 * real detached subprocess.
 *
 * Why this exists: the production launch path spawns a detached shell whose
 * launch script asynchronously writes the task's state file. In tests that
 * race against the test's own `writeState` calls and against real PID liveness
 * checks, producing flaky pass/fail. Routing `spawnProcess` + `isProcessRunning`
 * through this double removes both sources of nondeterminism while exercising
 * the exact same code paths.
 *
 * Fake PIDs start far above any plausible OS pid (well past Linux `pid_max`), so
 * the production `stop()` path — which calls `process.kill(-pid, signal)`
 * directly — reliably gets `ESRCH` and never signals a real process group.
 */
export class InMemoryBackgroundProcesses {
  /** Base well above Linux's default pid_max (4_194_304) → kill(-pid) === ESRCH. */
  private nextPid = 2_000_000_000;
  private readonly alive = new Map<number, boolean>();

  /** Drop-in for `backgroundTaskSpawnProcess`: records a live fake PID, runs nothing. */
  readonly spawnProcess: SpawnBackgroundProcess = () => {
    const pid = this.nextPid++;
    this.alive.set(pid, true);
    return { pid, unref: () => {} };
  };

  /** Drop-in for `backgroundTaskIsProcessRunning`: consults the in-memory table. */
  readonly isProcessRunning: IsProcessRunning = (pid: number): boolean => this.alive.get(pid) ?? false;

  /** Simulate a tracked process exiting (subsequent liveness checks return false). */
  kill(pid: number): void {
    if (this.alive.has(pid)) {
      this.alive.set(pid, false);
    }
  }

  /** Simulate every tracked process exiting at once. */
  killAll(): void {
    for (const pid of this.alive.keys()) {
      this.alive.set(pid, false);
    }
  }

  /** All PIDs this double has ever handed out, in spawn order. */
  pids(): number[] {
    return [...this.alive.keys()];
  }
}

export function createInMemoryBackgroundProcesses(): InMemoryBackgroundProcesses {
  return new InMemoryBackgroundProcesses();
}
