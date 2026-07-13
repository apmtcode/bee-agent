import type { SpawnBackgroundProcess } from "./background-tasks.js";

/**
 * A single simulated launch, surfaced to {@link SimulatedBackgroundSpawnOptions.onLaunch}
 * so callers can assert on what a background task *would* have executed without a
 * real process ever being forked.
 */
export interface SimulatedBackgroundLaunch {
  command: string;
  args: string[];
  cwd: string;
  pid: number;
}

export interface SimulatedBackgroundSpawnOptions {
  /**
   * First fake pid handed out. Subsequent launches increment by one, so pids are
   * deterministic and unique within a runtime. Defaults to 100000 (well above the
   * range a real short-lived process would land in, to make simulated pids obvious
   * in diagnostics).
   */
  basePid?: number;
  /** Invoked for every simulated launch, in order. Useful for assertions/telemetry. */
  onLaunch?: (launch: SimulatedBackgroundLaunch) => void;
}

/**
 * Build a {@link SpawnBackgroundProcess} that never touches the operating system.
 *
 * The default background-task spawner forks a real detached child that
 * asynchronously writes its own execution-state file. In tests and simulations
 * that is pure liability: the child leaks, is slow, and — because the state-file
 * write races the assertions — makes a "running" task intermittently look like a
 * missing process (or clobbers a manually-authored state file). This factory
 * returns a spawner that hands back a deterministic, monotonically increasing fake
 * pid and a no-op child handle, so background-task orchestration can be driven
 * hermetically. The real on-device spawner remains the production default; this is
 * the simulated seam mandated for anything that would otherwise touch the host OS.
 */
export function createSimulatedBackgroundSpawn(
  options: SimulatedBackgroundSpawnOptions = {},
): SpawnBackgroundProcess {
  let nextPid = options.basePid ?? 100_000;
  return (command, args, spawnOptions) => {
    const pid = nextPid++;
    options.onLaunch?.({ command, args, cwd: spawnOptions.cwd, pid });
    return {
      pid,
      unref() {
        /* no-op: a simulated launch has nothing to detach */
      },
    };
  };
}
